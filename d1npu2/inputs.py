"""Exact fixed-image input construction using the checkpoint's pure prompt code."""
import ast,importlib,json,pathlib,sys,types
import numpy as np
from tokenizers import Tokenizer



class Tokens:
    def __init__(self,source):
        SRC=pathlib.Path(source)
        self.backend=Tokenizer.from_file(str(SRC/'tokenizer.json'))
        cfg=json.loads((SRC/'tokenizer_config.json').read_text(encoding='utf-8'))
        model=json.loads((SRC/'config.json').read_text(encoding='utf-8'))
        self.bos_token=cfg['bos_token']
        for field,default in [('d1_codes',1),('d1_numbers',False),('d1_prompt',1),('d1_box_edges','median')]:setattr(self,field,model.get(field,default))
    def encode(self,text,add_special_tokens=False):return self.backend.encode(text,add_special_tokens=add_special_tokens).ids
    def decode(self,ids,**kwargs):return self.backend.decode(list(ids),skip_special_tokens=kwargs.get('skip_special_tokens',False))
    def convert_tokens_to_ids(self,text):return self.backend.token_to_id(text)

class Inputs:
    def __init__(self,source):
        SRC=pathlib.Path(source).resolve()
        package=types.ModuleType('d1checkpoint_'+__import__('hashlib').sha256(str(SRC).encode()).hexdigest()[:12]);package.__path__=[str(SRC)];sys.modules[package.__name__]=package
        self.prompt=importlib.import_module(package.__name__+'.prompt')
        calibration=importlib.import_module(package.__name__+'.calibration')
        self.calibration=calibration.Calibration.load(SRC/'calibration.json');self.tokenizer=Tokens(SRC)
        # Only these two functions are pure packing logic. Importing attention.py
        # itself would load Torch and transformer model implementations.
        tree=ast.parse((SRC/'attention.py').read_text(encoding='utf-8'))
        nodes=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
        nodes.extend(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('state_length','pack_set'))
        code=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]));namespace={}
        exec(compile(code,str(SRC/'attention.py'),'exec'),namespace);self.pack_set=namespace['pack_set']

    def prepare(self,state,questions,has_image=True):
        prompt=self.prompt;tok=self.tokenizer;qs=[prompt.as_question(q) for q in questions.values()]
        if len(qs)>16:raise ValueError('This image runtime supports at most 16 packed questions per set')
        markup=('<|vision_start|>'+'<|image|>'*240+'<|vision_end|>') if has_image else ''
        prefix=prompt.prefix_text(tok,state,tok.bos_token,prompt.DEFAULT_STATE_STYLE,prompt.DEFAULT_SYSTEM,markup)
        layouts=[prompt.suffix_layout(tok,q,'<think></think>','desc') for q in qs]
        suffixes=[v[0] for v in layouts];offsets=[v[1] for v in layouts]
        ids,state_length,question,ends=self.pack_set(tok,prefix,suffixes,prompt.question_header(tok,state),offsets)
        if len(ids)>4096:raise ValueError('Input exceeds 4096 tokens')
        positions=[p for own in ends for p in own]
        if len(positions)!=len(qs):raise ValueError('Only single-slot question types are supported')
        feeds={'input_ids':np.array([ids],np.int64),'attention_mask':np.ones((1,len(ids)),np.int64),
               'position_ids':np.arange(len(ids),dtype=np.int64)[None],
               'question_ids':np.array([question],np.int64),'slot_positions':np.array(positions,np.int64),
               'video_features':np.zeros((0,2048),np.float32)}
        return feeds,qs

    def probabilities(self,logits,questions):
        # Same float32 log-softmax and unchanged checkpoint readout/calibration.
        maximum=logits.max(axis=-1,keepdims=True)
        logsum=maximum+np.log(np.exp(logits-maximum).sum(axis=-1,keepdims=True,dtype=np.float32))
        normalized=logits-logsum
        return [self.prompt.readout(self.tokenizer,q,row,self.calibration) for q,row in zip(questions,normalized)]

def pixels(image):
    if image.size!=(640,384):raise ValueError('Pre-resize the image to 640x384')
    value=np.asarray(image.convert('RGB'),dtype=np.uint8).transpose(2,0,1).astype(np.float32)
    value-=np.float32(127.5);value/=np.float32(127.5)
    patches=value.reshape(1,3,12,2,16,20,2,16).transpose(0,2,5,3,6,1,4,7)
    patches=np.repeat(np.expand_dims(patches,6),2,axis=6)
    blocks=patches.reshape(12,20,2,2,1536)
    return blocks.transpose(0,2,1,3,4).reshape(1,960,1536).copy()
