"""Fixed-image, packed-question d1-2B inference without ML frameworks."""
from __future__ import annotations
import os,pathlib,threading,time
from .questions import validate_questions,validate_state,format_answer

class D1:
    def __init__(self,model_dir,*,ryzen_ai_dir=None,cache_dir=None,threads=4,profile_dir=None):
        from .inputs import Inputs
        from .sessions import cache_directory,language_session
        self.model_dir=pathlib.Path(model_dir).resolve()
        for name in ['model-manifest.json','prompt.py','attention.py','language/model.onnx','vision/vision_encoder_npu.onnx']:
            if not (self.model_dir/name).is_file():raise FileNotFoundError('Model file missing: '+name)
        self.cache=pathlib.Path(cache_dir or cache_directory()).resolve();self.cache.mkdir(parents=True,exist_ok=True)
        self.profile_dir=pathlib.Path(profile_dir).resolve() if profile_dir else None
        if self.profile_dir:self.profile_dir.mkdir(parents=True,exist_ok=True)
        self.encoder=Inputs(self.model_dir);self.vision=None;self.vision_info=None;self.lock=threading.Lock();self.last_timing={}
        self._language_calls=0;self._vision_calls=0
        deploy=pathlib.Path(ryzen_ai_dir or os.environ.get('RYZEN_AI_DIR','C:/Program Files/RyzenAI/1.8.0'))
        if deploy.name.lower()!='deployment':deploy=deploy/'deployment'
        self.language=language_session(self.model_dir/'language',deploy,self.cache/'language',threads,self.profile_dir/'language' if self.profile_dir else None)

    def prepare_vision(self):
        from .sessions import vision_session
        if self.vision is None:self.vision,self.vision_info=vision_session(self.model_dir/'vision'/'vision_encoder_npu.onnx',self.cache,self.profile_dir/'vision' if self.profile_dir else None)

    def decide(self,state,questions,image=None):
        import numpy as np
        from PIL import Image
        from .inputs import pixels
        state=validate_state(state);questions=validate_questions(questions)
        with self.lock:
            started=time.perf_counter();features=np.zeros((0,2048),np.float32);vision_s=0.
            if image is not None:
                self.prepare_vision()
                if isinstance(image,(str,pathlib.Path)):
                    with Image.open(image) as opened:picture=opened.convert('RGB')
                else:picture=image.convert('RGB')
                picture=picture.resize((640,384),Image.Resampling.BICUBIC);raster=pixels(picture)
                stage=time.perf_counter();features=self.vision.run(None,{'pixel_values':raster})[0];vision_s=time.perf_counter()-stage
                self._vision_calls+=1
                if not np.isfinite(features).all():raise ValueError('Nonfinite image features')
            result={};items=list(questions.items());tokens=[]
            for start in range(0,len(items),12):
                group=dict(items[start:start+12]);feeds,typed=self.encoder.prepare(state,group,has_image=image is not None)
                feeds['image_features']=features;tokens.append(feeds['input_ids'].shape[1])
                logits=self.language.run(['logits'],feeds)[0]
                self._language_calls+=1
                if not np.isfinite(logits).all():raise ValueError('Nonfinite decision logits')
                probabilities=self.encoder.probabilities(logits,typed)
                for (key,q),values in zip(group.items(),probabilities):
                    if not np.isfinite(values).all():raise ValueError('Nonfinite decision probabilities')
                    labels=['yes','no'] if q['type']=='noul' else list(q['criteria']) if q['type']=='choice' else [str(i) for i in range(len(q['criteria']))]
                    result[key]=format_answer(q['type'],labels,values,q.get('threshold',.5),q.get('invert',False),q['criteria'] if q['type']=='score' else None)
            self.last_timing={'total_s':time.perf_counter()-started,'vision_s':vision_s,'packed_tokens':tokens}
            return result

    def profiles(self):
        if not self.profile_dir:raise ValueError('Create D1 with profile_dir to collect NPU placement')
        import collections,json
        result={}
        for name,session in [('language',self.language),('vision',self.vision)]:
            if session is None:continue
            path=session.end_profiling();counts=collections.Counter();ops=collections.Counter()
            for event in json.loads(pathlib.Path(path).read_text()):
                args=event.get('args',{});provider=args.get('provider')
                if provider:counts[provider]+=1;ops[provider+'::'+args.get('op_name','')]+=1
            result[name]={'path':path,'events':dict(counts),'operators':dict(ops)}
        if self.vision is not None and result['vision']['events']!={'VitisAIExecutionProvider':self._vision_calls}:raise RuntimeError('Vision CPU fallback or NPU execution count mismatch')
        actual=sum(count for op,count in result['language']['operators'].items() if op.endswith('::MatMulNBitsBf'))
        result['language']['npu_projection_events']={'actual':actual,'expected':143*self._language_calls}
        if actual!=143*self._language_calls or not actual:raise RuntimeError('Language NPU execution count mismatch')
        return result
