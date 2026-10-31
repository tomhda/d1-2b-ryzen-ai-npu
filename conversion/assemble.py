"""Assemble a portable model folder from converted weights and checkpoint files."""
import argparse,hashlib,json,pathlib,shutil
import onnx

def assemble(source,language,vision,transactions,output):
    source,language,vision,transactions,output=map(lambda p:pathlib.Path(p).resolve(),(source,language,vision,transactions,output))
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    (output/'language'/'cache').mkdir(parents=True);(output/'vision').mkdir()
    for name in ['prompt.py','media.py','calibration.py','attention.py','calibration.json','tokenizer.json','tokenizer_config.json','config.json','processor_config.json','chat_template.jinja','LICENSE']:
        shutil.copyfile(source/name,output/name)
    for name in ['NOTICE','NOTICE.md','NOTICE.txt']:
        if (source/name).exists():shutil.copyfile(source/name,output/('UPSTREAM_'+name))
    model=onnx.load(language/'model.onnx',load_external_data=False)
    locations={x.value for tensor in model.graph.initializer for x in tensor.external_data if x.key=='location'}
    for name in locations:
        if pathlib.PurePosixPath(name).name!=name:raise ValueError('External weights must be sibling files')
        shutil.copyfile(language/name,output/'language'/name)
    model.doc_string='Converted and shape-corrected for Ryzen AI NPU; 4-bit dense projections, FP32 attention gates, 4096-token context.'
    onnx.save(model,output/'language'/'model.onnx')
    for name in ['model.pb.bin','rai_config.json']:shutil.copyfile(language/name,output/'language'/name)
    config=json.loads((language/'genai_config.json').read_text(encoding='utf8'))
    config['model']['decoder']['session_options']['hybrid_opt_max_seq_length']='4096'
    config['model']['decoder']['session_options'].pop('custom_ops_library',None)
    (output/'language'/'genai_config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    shutil.copyfile(transactions,output/'language'/'cache'/'txn_bins.zip')
    image=onnx.load(vision,load_external_data=False);sources=set()
    for tensor in image.graph.initializer:
        for item in tensor.external_data:
            if item.key=='location':sources.add(item.value);item.value='vision_encoder_npu.onnx_data'
    if len(sources)!=1:raise ValueError('Expected one external vision weight file')
    shutil.copyfile(vision.parent/next(iter(sources)),output/'vision'/'vision_encoder_npu.onnx_data')
    image.doc_string='Fixed 640x384 LFM3-VL vision encoder for Ryzen AI NPU; decomposed attention/rotary and mathematically equivalent 2048x2 projector; original GELU approximation preserved.'
    onnx.save(image,output/'vision'/'vision_encoder_npu.onnx')
    notice='''Derived from LiquidAI/d1-2B and LiquidAI/d1-2B-ONNX, under the LFM Open License v1.0 (LICENSE).

Modified files:
- language/model.onnx and external weights: dense projections quantized to 4-bit and converted for Ryzen AI NPU; FP32 attention gates; dynamic answer-slot/media dimensions corrected. Embedding is 8-bit.
- language/genai_config.json: portable runtime paths and 4096-token context.
- vision/vision_encoder_npu.onnx: fixed 640x384 shapes, attention/rotary decomposition, equivalent 2048x2 projector split with original GELU attributes, NPU-compatible layouts; external data location renamed.
- vision/vision_encoder_npu.onnx_data: original floating-point vision weights rearranged for the graph.
- model-manifest.json: model layout and file checksums.

prompt.py, media.py, calibration.py, attention.py and checkpoint text/configuration files are retained from the upstream ONNX checkpoint. No AMD SDK DLLs or machine-specific compiled vision cache/EPContext are included.
'''
    (output/'NOTICE').write_text(notice,encoding='utf8')
    files=[]
    for path in sorted(output.rglob('*')):
        if path.is_file():
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            files.append({'path':path.relative_to(output).as_posix(),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
    (output/'model-manifest.json').write_text(json.dumps({'format_version':1,'context_length':4096,'image_size':[640,384],'files':files},indent=2),encoding='utf8')
    print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','language','vision','transactions','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();assemble(a.source,a.language,a.vision,a.transactions,a.output)
