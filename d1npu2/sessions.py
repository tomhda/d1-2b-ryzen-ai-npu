"""Ryzen AI providers and locally generated vision contexts."""
import hashlib,json,os,pathlib,sys
HANDLES=[]

def cache_directory():
    return pathlib.Path(os.environ.get('LOCALAPPDATA',pathlib.Path.home()/'.cache'))/'d1npu2'

def vision_fingerprint(model):
    model=pathlib.Path(model);weights=model.parent/'vision_encoder_npu.onnx_data';stat=weights.stat()
    manifest=json.loads((model.parent.parent/'model-manifest.json').read_text(encoding='utf8'))
    item=next(v for v in manifest['files'] if v['path']=='vision/vision_encoder_npu.onnx_data')
    if stat.st_size!=item['bytes']:raise ValueError('Vision weight size does not match the model manifest')
    return {'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'weights_bytes':stat.st_size,
            'weights_mtime_ns':stat.st_mtime_ns,'manifest_weights_sha256':item['sha256']}

def context_files(context):return [context,context.with_name(context.name+'_VITISAI.bin')]

def file_sha256(path):
    digest=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def complete_context(root,identity):
    """Atomically mark a successfully created local context and its external bin."""
    root=pathlib.Path(root);files=context_files(root/'ep'/'vision.onnx')
    if any(not file.is_file() or file.stat().st_size==0 for file in files):raise RuntimeError('EPContext export did not produce the ONNX and external bin')
    record={'format_version':1,'identity':identity,'files':{file.relative_to(root).as_posix():hashlib.sha256(file.read_bytes()).hexdigest() for file in files}}
    temporary=root/'complete.json.tmp';temporary.write_text(json.dumps(record,indent=2),encoding='utf8');temporary.replace(root/'complete.json')

def language_session(model,deployment,cache,threads=4,profile=None):
    import onnxruntime as ort
    model,deployment,cache=map(pathlib.Path,(model,deployment,cache));dll=deployment/'onnxruntime_providers_ryzenai.dll'
    if not dll.is_file():raise FileNotFoundError('Install Ryzen AI Software 1.8.0 or set --ryzen-ai-dir')
    HANDLES.append(os.add_dll_directory(str(deployment)));os.environ['PATH']=str(deployment)+os.pathsep+os.environ['PATH']
    ort.register_execution_provider_library('RyzenAILightExecutionProvider',str(dll))
    # The provider can also list the integrated GPU; keep the language model on the NPU.
    devices=[d for d in ort.get_ep_devices() if d.ep_name=='RyzenAILightExecutionProvider' and d.device.type==ort.OrtHardwareDeviceType.NPU]
    if not devices:raise RuntimeError('Ryzen AI NPU execution provider unavailable')
    so=ort.SessionOptions();so.intra_op_num_threads=threads;so.inter_op_num_threads=1
    so.enable_cpu_mem_arena=False;so.enable_mem_pattern=False;so.add_provider_for_devices(devices,{})
    if profile:so.enable_profiling=True;so.profile_file_prefix=str(profile)
    config=json.loads((model/'genai_config.json').read_text(encoding='utf8'))['model']['decoder']['session_options']
    config={**config,'hybrid_opt_max_seq_length':'4096','external_data_file':str(model/config['external_data_file'])}
    for key,value in config.items():
        if key not in ('provider_options','custom_ops_library'):so.add_session_config_entry(key,str(value))
    for key in ['session.intra_op.allow_spinning','session.inter_op.allow_spinning']:so.add_session_config_entry(key,'0')
    cache.mkdir(parents=True,exist_ok=True)
    # Generated transaction bins are distributed with the language model.
    so.add_session_config_entry('dd_cache',str(model/'cache'));so.register_custom_ops_library(str(dll))
    return ort.InferenceSession(str(model/'model.onnx'),so)

def vision_session(model,cache,profile=None):
    import onnxruntime as ort
    model=pathlib.Path(model).resolve();cache=pathlib.Path(cache).resolve()
    if 'VitisAIExecutionProvider' not in ort.get_available_providers():raise RuntimeError('VitisAI NPU provider unavailable')
    identity=vision_fingerprint(model);key=identity['model_sha256'][:20];root=cache/'vision'/key
    had_state=root.exists() and any(root.iterdir());marker=root/'complete.json'
    direct=marker.is_file()
    if direct:
        record=json.loads(marker.read_text(encoding='utf8'))
        expected={'ep/vision.onnx','ep/vision.onnx_VITISAI.bin'}
        if record.get('format_version')!=1 or set(record.get('files',{}))!=expected:raise RuntimeError('Incomplete EPContext marker; use a new --cache-dir. Existing cache was retained.')
        if record.get('identity')!=identity:
            previous_identity=record.get('identity',{})
            changed={name for name in set(previous_identity)|set(identity) if previous_identity.get(name)!=identity.get(name)}
            if changed=={'weights_mtime_ns'} and file_sha256(model.parent/'vision_encoder_npu.onnx_data')==identity['manifest_weights_sha256']:
                record['identity']=identity;temporary=root/'complete.json.tmp'
                temporary.write_text(json.dumps(record,indent=2),encoding='utf8');temporary.replace(marker)
            else:raise RuntimeError('Vision weights changed; use a new --cache-dir. Existing cache was retained.')
        for name,digest in record['files'].items():
            file=root/name
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:raise RuntimeError('Incomplete or changed EPContext; use a new --cache-dir. Existing cache was retained.')
    elif had_state:raise RuntimeError('Incomplete vision cache without a completion marker; use a new --cache-dir. Existing cache was retained.')
    root.mkdir(parents=True,exist_ok=True);context=root/'ep'/'vision.onnx';context.parent.mkdir(exist_ok=True)
    so=ort.SessionOptions();so.intra_op_num_threads=1;so.inter_op_num_threads=1
    so.enable_cpu_mem_arena=False;so.enable_mem_pattern=False
    for name in ['session.intra_op.allow_spinning','session.inter_op.allow_spinning']:so.add_session_config_entry(name,'0')
    if profile:so.enable_profiling=True;so.profile_file_prefix=str(profile)
    if not direct:
        print('Compiling the vision encoder for the NPU; first run can take about 10 minutes.',file=sys.stderr,flush=True)
        so.add_session_config_entry('ep.context_enable','1');so.add_session_config_entry('ep.context_file_path',str(context));so.add_session_config_entry('ep.context_embed_mode','0')
    previous=os.getcwd()
    try:
        os.chdir(root)
        session=ort.InferenceSession(str(context if direct else model),so,providers=['VitisAIExecutionProvider'],provider_options=[{'cache_dir':str(root),'cache_key':'vision','enable_cache_file_io_in_mem':0}])
    finally:os.chdir(previous)
    if 'VitisAIExecutionProvider' not in session.get_providers():raise RuntimeError('Vision NPU provider failed to load')
    if not direct:complete_context(root,identity)
    return session,{'context':str(context),'direct_context_load':direct,'cache':str(root)}
