"""Download and run the fixed-image d1-2B model."""
import argparse,contextlib,json,os,pathlib,sys,traceback
REPO_ID='Tomoali/d1-2B-ryzen-ai-npu'
NATIVE_STARTED=False

@contextlib.contextmanager
def native_output_to_stderr():
    """Keep native provider diagnostics separate from the JSON result."""
    sys.stdout.flush();saved=os.dup(1)
    windows=None;previous_handle=None
    try:
        os.dup2(2,1)
        if os.name=='nt':
            import ctypes,msvcrt
            windows=ctypes.WinDLL('kernel32',use_last_error=True)
            windows.GetStdHandle.argtypes=[ctypes.c_uint32];windows.GetStdHandle.restype=ctypes.c_void_p
            windows.SetStdHandle.argtypes=[ctypes.c_uint32,ctypes.c_void_p];windows.SetStdHandle.restype=ctypes.c_int
            previous_handle=windows.GetStdHandle(0xFFFFFFF5)
            if not windows.SetStdHandle(0xFFFFFFF5,msvcrt.get_osfhandle(2)):raise ctypes.WinError(ctypes.get_last_error())
        yield
    finally:
        sys.stdout.flush();os.dup2(saved,1);os.close(saved)
        if windows is not None:windows.SetStdHandle(0xFFFFFFF5,previous_handle)

def main(argv=None):
    global NATIVE_STARTED
    p=argparse.ArgumentParser(prog='d1npu2');sub=p.add_subparsers(dest='command',required=True)
    download=sub.add_parser('download');download.add_argument('--to',required=True);download.add_argument('--revision')
    for name in ['decide','screen-check']:
        parser=sub.add_parser(name);parser.add_argument('--model',required=True);parser.add_argument('--image')
        parser.add_argument('--state',default='Inspect the provided browser screenshot.')
        if name=='decide':parser.add_argument('--questions',required=True)
        parser.add_argument('--cache-dir');parser.add_argument('--ryzen-ai-dir');parser.add_argument('--profile-dir');parser.add_argument('--output')
    a=p.parse_args(argv)
    if a.command=='download':
        from huggingface_hub import snapshot_download
        options={'local_dir':a.to}
        if a.revision:options['revision']=a.revision
        snapshot_download(REPO_ID,**options);print('Downloaded '+REPO_ID,flush=True);return False
    from .engine import D1
    if a.command=='screen-check':
        if not a.image:p.error('screen-check requires --image')
        questions=json.loads((pathlib.Path(__file__).parent/'screen_check.json').read_text(encoding='utf8'))
    else:questions=json.loads(pathlib.Path(a.questions).read_text(encoding='utf8'))
    state=pathlib.Path(a.state[1:]).read_text(encoding='utf8') if a.state.startswith('@') else a.state
    from .questions import validate_questions,validate_state
    validate_questions(questions);validate_state(state)
    if a.image and not pathlib.Path(a.image).is_file():raise FileNotFoundError(a.image)
    if a.output and not pathlib.Path(a.output).parent.is_dir():raise FileNotFoundError('Output directory does not exist')
    NATIVE_STARTED=True
    with native_output_to_stderr():
        engine=D1(a.model,cache_dir=a.cache_dir,ryzen_ai_dir=a.ryzen_ai_dir,profile_dir=a.profile_dir)
        result=engine.decide(state,questions,a.image)
        if a.profile_dir:
            profiles=engine.profiles();(pathlib.Path(a.profile_dir)/'placement.json').write_text(json.dumps(profiles,indent=2),encoding='utf8')
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    if a.output:pathlib.Path(a.output).write_text(text+'\n',encoding='utf8')
    print(text,flush=True);return True

def entrypoint():
    try:native=main()
    except BaseException as error:
        if not NATIVE_STARTED:raise
        traceback.print_exception(error,file=sys.stderr);sys.stdout.flush();sys.stderr.flush();os._exit(1)
    sys.stdout.flush();sys.stderr.flush()
    # Ryzen AI native providers can fail during Python interpreter shutdown.
    if native:os._exit(0)
