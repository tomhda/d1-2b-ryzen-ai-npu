"""Run two repeated decisions on a generated, non-private example image."""
import argparse,json,os,pathlib,sys,time
from PIL import Image,ImageDraw
from d1npu2 import D1

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--cache-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
picture=Image.new('RGB',(800,480),'white');draw=ImageDraw.Draw(picture)
draw.rectangle((0,0,800,70),fill='#244b74');draw.text((30,25),'EXAMPLE PORTAL',fill='white')
draw.text((280,120),'Sign in',fill='black');draw.text((220,180),'Email',fill='black');draw.rectangle((220,200,580,240),outline='gray',width=2)
draw.text((220,260),'Password',fill='black');draw.rectangle((220,280,580,320),outline='gray',width=2)
draw.rectangle((220,350,580,390),fill='#244b74');draw.text((370,365),'Continue',fill='white')
picture.save(out.with_suffix('.png'))
questions={'login':{'type':'noul','instructions':'Does the image show a sign-in form?'},
 'kind':{'type':'choice','instructions':'What kind of page is shown?','criteria':{'sign_in':'A sign-in page','article':'A news article','map':'A map'}},
 'urgency':{'type':'score','instructions':'How urgent is the message?','criteria':['low','medium','high']}}
t=time.perf_counter();engine=D1(a.model,cache_dir=a.cache_dir,profile_dir=out.parent/(out.stem+'-profiles'))
engine.prepare_vision();ready=time.perf_counter()-t;print('READY',ready,flush=True)
answers=[]
for _ in range(2):answers.append(engine.decide('Inspect this example sign-in page.',questions,picture))
if answers[0]!=answers[1]:raise ValueError('Unstable repeated answers')
text=engine.decide('Please refund the duplicate charge.',{'refund':{'type':'noul','instructions':'Is the customer asking for a refund?'}})
report={'status':'ok','ready_s':ready,'vision':engine.vision_info,'answers':answers,'text_only':text,'profiles':engine.profiles(),
 'frameworks_imported':{name:name in sys.modules for name in ['torch','transformers','torchvision']}}
if any(report['frameworks_imported'].values()):raise ValueError('Heavy ML framework imported')
out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(report,indent=2),flush=True)
sys.stdout.flush();os._exit(0)
