"""CPU-only expert-label generation from accepted numeric/safety caches.

No model, images, invented future states or candidate-score caching. Resume
reuses content-addressed records; each run audits every requested frame.
"""
import os

# This must precede imports that can load cv2.  Without it every spawned cache
# worker tries to create a host-sized OpenCV thread pool.
os.environ.setdefault("OPENCV_FOR_THREADS_NUM", "1")

import argparse,atexit,concurrent.futures,gzip,hashlib,json,multiprocessing,pickle,time,warnings
from pathlib import Path
from types import SimpleNamespace
import hydra,numpy as np,torch
from omegaconf import OmegaConf
from safety_data import DenseDataset,collate
from safety_labels import Labels
from expert_bundle import expert_rows
from expert_cache import _signature

STATE={}


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def initialize(root,index,config,output,signature):
    warnings.filterwarnings('ignore', message='invalid value encountered in line_locate_point', category=RuntimeWarning)
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    cfg=hydra.utils.instantiate(OmegaConf.load(config).config)
    agent=SimpleNamespace(_config=cfg,_fast_final_only=True,traffic_labels=Labels(traffic_only=True))
    # Road legality and traffic-rule requests are sequential and use the same
    # protocol. Share one map subprocess instead of loading two CARLA maps per
    # cache worker; this does not cache or approximate any returned label.
    agent.road_metric_worker=agent.traffic_labels.road
    if _signature(agent)!=signature:raise RuntimeError('Cache version changed during launch')
    STATE.update(agent=agent,datasets={s:DenseDataset(root,index,s) for s in ('train','val')},
                 output=Path(output),signature=signature)
    def close():
        agent.traffic_labels.road.close()
    atexit.register(close)


def read_targets(split,clip,frame):
    ds=STATE['datasets'][split]
    with gzip.open(ds.cache/clip/f'{frame:05d}.pkl.gz','rb') as f:sample,_,_=pickle.load(f)
    with gzip.open(ds.safety_cache/split/clip/f'{frame:05d}.pkl.gz','rb') as f:prepared=pickle.load(f)
    if sample['clip']!=clip or int(sample['frame_id'])!=frame:raise ValueError('numeric sample identity mismatch')
    return {'_scorer':[sample],'_safety':[prepared],'token':[sample['token']],
            'trajectory':torch.as_tensor(sample['trajectory'],dtype=torch.float32)[None]}


def validate(row):
    if len(row['checks'][1])!=4:raise ValueError('Bad expert checks')
    if not np.isfinite(row['base'][0]):raise ValueError('Nonfinite base score')
    for key in ('aux_value','aux_valid'):
        if row['traffic'][key].shape!=(1,1,3):raise ValueError('Bad expert traffic shape')
    if not np.isfinite(row['traffic']['aux_value']).all():raise ValueError('Nonfinite traffic score')


def work(task):
    split,clip,frames=task;start=time.monotonic();done=0;failures=[]
    agent=STATE['agent'];before=dict(getattr(agent,'_expert_cache_stats',{}))
    progress=STATE['output']/'workers'/f'{os.getpid()}.json'
    for frame in frames:
        try:
            row=expert_rows(agent,read_targets(split,clip,frame))[0];validate(row);done+=1
        except Exception as exc:
            failures.append({'frame':frame,'error':type(exc).__name__+': '+str(exc)})
        if (done+len(failures))%20==0:
            atomic_json(progress,{'pid':os.getpid(),'split':split,'clip':clip,'done':done,
                                  'failed':len(failures),'total':len(frames),'time':time.time()})
    stats={k:v-before.get(k,0) for k,v in getattr(agent,'_expert_cache_stats',{}).items()}
    result={'split':split,'clip':clip,'done':done,'total':len(frames),'failures':failures,
            'seconds':time.monotonic()-start,'cache_stats':stats,'signature':STATE['signature']}
    atomic_json(STATE['output']/'clips'/split/(clip+'.json'),result)
    atomic_json(progress,{'pid':os.getpid(),'idle':True,'time':time.time()})
    return result


def key(targets):
    return hashlib.sha256(pickle.dumps((targets['_scorer'][0],targets['_safety'][0],
                         targets['trajectory'].numpy()),protocol=4)).hexdigest()


def preflight(index):
    checks=[]
    for split in ('train','val'):
        ds=STATE['datasets'][split]
        for i in (0,len(ds)//2):
            clip,frame,_=ds.entries[i];fast=read_targets(split,clip,frame)
            _,training=collate([ds[i]])
            if key(fast)!=key(training):raise RuntimeError('Training/offline cache key mismatch')
            agent=STATE['agent'];row=expert_rows(agent,fast)[0];validate(row)
            agent._expert_label_memory.clear()
            hits=agent._expert_cache_stats['disk_hits']
            loaded=expert_rows(agent,training)[0]
            if agent._expert_cache_stats['disk_hits']!=hits+1:raise RuntimeError('Training lookup missed offline record')
            if pickle.dumps(row,protocol=4)!=pickle.dumps(loaded,protocol=4):raise RuntimeError('Record mismatch')
            checks.append({'split':split,'token':training['token'][0],'key_equal':True,'training_disk_hit':True})
    return checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=32)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--preflight-only',action='store_true')
    args=ap.parse_args()
    if not 1<=args.workers<=128:raise ValueError('workers must be between 1 and 128')
    if os.environ.get('B2D_EXPERT_CACHE')=='off' or os.environ.get('B2D_EXPERT_BUNDLE')=='off':raise ValueError('Expert cache/bundle must be enabled')
    root=Path(os.environ['B2D_NEW_CACHE']);base=Path(os.environ['B2D_DENSE_INDEX'])
    from audit_ready import verify_acceptance
    verify_acceptance(root)
    index=json.loads((base/'index.json').read_text())
    if index['contract']['source']!=json.loads((root/'DATA_READY.json').read_text()):raise RuntimeError('Stale dense index')
    from tail_data import expand_index
    index=expand_index(index)
    for split in ('train','val'):
        for clip in index[split]:clip['frames']=clip['complete_frames']
    config=Path(__file__).parent.parent/'agent_config.yaml'
    cfg=hydra.utils.instantiate(OmegaConf.load(config).config)
    signature=_signature(SimpleNamespace(_config=cfg))
    args.output.mkdir(parents=True,exist_ok=True)
    initialize(root,index,config,args.output,signature)
    if args.preflight_only:
        report={'status':'passed','signature':signature,'checks':preflight(index)}
        atomic_json(args.output/'PREFLIGHT.json',report);print(json.dumps(report));return
    pre=args.output/'PREFLIGHT.json'
    if not pre.exists() or json.loads(pre.read_text()).get('signature')!=signature:
        raise RuntimeError('Run --preflight-only for this version first')
    # Fail early on missing source caches instead of rebuilding them with images.
    tasks=[]
    for split in ('train','val'):
        for clip in index[split]:
            frames=clip['frames']
            if len(frames)!=len(set(frames)):raise ValueError('Duplicate index frames')
            tasks.append((split,clip['folder'],frames))
    start=time.time();total=sum(len(t[2]) for t in tasks)
    report={'status':'running','pid':os.getpid(),'workers':args.workers,'signature':signature,
            'cache_root':str(base/'expert_labels'/signature),'total':total,'done':0,'failed':0,
            'clips_done':0,'clips_total':len(tasks),'start_time':start,'cache_stats':{}}
    def save():
        elapsed=time.time()-start;report['elapsed_s']=elapsed
        report['frames_per_s']=report['done']/max(elapsed,1)
        report['eta_s']=(total-report['done']-report['failed'])/report['frames_per_s'] if report['done'] else None
        report['updated_at']=time.time();atomic_json(args.output/'PROGRESS.json',report)
    save()
    try:
        with concurrent.futures.ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize,initargs=(root,index,config,args.output,signature)) as pool:
            pending={pool.submit(work,t) for t in tasks}
            while pending:
                ready,pending=concurrent.futures.wait(pending,timeout=20,return_when=concurrent.futures.FIRST_COMPLETED)
                for f in ready:
                    result=f.result();report['done']+=result['done'];report['failed']+=len(result['failures']);report['clips_done']+=1
                    for k,v in result['cache_stats'].items():report['cache_stats'][k]=report['cache_stats'].get(k,0)+v
                save()
                if ready:print(json.dumps(report),flush=True)
        if _signature(SimpleNamespace(_config=cfg))!=signature:raise RuntimeError('Scoring dependencies changed during run')
        report['status']='complete' if report['done']==total and not report['failed'] else 'failed'
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc));save();raise
    save()
    if report['status']!='complete':raise RuntimeError('Incomplete cache: see per-clip failures')


if __name__=='__main__':main()
