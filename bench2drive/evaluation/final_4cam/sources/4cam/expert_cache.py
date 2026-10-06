"""Content-addressed expert labels; dynamic candidate labels are never cached."""
import hashlib
import json
import os
import pickle
import tempfile
import copy
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch


def _signature(agent):
    root=Path(__file__).parent
    # Model-only variants share deterministic rule labels with the validated
    # TrafficScore task.  Reuse its content identity when every label-affecting
    # source file is byte-identical; any scorer edit falls back to this task's
    # own root and therefore produces a new signature/cache namespace.
    canonical=Path('/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/experiments/Bench2Drive-PostTraj-TrafficScore-10Hz/src')
    deps=Path('/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/geometry/src')
    # Training/model/benchmark edits must not invalidate deterministic labels.
    names=('expert_cache.py','expert_bundle.py','corrected_score.py','safety_labels.py',
           'road_worker.py','traffic_rules.py','metric_geometry.py','horizon_geometry.py','static_collision.py','candidate_geometry_fast.py','collision_reuse.py')
    semantic=tuple(n for n in names if n!='expert_cache.py')
    if canonical.is_dir() and all((root/n).read_bytes()==(canonical/n).read_bytes() for n in semantic):
        signature_root=canonical
    else:
        signature_root=root
    files=[signature_root/n for n in names]+[deps/n for n in ('candidate_geometry.py','training_provider.py','prepared_geometry.py','recorded_geometry.py')]
    from navsim.agents.drive_jepa_perception_based.score_module import compute_b2d_score
    from navsim.agents.drive_jepa_perception_based.ema_jqtf import formula_selection
    files += [Path(compute_b2d_score.__file__),Path(formula_selection.__file__)]
    h=hashlib.sha256()
    for p in sorted(files):h.update(str(p).encode());h.update(p.read_bytes())
    # Track map revisions without re-reading all town maps per sample.
    maps=Path('/mnt/c2-worldmodel/2639639/Bench2Drive/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps')
    paths=list(maps.glob('*/OpenDrive/*.xodr'))+list(maps.glob('OpenDrive/*.xodr'))
    cfg=getattr(agent,'_config',None)
    if cfg is not None and getattr(cfg,'b2d_map_cache_path',None):paths.append(Path(cfg.b2d_map_cache_path))
    for p in sorted(paths):
        stat=p.stat();h.update(str((str(p),stat.st_size,stat.st_mtime_ns)).encode())
    if cfg is not None:
        # Keep the already validated label namespace stable across optimizer
        # and auxiliary-loss scheduling changes. These fields never enter any
        # rule geometry or target calculation; normalize them to the values
        # used when the scalar-speed-v6 expert cache was produced.
        label_cfg=copy.deepcopy(cfg)
        label_cfg.ema_jqtf_formula_scorer_lr_scale=3.0
        label_cfg.ema_jqtf_formula_future_lr_scale=5.0
        label_cfg.future_prediction_weight=0.1
        label_cfg.future_prediction_weight_schedule=()
        h.update(str(label_cfg).encode())
    return h.hexdigest()


def _one(targets,index):
    batch=len(targets['_scorer']);out={}
    for k,v in targets.items():
        if torch.is_tensor(v) and v.ndim and len(v)==batch:out[k]=v[index:index+1]
        elif isinstance(v,(list,tuple)) and len(v)==batch:out[k]=v[index:index+1]
        else:out[k]=v
    return out


def cached_expert_rows(agent,targets,kind,compute):
    """compute(single_sample_targets) returns a small CPU-only result.

    Fingerprint includes actual recorded/scorer payloads and expert trajectory,
    so revised annotations and different experts cannot reuse stale labels.
    B2D_EXPERT_CACHE=off is the reference/diagnostic escape hatch.
    """
    if '_scorer' not in targets or os.environ.get('B2D_EXPERT_CACHE')=='off':
        return None
    if not hasattr(agent,'_expert_label_memory'):
        agent._expert_label_memory=OrderedDict()
        base=Path(os.environ.get('B2D_DENSE_INDEX','/mnt/c2-worldmodel/2639639/Bench2Drive/ema_jqtf_10hz'))
        agent._expert_label_root=base/'expert_labels'/ _signature(agent)
        agent._expert_cache_stats={'memory_hits':0,'disk_hits':0,'misses':0}
    results=[]
    # One batch transfer rather than a synchronization per sample.
    expert_cpu=targets['trajectory'].detach().cpu().numpy()
    for i in range(len(targets['_scorer'])):
        one=_one(targets,i)
        payload=(one['_scorer'][0],one['_safety'][0],expert_cpu[i:i+1])
        key=hashlib.sha256(pickle.dumps(payload,protocol=4)).hexdigest()
        name=kind+'_'+key;memory=agent._expert_label_memory
        if name in memory:
            value=memory.pop(name);memory[name]=value;agent._expert_cache_stats['memory_hits']+=1
        else:
            path=agent._expert_label_root/key[:2]/(name+'.pkl')
            if path.exists():
                with path.open('rb') as f:value=pickle.load(f)
                agent._expert_cache_stats['disk_hits']+=1
            else:
                value=compute(one);path.parent.mkdir(parents=True,exist_ok=True)
                fd,tmp=tempfile.mkstemp(prefix=name+'.',suffix='.tmp',dir=path.parent)
                try:
                    with os.fdopen(fd,'wb') as f:pickle.dump(value,f,protocol=4)
                    os.replace(tmp,path)
                finally:
                    if os.path.exists(tmp):os.unlink(tmp)
                agent._expert_cache_stats['misses']+=1
            memory[name]=value
            if len(memory)>2048:memory.popitem(last=False)
        results.append(value)
    return results
