"""Build current numeric, measured safety, tail masks and expert caches.

Uses the actual training Dataset and scorer; no candidate-score caching.
All artifacts live under the external cache root. Safe to resume.
"""
import os

# OpenCV otherwise sizes its internal pool from all visible host CPUs in every
# spawned cache worker (176 threads on the current machine).  Set this before
# importing dataset/scoring modules that can import cv2 transitively.
os.environ.setdefault("OPENCV_FOR_THREADS_NUM", "1")

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import precache_experts as experts
from tail_data import DenseDataset, expand_index, collate

STATE = {}


def initialize(root, index, config, output, signature):
    experts.initialize(root, index, config, output, signature)
    # Navigation targets are independently complete and are not part of rule
    # or expert labels. Avoid loading them in 128 cache-generation workers.
    datasets={s: DenseDataset(root, index, s, load_route_targets=False)
              for s in ('train', 'val')}
    source_contract=tuple(str(datasets['train'].__dict__[name])
                          for name in ('cache','safety_cache','tail_cache'))
    STATE.update(datasets=datasets,
                 source_signature=hashlib.sha256(repr(source_contract).encode()).hexdigest())


def work(task):
    split, clip, indices = task
    ds = STATE['datasets'][split]
    agent = experts.STATE['agent']
    done = full = tail = hits = 0
    failures = []
    start = time.time()
    progress = experts.STATE['output'] / 'workers' / f'{os.getpid()}.json'
    for idx in indices:
        frame = ds.entries[idx][1]
        try:
            item = ds[idx]
            if bool(item[1]['rule_score_valid']):
                # collate is shared with training, including expert tensor shape.
                _, targets = collate([item])
                before = agent.__dict__.get('_expert_cache_stats', {}).get('disk_hits', 0)
                row = experts.expert_rows(agent, targets)[0]
                experts.validate(row)
                hits += agent._expert_cache_stats['disk_hits'] - before
                full += 1
            else:
                tail += 1
            done += 1
        except Exception as exc:
            failures.append({'frame': frame, 'error': type(exc).__name__ + ': ' + str(exc)})
        if (done + len(failures)) % 20 == 0:
            experts.atomic_json(progress, dict(pid=os.getpid(), split=split, clip=clip,
                done=done, failed=len(failures), total=len(indices), updated_at=time.time()))
    result = dict(split=split, clip=clip, done=done, total=len(indices), expert_done=full,
                  tail_done=tail, disk_hits=hits, failures=failures, seconds=time.time()-start)
    result['source_signature']=STATE['source_signature']
    result['signature']=experts.STATE['signature']
    experts.atomic_json(experts.STATE['output'] / 'clips' / split / (clip+'.json'), result)
    experts.atomic_json(progress, dict(pid=os.getpid(), idle=True, updated_at=time.time()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 128:
        raise ValueError('Invalid worker count')
    args.output.mkdir(parents=True, exist_ok=True)
    generation_lock = None
    if not args.preflight_only:
        generation_lock = (args.output/'.generation.lock').open('a+')
        print(f'Waiting for cache-generation lock: {generation_lock.name}', flush=True)
        fcntl.flock(generation_lock.fileno(), fcntl.LOCK_EX)
        print(f'Acquired cache-generation lock: {generation_lock.name}', flush=True)
    from audit_ready import verify_acceptance
    root = Path(os.environ['B2D_NEW_CACHE'])
    base = Path(os.environ['B2D_DENSE_INDEX'])
    verify_acceptance(root)
    index = json.loads((base/'index.json').read_text())
    if index['contract']['source'] != json.loads((root/'DATA_READY.json').read_text()):
        raise RuntimeError('Stale source acceptance')
    index = expand_index(index)
    config = Path(__file__).parent.parent/'agent_config.yaml'
    cfg = experts.hydra.utils.instantiate(experts.OmegaConf.load(config).config)
    signature = experts._signature(experts.SimpleNamespace(_config=cfg))
    initialize(root, index, config, args.output, signature)
    source_contract = tuple(str(STATE['datasets']['train'].__dict__[name])
                            for name in ('cache','safety_cache','tail_cache'))
    source_signature = hashlib.sha256(repr(source_contract).encode()).hexdigest()
    if args.preflight_only:
        checks = []
        for split, ds in STATE['datasets'].items():
            clip = index[split][0]
            for i in (0, len(clip['complete_frames'])-1, clip['raw']-6, clip['raw']-1):
                item = ds[i]
                record = dict(split=split, token=item[1]['token'], valid_points=int(item[1]['trajectory_valid'].sum()))
                if bool(item[1]['rule_score_valid']):
                    _, target = collate([item])
                    row = experts.expert_rows(experts.STATE['agent'], target)[0]
                    experts.validate(row)
                    experts.STATE['agent']._expert_label_memory.clear()
                    before = experts.STATE['agent']._expert_cache_stats['disk_hits']
                    experts.expert_rows(experts.STATE['agent'], target)
                    assert experts.STATE['agent']._expert_cache_stats['disk_hits'] == before+1
                    record['training_disk_hit'] = True
                checks.append(record)
        report = dict(status='passed', signature=signature, source_signature=source_signature, checks=checks)
        experts.atomic_json(args.output/'PREFLIGHT.json', report)
        print(json.dumps(report), flush=True)
        return
    pre = json.loads((args.output/'PREFLIGHT.json').read_text())
    if pre['signature'] != signature or pre['source_signature'] != source_signature:
        raise RuntimeError('Run matching preflight first')
    tasks = []
    completed = []
    for split in ('train','val'):
        offset = 0
        for clip in index[split]:
            count = len(clip['frames'])
            result_path = args.output/'clips'/split/(clip['folder']+'.json')
            result = None
            if result_path.exists():
                try: result = json.loads(result_path.read_text())
                except (OSError, json.JSONDecodeError): pass
            if (result is not None and result.get('done') == count
                    and result.get('total') == count and not result.get('failures')
                    and result.get('source_signature') == source_signature
                    # Clip reports written before this field was introduced are
                    # still valid when the matching top-level preflight and
                    # source signatures are present.
                    and result.get('signature', signature) == signature):
                completed.append(result)
            else:
                tasks.append((split, clip['folder'], list(range(offset, offset+count))))
            offset += count
    total_frames = sum(len(c['frames']) for split in ('train','val') for c in index[split])
    report = dict(status='running', pid=os.getpid(), workers=args.workers, signature=signature,
        source_signature=source_signature, total=total_frames,
        expert_total=sum(len(c['complete_frames']) for s in ('train','val') for c in index[s]),
        done=sum(r['done'] for r in completed), failed=0,
        expert_done=sum(r['expert_done'] for r in completed),
        tail_done=sum(r['tail_done'] for r in completed), clips_done=len(completed),
        clips_total=len(tasks)+len(completed), resumed_clips=len(completed), start_time=time.time())
    def save():
        report['elapsed_s'] = time.time()-report['start_time']
        report['frames_per_s'] = report['done']/max(report['elapsed_s'], 1)
        report['updated_at'] = time.time()
        experts.atomic_json(args.output/'PROGRESS.json', report)
    save()
    try:
        with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize, initargs=(root,index,config,args.output,signature)) as pool:
            pending = {pool.submit(work,t) for t in tasks}
            while pending:
                ready,pending = concurrent.futures.wait(pending, timeout=20,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                for future in ready:
                    result = future.result()
                    for field in ('done','expert_done','tail_done'): report[field] += result[field]
                    report['failed'] += len(result['failures'])
                    report['clips_done'] += 1
                save()
                if ready: print(json.dumps(report), flush=True)
        if experts._signature(experts.SimpleNamespace(_config=cfg)) != signature:
            raise RuntimeError('Scoring dependencies changed during generation')
        current_contract = tuple(str(STATE['datasets']['train'].__dict__[name])
                                 for name in ('cache','safety_cache','tail_cache'))
        if hashlib.sha256(repr(current_contract).encode()).hexdigest() != source_signature:
            raise RuntimeError('Dataset code changed during generation')
        if report['failed'] or report['done'] != report['total'] or report['expert_done'] != report['expert_total']:
            raise RuntimeError('Incomplete cache; inspect clip reports')
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
