"""Build small per-clip route-target caches for every observed 10 Hz frame."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from precache_experts import atomic_json
from route_targets import VERSION, route_targets, signature
from tail_data import expand_index


def build_clip(task):
    root, output, split, clip, frames = task
    root, output = Path(root), Path(output)
    destination = output/split/(clip+'.npz')
    if destination.exists():
        try:
            with np.load(destination, allow_pickle=False) as old:
                if old['version'].item() == VERSION and np.array_equal(old['frame_ids'], frames):
                    return {'split': split, 'clip': clip, 'frames': len(frames),
                            'near_valid': int(old['near_valid'].sum()),
                            'far_valid': int(old['far_valid'].sum()), 'reused': True}
        except (OSError, ValueError, KeyError):
            pass
    near = np.zeros((len(frames), 2), np.float32)
    far = np.zeros((len(frames), 2), np.float32)
    near_valid = np.zeros(len(frames), bool)
    far_valid = np.zeros(len(frames), bool)
    near_command = np.zeros(len(frames), np.int8)
    far_command = np.zeros(len(frames), np.int8)
    source = hashlib.sha256()
    for i, frame in enumerate(frames):
        path = root/'v1'/clip/'anno'/f'{frame:05d}.json.gz'
        raw = path.read_bytes()
        source.update(frame.to_bytes(8, 'little', signed=True))
        source.update(hashlib.sha256(raw).digest())
        row = route_targets(json.loads(gzip.decompress(raw)))
        near[i], far[i] = row['near_xy'], row['far_xy']
        near_valid[i], far_valid[i] = row['near_valid'], row['far_valid']
        near_command[i], far_command[i] = row['near_command'], row['far_command']
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(temporary, version=np.asarray(VERSION), frame_ids=np.asarray(frames),
        near_xy=near, far_xy=far, near_valid=near_valid, far_valid=far_valid,
        near_command=near_command, far_command=far_command,
        source_sha256=np.asarray(source.hexdigest()))
    os.replace(temporary, destination)
    return {'split': split, 'clip': clip, 'frames': len(frames),
            'near_valid': int(near_valid.sum()), 'far_valid': int(far_valid.sum()), 'reused': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        raise ValueError('workers must be in [1, 64]')
    root = Path(os.environ['B2D_NEW_CACHE'])
    base = Path(os.environ['B2D_DENSE_INDEX'])
    index = expand_index(json.loads((base/'index.json').read_text()))
    output = args.output or base/'route_targets'/signature()
    output.mkdir(parents=True, exist_ok=True)
    tasks = [(str(root), str(output), split, clip['folder'], clip['frames'])
             for split in ('train','val') for clip in index[split]]
    report = {'status': 'running', 'version': VERSION, 'signature': signature(),
              'workers': args.workers, 'clips_total': len(tasks), 'clips_done': 0,
              'frames_total': sum(len(t[4]) for t in tasks), 'frames_done': 0,
              'near_valid': 0, 'far_valid': 0, 'reused_clips': 0,
              'start_time': time.time()}
    def save():
        report['elapsed_s'] = time.time()-report['start_time']
        report['frames_per_s'] = report['frames_done']/max(report['elapsed_s'], 1e-6)
        report['updated_at'] = time.time()
        atomic_json(output/'PROGRESS.json', report)
    save()
    try:
        with concurrent.futures.ProcessPoolExecutor(args.workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            for result in pool.map(build_clip, tasks, chunksize=1):
                report['clips_done'] += 1
                report['frames_done'] += result['frames']
                report['near_valid'] += result['near_valid']
                report['far_valid'] += result['far_valid']
                report['reused_clips'] += int(result['reused'])
                if report['clips_done'] % 20 == 0: save()
        if report['frames_done'] != report['frames_total']:
            raise RuntimeError('route-target cache is incomplete')
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        save()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
