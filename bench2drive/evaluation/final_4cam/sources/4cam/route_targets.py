"""Navigation target points derived only from recorded online route commands."""
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

from bench2drive.training.scorer_cache import _transform_points, _world2ego


VERSION = 'b2d_route_command_local_x_forward_v1'
CACHE_SIGNATURE = '497de7b23a95ef3c'


def signature():
    # Stable for this exact generation contract. Reader-only edits must not
    # invalidate already audited data; a generation change requires v2 plus a
    # new signature.
    return CACHE_SIGNATURE


def route_targets(annotation):
    """Return near/far xy in the same x-forward ego frame as GT trajectories.

    x_command_* and y_command_* are RoutePlanner outputs recorded at collection
    time. No future ego pose or expert trajectory is read here.
    """
    ego = next((b for b in annotation.get('bounding_boxes', [])
                if b.get('class') == 'ego_vehicle'), None)
    if ego is None:
        raise ValueError('annotation has no ego_vehicle')
    z = float(np.asarray(ego['location'], dtype=np.float64)[2])
    world2ego = _world2ego(annotation)
    result = {}
    for name in ('near', 'far'):
        keys = (f'x_command_{name}', f'y_command_{name}')
        values = np.asarray([annotation.get(keys[0], np.nan),
                             annotation.get(keys[1], np.nan), z], dtype=np.float64)
        valid = bool(np.isfinite(values).all())
        xy = (_transform_points(world2ego, values)[:2] if valid
              else np.zeros(2, dtype=np.float64))
        valid = valid and bool(np.isfinite(xy).all())
        result[f'{name}_xy'] = xy.astype(np.float32) if valid else np.zeros(2, np.float32)
        result[f'{name}_valid'] = valid
        command = int(annotation.get(f'command_{name}', 4))
        result[f'{name}_command'] = command if 1 <= command <= 6 else 4
    return result


class RouteTargetCache:
    """Small per-worker LRU over versioned per-clip target arrays."""
    def __init__(self, root=None, max_clips=64):
        base = Path(root or os.environ['B2D_DENSE_INDEX'])
        self.root = base/'route_targets'/signature()
        progress = json.loads((self.root/'PROGRESS.json').read_text())
        if (progress.get('status') != 'complete' or
                progress.get('frames_done') != progress.get('frames_total')):
            raise RuntimeError(f'route-target cache is not complete: {self.root}')
        self.max_clips = int(max_clips)
        self.memory = OrderedDict()

    def _clip(self, split, clip):
        key = (split, clip)
        if key in self.memory:
            value = self.memory.pop(key)
            self.memory[key] = value
            return value
        path = self.root/split/(clip+'.npz')
        with np.load(path, allow_pickle=False) as data:
            if data['version'].item() != VERSION:
                raise RuntimeError(f'route-target version mismatch: {path}')
            value = {name: data[name].copy() for name in
                     ('frame_ids','near_xy','far_xy','near_valid','far_valid',
                      'near_command','far_command')}
        self.memory[key] = value
        if len(self.memory) > self.max_clips:
            self.memory.popitem(last=False)
        return value

    def get(self, split, clip, frame):
        data = self._clip(split, clip)
        index = int(np.searchsorted(data['frame_ids'], frame))
        if index == len(data['frame_ids']) or int(data['frame_ids'][index]) != int(frame):
            raise KeyError(f'missing route target {split}/{clip}/{frame:05d}')
        return {name: data[name][index] for name in
                ('near_xy','far_xy','near_valid','far_valid','near_command','far_command')}
