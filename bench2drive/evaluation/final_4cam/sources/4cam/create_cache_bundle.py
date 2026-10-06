"""Create one versioned entry point for the current external caches."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import hydra
from omegaconf import OmegaConf

from expert_cache import _signature
from precache_experts import atomic_json
from route_targets import signature as route_signature
from tail_data import DenseDataset, expand_index


def add_link(directory, name, target):
    path = directory/name
    if path.is_symlink():
        if path.resolve() != target.resolve():
            raise RuntimeError(f'{path} points to another cache version')
    elif path.exists():
        raise RuntimeError(f'{path} is not a symlink')
    else:
        path.symlink_to(target, target_is_directory=True)


def main():
    root = Path(os.environ['B2D_NEW_CACHE'])
    base = Path(os.environ['B2D_DENSE_INDEX'])
    generation = Path(os.environ.get(
        'CACHE_OUTPUT', base/'audits/full_3s_scalar_speed_128_20260922'
    ))
    bundle = base/'bundles/traffic_3s_scalar_speed_v6'
    bundle.mkdir(parents=True, exist_ok=True)
    index = expand_index(json.loads((base/'index.json').read_text()))
    dataset = DenseDataset(root, index, 'train')
    config = Path(__file__).parent.parent/'agent_config.yaml'
    cfg = hydra.utils.instantiate(OmegaConf.load(config).config)
    signature = _signature(SimpleNamespace(_config=cfg))
    route_root = base/'route_targets'/route_signature()
    components = {
        'observations': root/'v1',
        'numeric': dataset.cache,
        'safety': dataset.safety_cache,
        'tail_numeric': dataset.tail_cache,
        'expert_labels': base/'expert_labels'/signature,
        'route_targets': route_root,
        'generation': generation,
    }
    for name, target in components.items():
        add_link(bundle, name, target)
    route_progress = {}
    if (route_root/'PROGRESS.json').exists():
        route_progress = json.loads((route_root/'PROGRESS.json').read_text())
    atomic_json(bundle/'manifest.json', {
        'schema_version': 2,
        'name': bundle.name,
        'data_root': str(root),
        'dense_index_root': str(base),
        'expert_signature': signature,
        'future_seconds': 3.0,
        'future_dt': 0.5,
        'observation_hz': 10,
        'collision_dt': 0.5,
        'ego_status': {
            'dimension': 12,
            'layout': 'pose_zero3 + scalar_speed + acceleration2 + command6',
            'velocity_mode': 'scalar_speed',
        },
        'counts': {'observations': 247656, 'complete_expert': 217656, 'partial_tail': 30000},
        'components': {name: str(target) for name, target in components.items()},
        'navigation_target_xy': {
            'status': route_progress.get('status', 'not_generated'),
            'cache_signature': route_signature(),
            'frames_done': route_progress.get('frames_done', 0),
            'near_valid': route_progress.get('near_valid', 0),
            'far_valid': route_progress.get('far_valid', 0),
            'model_default': 'near_xy',
            'source_requirement': 'current navigation route; never expert future',
        },
        'progress_file': str(generation/'PROGRESS.json'),
        'ready_policy': 'progress status complete, zero failures, and all counts match',
    })
    print(bundle)


if __name__ == '__main__':
    main()
