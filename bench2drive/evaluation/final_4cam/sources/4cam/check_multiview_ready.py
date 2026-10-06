"""Read-only image coverage check in addition to the unchanged label check."""

import json
import os
from pathlib import Path
from camera_contract import CAMERA_ORDER


def validate():
    root = Path(os.environ['B2D_NEW_CACHE']) / 'v1'
    index = json.loads((Path(os.environ['B2D_DENSE_INDEX']) / 'index.json').read_text())
    counts = {name: 0 for name in CAMERA_ORDER}
    for split in ('train', 'val'):
        for clip in index[split]:
            expected = {f'{i:05d}.jpg' for i in range(clip['first'], clip['first'] + clip['raw'])}
            for name in CAMERA_ORDER:
                folder = root / clip['folder'] / 'camera' / ('rgb_' + name)
                found = {p.name for p in os.scandir(folder) if p.is_file() and p.name.endswith('.jpg')}
                missing = expected - found
                if missing:
                    raise RuntimeError(f'{folder}: missing {len(missing)} frames; first={min(missing)}')
                counts[name] += len(expected)
    if any(count != 247656 for count in counts.values()):
        raise RuntimeError(f'Unexpected camera counts: {counts}')
    return dict(status='ready', observed_frames=247656, camera_order=list(CAMERA_ORDER),
                camera_images=counts, total_input_images=sum(counts.values()),
                future_camera='front', future_target_offset_seconds=.5)


if __name__ == '__main__':
    print('MULTIVIEW_READY ' + json.dumps(validate(), sort_keys=True), flush=True)
