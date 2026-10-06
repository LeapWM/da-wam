"""Add side/rear observations without changing any rule or future label cache."""

import torch
from camera_contract import CAMERA_ORDER
from tail_data import DenseDataset as FrontDataset, collate, expand_index
from bench2drive.training.drive_jepa_cache import preprocess_front_jpeg


class DenseDataset(FrontDataset):
    def __getitem__(self, index):
        features, targets = super().__getitem__(index)
        idx = index[0] if isinstance(index, tuple) else index
        clip, frame, first = self.entries[idx]
        folder = self.root / 'v1' / clip / 'camera'
        for key, number in (('camera_feature_1', frame),
                            ('camera_feature_2', max(first, frame - 5))):
            images = [features[key]]
            for name in CAMERA_ORDER[1:]:
                path = folder / ('rgb_' + name) / f'{number:05d}.jpg'
                images.append(preprocess_front_jpeg(path.read_bytes()))
            features[key] = torch.stack(images)
        # targets['future_camera_features'] remains [1,3,256,512]: front t+0.5.
        return features, targets
