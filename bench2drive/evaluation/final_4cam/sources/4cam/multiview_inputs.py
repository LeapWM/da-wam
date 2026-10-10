"""Online observations use the exact training JPEG preprocessing and order."""

import numpy as np
import torch
from camera_contract import CAMERA_ORDER, validate_camera_contract
from bench2drive.model_server import _bench2drive_front_lidar_to_image, _normalize_ego_status, _normalize_target_point
from bench2drive.training.da_wam_cache import preprocess_front_jpeg


def build_online_features(request, device, expected_ego_status_dim, feature_config):
    feature_config = dict(feature_config or {})
    validate_camera_contract(feature_config)
    if feature_config.get('profile') != 'bench2drive' or expected_ego_status_dim != 12:
        raise ValueError('Four-camera recipe requires B2D scalar-speed 12D ego state')
    if tuple(request.get('camera_order', ())) != CAMERA_ORDER:
        raise ValueError('Inference request camera order does not match training')
    features = {}
    for key, field in (('camera_feature_1', 'current_jpegs'),
                       ('camera_feature_2', 'previous_jpegs')):
        packet = request[field]
        if set(packet) != set(CAMERA_ORDER):
            raise ValueError('Every requested camera must have a real JPEG')
        features[key] = torch.stack([preprocess_front_jpeg(packet[name])
                                     for name in CAMERA_ORDER])[None].to(device)
    status = _normalize_ego_status(request['ego_status'], expected_ego_status_dim)
    point, valid = _normalize_target_point(request)
    projection = np.repeat(_bench2drive_front_lidar_to_image()[None], 4, axis=0)
    features.update(
        ego_status=torch.from_numpy(status).view(1, 1, 12).to(device),
        target_point=torch.from_numpy(point).view(1, 2).to(device),
        target_point_valid=torch.tensor([valid], device=device),
        lidar2img=torch.from_numpy(projection)[None].to(device),
        img_shape=torch.tensor([[[256., 512., 3.]] * 4], device=device))
    return features
