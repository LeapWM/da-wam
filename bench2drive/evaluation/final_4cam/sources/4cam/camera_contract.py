"""Ordered four-camera contract, shared with the Python 3.8 CARLA agent."""

CAMERA_ORDER = ('front', 'front_left', 'front_right', 'back')
FUTURE_CAMERA = 'front'
HISTORY_SECONDS = 0.5
IMAGE_SIZE = (256, 512)

# Match the Base annotations' cam2ego, intrinsics and sensor dimensions.
CAMERA_SPECS = {
    'front': dict(x=0.8, y=0.0, z=1.6, yaw=0.0, fov=70.0),
    'front_left': dict(x=0.27, y=-0.55, z=1.6, yaw=-55.0, fov=70.0),
    'front_right': dict(x=0.27, y=0.55, z=1.6, yaw=55.0, fov=70.0),
    'back': dict(x=-2.0, y=0.0, z=1.6, yaw=180.0, fov=110.0),
}


def camera_sensors():
    return [dict(type='sensor.camera.rgb', id=name, width=1600, height=900,
                 roll=0.0, pitch=0.0, **CAMERA_SPECS[name])
            for name in CAMERA_ORDER]


def validate_camera_contract(config):
    if tuple(config.get('camera_order', ())) != CAMERA_ORDER:
        raise ValueError('camera_order must be front, front_left, front_right, back')
    if config.get('future_camera') != FUTURE_CAMERA:
        raise ValueError('Only the front camera has a future prediction target')
