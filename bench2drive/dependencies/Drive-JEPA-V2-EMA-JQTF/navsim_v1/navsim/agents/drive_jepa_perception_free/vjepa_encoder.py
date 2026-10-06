"""V-JEPA 2.0 / 2.1 encoder loaders for Drive-JEPA perception-free."""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Union

import torch
import yaml

logger = logging.getLogger(__name__)

Resolution = Union[int, Tuple[int, int]]

VJEPA20_MODEL_DICT: Dict[str, Dict[str, object]] = {
    "vit_large": {"filename": "vitl.pt", "dim": 1024, "model_name": "vit_large", "checkpoint_key": "target_encoder"},
    "vit_huge": {"filename": "vith.pt", "dim": 1280, "model_name": "vit_huge", "checkpoint_key": "target_encoder"},
    "vit_giant_xformers": {
        "filename": "vitg.pt",
        "dim": 1408,
        "model_name": "vit_giant_xformers",
        "checkpoint_key": "target_encoder",
    },
}

VJEPA21_MODEL_DICT: Dict[str, Dict[str, object]] = {
    "vjepa2_1_vit_base_384": {
        "filename": "vjepa2_1_vitb_dist_vitG_384.pt",
        "dim": 768,
        "model_name": "vit_base",
        "checkpoint_key": "ema_encoder",
    },
    "vjepa2_1_vit_large_384": {
        "filename": "vjepa2_1_vitl_dist_vitG_384.pt",
        "dim": 1024,
        "model_name": "vit_large",
        "checkpoint_key": "ema_encoder",
    },
    "vjepa2_1_vit_giant_384": {
        "filename": "vjepa2_1_vitg_384.pt",
        "dim": 1408,
        "model_name": "vit_giant_xformers",
        "checkpoint_key": "target_encoder",
    },
    "vjepa2_1_vit_gigantic_384": {
        "filename": "vjepa2_1_vitG_384.pt",
        "dim": 1664,
        "model_name": "vit_gigantic_xformers",
        "checkpoint_key": "target_encoder",
    },
}

VJEPA20_CONFIG = "./vjepa2/configs/eval/vitl/in1k.yaml"
VJEPA21_CONFIG = "./vjepa2/configs/eval_2_1/vitl-384/in1k.yaml"


def normalize_vjepa_version(vjepa_version) -> str:
    """Hydra may pass 2.0/2.1 as float; normalize to '2.0' / '2.1'."""
    if isinstance(vjepa_version, float):
        if vjepa_version.is_integer():
            return f"{int(vjepa_version)}.0"
        return str(vjepa_version)
    if isinstance(vjepa_version, int):
        return f"{vjepa_version}.0"
    version = str(vjepa_version).strip()
    if version in {"2", "2.0"}:
        return "2.0"
    if version == "2.1":
        return "2.1"
    return version


def get_model_spec(vjepa_version: str, image_architecture: str) -> Dict[str, object]:
    vjepa_version = normalize_vjepa_version(vjepa_version)
    registry = VJEPA21_MODEL_DICT if vjepa_version == "2.1" else VJEPA20_MODEL_DICT
    if image_architecture not in registry:
        supported = ", ".join(sorted(registry))
        raise ValueError(
            f"Unsupported image_architecture={image_architecture!r} for vjepa_version={vjepa_version!r}. "
            f"Supported: {supported}"
        )
    return registry[image_architecture]


def get_embed_dim(vjepa_version: str, image_architecture: str) -> int:
    return int(get_model_spec(vjepa_version, image_architecture)["dim"])


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def _load_checkpoint_state(checkpoint_path: str, checkpoint_key: str) -> Dict[str, torch.Tensor]:
    logger.info("Loading pretrained model from checkpoint=%r key=%r", checkpoint_path, checkpoint_key)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint_key not in checkpoint:
        available = [k for k in checkpoint.keys() if not isinstance(checkpoint[k], (int, float, str))]
        raise KeyError(
            f"Checkpoint key {checkpoint_key!r} not found in {checkpoint_path}. "
            f"Available tensor keys: {available}"
        )
    return _clean_state_dict(checkpoint[checkpoint_key])


def _load_state_dict_flexible(model: torch.nn.Module, pretrained_dict: Dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    for key, value in model_state.items():
        if key not in pretrained_dict:
            logger.info('key "%s" could not be found in loaded state dict', key)
        elif pretrained_dict[key].shape != value.shape:
            logger.info(
                'key "%s" is of different shape in model and loaded state dict: %s vs %s',
                key,
                tuple(pretrained_dict[key].shape),
                tuple(value.shape),
            )
            pretrained_dict[key] = value
    msg = model.load_state_dict(pretrained_dict, strict=False)
    logger.info("loaded pretrained model with msg: %s", msg)


def init_vjepa20_encoder(
    resolution: Resolution,
    checkpoint: str,
    image_architecture: str,
    num_frames: int = 2,
    register_prehook: bool = False,
    config_path: str = VJEPA20_CONFIG,
) -> torch.nn.Module:
    import src.models.vision_transformer as vit

    with open(config_path, "r") as yaml_file:
        params = yaml.load(yaml_file, Loader=yaml.FullLoader)

    model_kwargs = params["model_kwargs"]["pretrain_kwargs"]
    enc_kwargs = dict(model_kwargs["encoder"])
    enc_kwargs["model_name"] = get_model_spec("2.0", image_architecture)["model_name"]
    checkpoint_key = str(enc_kwargs.get("checkpoint_key", "target_encoder"))

    model = vit.__dict__[str(enc_kwargs["model_name"])](
        input_size=resolution,
        num_frames=num_frames,
        **enc_kwargs,
    )

    if register_prehook:
        def forward_prehook(module, input_tensor):
            tensor = input_tensor[0]
            tensor = tensor.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)
            return tensor

        model.register_forward_pre_hook(forward_prehook)

    pretrained_dict = _load_checkpoint_state(checkpoint, checkpoint_key)
    _load_state_dict_flexible(model, pretrained_dict)
    return model


def init_vjepa21_encoder(
    resolution: Resolution,
    checkpoint: str,
    image_architecture: str,
    num_frames: int = 2,
) -> torch.nn.Module:
    import app.vjepa_2_1.models.vision_transformer as vit21

    spec = get_model_spec("2.1", image_architecture)
    model_name = str(spec["model_name"])
    checkpoint_key = str(spec["checkpoint_key"])

    model = vit21.__dict__[model_name](
        img_size=resolution,
        num_frames=num_frames,
        patch_size=16,
        tubelet_size=2,
        use_sdpa=True,
        use_rope=True,
        uniform_power=False,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )

    pretrained_dict = _load_checkpoint_state(checkpoint, checkpoint_key)
    _load_state_dict_flexible(model, pretrained_dict)
    return model


def load_vjepa_encoder(
    vjepa_version: str,
    resolution: Resolution,
    checkpoint: str,
    image_architecture: str,
    num_frames: int = 2,
    register_prehook: bool = False,
) -> torch.nn.Module:
    vjepa_version = normalize_vjepa_version(vjepa_version)
    if vjepa_version == "2.1":
        return init_vjepa21_encoder(
            resolution=resolution,
            checkpoint=checkpoint,
            image_architecture=image_architecture,
            num_frames=num_frames,
        )
    if vjepa_version == "2.0":
        return init_vjepa20_encoder(
            resolution=resolution,
            checkpoint=checkpoint,
            image_architecture=image_architecture,
            num_frames=num_frames,
            register_prehook=register_prehook,
        )
    raise ValueError(f"Unsupported vjepa_version={vjepa_version!r}. Expected '2.0' or '2.1'.")
