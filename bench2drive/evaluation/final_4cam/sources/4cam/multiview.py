"""Fuse four observations onto a front-view grid; predict front futures only."""

import torch
from torch import nn
from camera_contract import CAMERA_ORDER


class FrontGridFusion(nn.Module):
    """Front-grid queries attend to all four spatial maps with camera IDs.

    This is image-context fusion, not a calibrated four-view BEV projection.
    The downstream geometric sampler keeps the real front calibration.
    """

    def __init__(self, dim=256, heads=8, ffn_dim=1024, spatial_hw=(16, 32), dropout=0.):
        super().__init__()
        self.dim = dim
        self.spatial_hw = tuple(spatial_hw)
        tokens = spatial_hw[0] * spatial_hw[1]
        self.camera_embedding = nn.Parameter(torch.empty(1, len(CAMERA_ORDER), 1, dim))
        self.spatial_embedding = nn.Parameter(torch.empty(1, 1, tokens, dim))
        nn.init.trunc_normal_(self.camera_embedding, std=.02)
        nn.init.trunc_normal_(self.spatial_embedding, std=.02)
        self.memory_norm = nn.LayerNorm(dim)
        self.decoder = nn.TransformerDecoderLayer(
            dim, heads, ffn_dim, dropout, activation='gelu',
            batch_first=True, norm_first=True)

    def forward(self, maps):
        if maps.ndim != 5 or tuple(maps.shape[1:]) != (4, self.dim, *self.spatial_hw):
            raise ValueError('Expected [B,4,D,H,W] in the fixed camera order')
        tokens = maps.flatten(3).transpose(2, 3)
        positioned = tokens + self.camera_embedding + self.spatial_embedding
        memory = self.memory_norm(positioned.flatten(1, 2))
        queries = tokens[:, 0] + self.spatial_embedding[:, 0]
        fused = self.decoder(queries, memory)
        return fused.transpose(1, 2).reshape(maps.shape[0], self.dim, *self.spatial_hw)


def encode_four_cameras(model, features):
    """Share the online backbone; keep the EMA backbone's input single-view."""
    current = features['camera_feature_1']
    previous = features['camera_feature_2']
    if current.ndim != 5 or current.shape[1:] != (4, 3, 256, 512):
        raise ValueError('Four-camera model requires [B,4,3,256,512] observations')
    if previous.shape != current.shape:
        raise ValueError('All cameras require matching current/history frames')
    current = model.transform(model._camera_to_float(current))
    previous = model.transform(model._camera_to_float(previous))
    # Encode [t-0.5,t] as a two-frame clip for each view, never as eight time steps.
    batch = current.shape[0]
    pairs = torch.stack((previous, current), dim=2).flatten(0, 1)
    encoded = model._backbone(pairs, img_metas={})
    projected = encoded[4].reshape(batch, 4, 256, 16, 32)
    fused = model.multiview_fusion(projected)
    front_context = encoded[5].reshape(batch, 4, 1024, 16, 32)[:, 0]
    metadata = dict(features)
    # Existing caches repeat front calibration four times. It is used only for
    # sampling the fused front grid, never as another camera's calibration.
    from input_projection import corrected_front_projection
    metadata['lidar2img'] = corrected_front_projection(features['lidar2img'][:, :1])
    metadata['img_shape'] = features['img_shape'][:, :1]
    flattened = fused.flatten(2).permute(2, 0, 1).unsqueeze(0)
    image_features = (flattened, encoded[1], encoded[2], {'img_metas': metadata},
                      fused, front_context)
    return torch.stack((previous[:, 0], current[:, 0]), dim=1), current[:, 0], image_features
