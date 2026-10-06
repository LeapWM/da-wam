import torch
import torch.nn as nn
from einops import rearrange

from navsim.agents.drive_jepa_perception_free.vjepa_encoder import (
    load_vjepa_encoder,
    get_embed_dim,
)
from navsim.agents.drive_jepa_perception_free.vjepa_lora import LoRA_VJEPA

from .grid_mask import GridMask
from ..drive_jepa_config import DriveJEPAConfig

class ImgEncoder(nn.Module):
    def __init__(self, config: DriveJEPAConfig, num_feature_levels=2):
        super().__init__()
        self.embed_dims = config.tf_d_model
        self.num_feature_levels = num_feature_levels
        num_cams = 1

        self.num_cams = num_cams
        self.use_lidar=False

        self.grid_mask = GridMask( True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = True

        resolution = (256, 512)

        # Load backbone via PF's load_vjepa_encoder
        self.img_backbone = load_vjepa_encoder(
            vjepa_version=config.vjepa_version,
            resolution=resolution,
            checkpoint=config.pretrain_pt_path,
            image_architecture=config.image_architecture,
            num_frames=2,
            register_prehook=False,
        )

        # Apply LoRA if enabled
        if config.use_lora:
            self.img_backbone = LoRA_VJEPA(self.img_backbone, r=config.lora_rank).lora_vit

        # Freeze backbone if freeze_encoder (already frozen by LoRA_VJEPA if use_lora)
        if config.freeze_encoder and not config.use_lora:
            for param in self.img_backbone.parameters():
                param.requires_grad = False

        backbone_dim = get_embed_dim(config.vjepa_version, config.image_architecture)
        self.projector = nn.Linear(backbone_dim, self.embed_dims)

    def forward(self,img,len_queue=None,**kwargs):

        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img = rearrange(img, '(B N) C H W -> B C N H W', B=B)
        img_feat = self.img_backbone(img)#7,12
        img_feat = self.projector(img_feat)
        img_feat = rearrange(img_feat, 'B (H W) C -> B C H W', H=16, W=32)

        BN, C, H, W = img_feat.size()
        feat = img_feat.view(B, int(BN / B), C, H, W)

        bs, num_cam, c, h, w = feat.shape#1,6,256,12,20
        spatial_shape = (h, w)
        feat = feat.flatten(3).permute(1, 0, 3, 2)#6,1,240,256
        #feat = feat +lidar2img_embed[:,:,None]

        spatial_shape = torch.as_tensor(
            [spatial_shape], dtype=torch.long, device=feat.device)
        level_start_index = torch.cat((spatial_shape.new_zeros(
            (1,)), spatial_shape.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        return feat_flatten, spatial_shape, level_start_index,kwargs
