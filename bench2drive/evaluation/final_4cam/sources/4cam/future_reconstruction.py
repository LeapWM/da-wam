"""Keep reconstruction active when its contribution to total loss is zero."""

import torch
import torch.nn.functional as F
from navsim.agents.drive_jepa_perception_based.ema_jqtf import build_future_target_camera_pair


def complete_future_reconstruction(agent, targets, pred):
    """Match the positive-weight parent's reconstruction for complete samples.

    The shared parent skips this calculation at weight zero. This local recipe
    explicitly keeps both the front decoder and front EMA teacher running.
    Partial-future rows are handled separately by TailTrafficAgent.
    """
    cfg = agent._config
    if (tuple(cfg.future_frame_offsets) != (1,)
            or cfg.future_prediction_type != 'l1'
            or cfg.future_target_pair_mode != 'current_future'
            or cfg.future_integration != 'ema_jqtf'):
        raise ValueError('Expected the Native256 single-front-future L1 contract')
    model = agent._pad_model
    predicted, matched_index, matched_distance = model.predict_matched_future_latents(
        pred['future_context_latent'], pred['candidate_future_tokens'],
        pred['proposals'], targets['trajectory'])
    if len(predicted) != 1:
        raise ValueError('Expected exactly one front future latent')
    predicted = predicted[0]
    with torch.no_grad():
        frame = model.transform(model._camera_to_float(targets['future_camera_features'][:, 0]))
        pair = build_future_target_camera_pair(pred['future_target_current_frame'], frame,
                                               cfg.future_target_pair_mode)
        target = model.encode_future_target(pair)
        if cfg.future_target_layernorm:
            target = F.layer_norm(target.permute(0, 2, 3, 1),
                                  (target.shape[1],)).permute(0, 3, 1, 2)
    if cfg.future_target_layernorm:
        predicted = F.layer_norm(predicted.permute(0, 2, 3, 1),
                                 (predicted.shape[1],)).permute(0, 3, 1, 2)
    pred['future_matched_index'] = matched_index
    pred['future_matched_distance'] = matched_distance.detach()
    return F.l1_loss(predicted, target)
