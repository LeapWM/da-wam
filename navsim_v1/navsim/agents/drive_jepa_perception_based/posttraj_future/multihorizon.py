"""Mean, time-conditioned EMA supervision with an unchanged inference scorer."""
import torch
import torch.nn.functional as F
from .target_pair import build_current_future_pair


def multihorizon_loss(model, pred, proposals, target_trajectory, future_cameras,
                      offsets, normalize=True):
    offsets = tuple(offsets)
    if offsets not in ((1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)):
        raise ValueError("expected contiguous future offsets 1..H, H<=4")
    if future_cameras.shape[1] != len(offsets):
        raise ValueError("camera count does not match requested offsets")
    previous = pred["future_target_current_frame"]
    losses = []
    full_candidate_latents = pred.get("candidate_full_latents")
    if full_candidate_latents is not None and offsets != (1,):
        raise ValueError(
            "full-candidate latent supervision requires offsets=(1,)"
        )
    for j, offset in enumerate(offsets):
        frame = future_cameras[:, j].to(proposals.device)
        if frame.dtype == torch.uint8:
            frame = frame.float().div(255.0)
        elif not frame.is_floating_point():
            frame = frame.float()
        frame = model.transform(frame)
        with torch.no_grad():
            target = model.encode_future_target(build_current_future_pair(previous, frame))
        if full_candidate_latents is None:
            predicted, index, distance = model.predict_matched_future_latent(
                pred["future_scene_map"], pred["candidate_future_tokens"],
                proposals, target_trajectory, horizon_offset=offset)
        else:
            index, distance = model._future_predictor.match_proposals(
                proposals, target_trajectory
            )
            batch_index = torch.arange(
                proposals.shape[0], device=proposals.device
            )
            predicted = full_candidate_latents[batch_index, index]
        if normalize:
            channels = predicted.shape[1]
            predicted = F.layer_norm(predicted.permute(0, 2, 3, 1), (channels,)).permute(0, 3, 1, 2)
            target = F.layer_norm(target.permute(0, 2, 3, 1), (channels,)).permute(0, 3, 1, 2)
        losses.append(F.l1_loss(predicted, target.detach()))
        previous = frame
    # Fixed total weight in the caller, independent of horizon count.
    return torch.stack(losses).mean(), index, distance, losses
