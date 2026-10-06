"""Deterministic local-hard and balanced sampling from the 8192 NAVSIM anchors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


ANCHOR_SCORE_HEADS = ("noc", "dac", "ep", "ttc", "comfort", "final")
EXPECTED_ANCHOR_SCORE_SHAPE = (8192, len(ANCHOR_SCORE_HEADS))


@dataclass(frozen=True)
class HardAnchorSamplerConfig:
    num_local_hard: int = 64
    num_gt_local_hard: int = 0
    num_balanced: int = 64
    nearest_per_proposal: int = 64
    nearest_per_gt: int = 64
    min_score_gap: float = 0.2
    yaw_weight: float = 0.5
    hardness_eps: float = 1e-3
    high_final_min: float = 0.95
    safe_metric_min: float = 0.95
    medium_final_min: float = 0.3
    medium_final_max: float = 0.8
    failure_metric_max: float = 0.5
    seed: int = 20260715


@dataclass(frozen=True)
class AnchorSample:
    indices: np.ndarray
    nearest_proposal_indices: np.ndarray
    distances: np.ndarray
    score_gaps: np.ndarray
    is_local_hard: np.ndarray
    is_gt_local_hard: np.ndarray
    buckets: np.ndarray

    @property
    def local_hard_count(self) -> int:
        return int(self.is_local_hard.sum())

    @property
    def gt_local_hard_count(self) -> int:
        return int(self.is_gt_local_hard.sum())


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def trajectory_distance_matrix(
    proposals: np.ndarray,
    anchors: np.ndarray,
    yaw_weight: float = 0.5,
) -> np.ndarray:
    """Return non-negative [P,A] pose distance."""
    proposals = np.asarray(proposals, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)
    if proposals.ndim != 3 or proposals.shape[-1] != 3:
        raise ValueError(f"proposals must be [P,T,3], got {proposals.shape}")
    if anchors.ndim != 3 or anchors.shape[-1] != 3:
        raise ValueError(f"anchors must be [A,T,3], got {anchors.shape}")
    if proposals.shape[1] != anchors.shape[1]:
        raise ValueError(
            f"proposal/anchor time dimensions differ: {proposals.shape[1]} vs {anchors.shape[1]}"
        )

    xy_delta = anchors[None, :, :, :2] - proposals[:, None, :, :2]
    mean_xy_l2 = np.linalg.norm(xy_delta, axis=-1).mean(axis=-1)
    yaw_delta = wrap_angle(
        anchors[None, :, :, 2] - proposals[:, None, :, 2]
    )
    mean_yaw_abs = np.abs(yaw_delta).mean(axis=-1)
    return mean_xy_l2 + float(yaw_weight) * mean_yaw_abs


def _validate_inputs(
    proposals: np.ndarray,
    proposal_pdms: np.ndarray,
    anchors: np.ndarray,
    anchor_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proposals = np.asarray(proposals, dtype=np.float32)
    proposal_pdms = np.asarray(proposal_pdms, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)
    anchor_scores = np.asarray(anchor_scores, dtype=np.float32)
    if proposals.ndim != 3 or proposals.shape[-1] != 3:
        raise ValueError(f"proposals must be [P,T,3], got {proposals.shape}")
    if proposal_pdms.shape != (proposals.shape[0],):
        raise ValueError(
            f"proposal_pdms must be [{proposals.shape[0]}], got {proposal_pdms.shape}"
        )
    if anchors.ndim != 3 or anchors.shape[1:] != proposals.shape[1:]:
        raise ValueError(
            f"anchors must be [A,{proposals.shape[1]},{proposals.shape[2]}], got {anchors.shape}"
        )
    if anchor_scores.shape != (anchors.shape[0], 6):
        raise ValueError(
            f"anchor_scores must be [{anchors.shape[0]},6], got {anchor_scores.shape}"
        )
    for name, value in (
        ("proposals", proposals),
        ("proposal_pdms", proposal_pdms),
        ("anchors", anchors),
        ("anchor_scores", anchor_scores),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    return proposals, proposal_pdms, anchors, anchor_scores


def _nearest_indices(distances: np.ndarray, count: int) -> np.ndarray:
    count = min(int(count), distances.shape[1])
    if count <= 0:
        raise ValueError("nearest trajectory count must be positive")
    partial = np.argpartition(distances, kth=count - 1, axis=1)[:, :count]
    ordered = np.empty_like(partial)
    for proposal_index, anchor_indices in enumerate(partial):
        order = np.lexsort(
            (anchor_indices, distances[proposal_index, anchor_indices])
        )
        ordered[proposal_index] = anchor_indices[order]
    return ordered


def _balanced_quotas(count: int) -> np.ndarray:
    # high-safe, medium, NOC failure, DAC failure, TTC failure, random
    weights = np.asarray([2, 2, 1, 1, 1, 1], dtype=np.int64)
    raw = count * weights / weights.sum()
    quotas = np.floor(raw).astype(np.int64)
    for index in np.argsort(-(raw - quotas), kind="stable")[: count - quotas.sum()]:
        quotas[index] += 1
    return quotas


class HardAnchorSampler:
    def __init__(
        self,
        anchors: np.ndarray,
        config: Optional[HardAnchorSamplerConfig] = None,
    ) -> None:
        self.anchors = np.asarray(anchors, dtype=np.float32)
        if self.anchors.ndim != 3 or self.anchors.shape[-1] != 3:
            raise ValueError(f"anchors must be [A,T,3], got {self.anchors.shape}")
        self.config = config or HardAnchorSamplerConfig()

    def _sample_balanced(
        self,
        anchor_scores: np.ndarray,
        count: int,
        excluded: Sequence[int],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        final = anchor_scores[:, 5]
        available = np.ones(len(anchor_scores), dtype=bool)
        if len(excluded):
            available[np.asarray(excluded, dtype=np.int64)] = False

        masks = (
            (final >= cfg.high_final_min)
            & (anchor_scores[:, 0] >= cfg.safe_metric_min)
            & (anchor_scores[:, 1] >= cfg.safe_metric_min)
            & (anchor_scores[:, 3] >= cfg.safe_metric_min),
            (final >= cfg.medium_final_min) & (final < cfg.medium_final_max),
            anchor_scores[:, 0] <= cfg.failure_metric_max,
            anchor_scores[:, 1] <= cfg.failure_metric_max,
            anchor_scores[:, 3] <= cfg.failure_metric_max,
            np.ones(len(anchor_scores), dtype=bool),
        )
        names = np.asarray(
            ["high_safe", "medium", "noc_failure", "dac_failure", "ttc_failure", "random"]
        )
        selected = []
        labels = []
        for mask, name, quota in zip(masks, names, _balanced_quotas(count)):
            candidates = np.flatnonzero(mask & available)
            take = min(int(quota), len(candidates))
            if take:
                chosen = rng.choice(candidates, size=take, replace=False)
                selected.extend(chosen.tolist())
                labels.extend([name] * take)
                available[chosen] = False

        remaining = count - len(selected)
        if remaining:
            candidates = np.flatnonzero(available)
            if len(candidates) < remaining:
                raise RuntimeError(
                    f"Cannot sample {count} unique balanced anchors; only "
                    f"{len(selected) + len(candidates)} are available"
                )
            chosen = rng.choice(candidates, size=remaining, replace=False)
            selected.extend(chosen.tolist())
            labels.extend(["random_fill"] * remaining)

        return np.asarray(selected, dtype=np.int64), np.asarray(labels)

    def sample(
        self,
        proposals: np.ndarray,
        proposal_pdms: np.ndarray,
        anchor_scores: np.ndarray,
        seed: Optional[int] = None,
        gt_trajectory: Optional[np.ndarray] = None,
        gt_pdms: Optional[float] = None,
    ) -> AnchorSample:
        proposals, proposal_pdms, anchors, anchor_scores = _validate_inputs(
            proposals, proposal_pdms, self.anchors, anchor_scores
        )
        cfg = self.config
        if cfg.num_gt_local_hard < 0:
            raise ValueError("num_gt_local_hard must be non-negative")
        if cfg.num_gt_local_hard > cfg.num_local_hard:
            raise ValueError("num_gt_local_hard cannot exceed num_local_hard")
        use_gt = cfg.num_gt_local_hard > 0
        if use_gt:
            if gt_trajectory is None or gt_pdms is None:
                raise ValueError(
                    "gt_trajectory and gt_pdms are required when num_gt_local_hard > 0"
                )
            gt_trajectory = np.asarray(gt_trajectory, dtype=np.float32)
            if gt_trajectory.shape != proposals.shape[1:]:
                raise ValueError(
                    f"gt_trajectory must be {proposals.shape[1:]}, got {gt_trajectory.shape}"
                )
            if not np.isfinite(gt_trajectory).all():
                raise ValueError("gt_trajectory contains non-finite values")
            gt_pdms = float(gt_pdms)
            if not np.isfinite(gt_pdms) or not 0.0 <= gt_pdms <= 1.0:
                raise ValueError(f"gt_pdms must be finite and in [0,1], got {gt_pdms}")

        distances = trajectory_distance_matrix(
            proposals, anchors, yaw_weight=cfg.yaw_weight
        )
        nearest = _nearest_indices(distances, cfg.nearest_per_proposal)
        proposal_indices = np.repeat(
            np.arange(len(proposals), dtype=np.int64), nearest.shape[1]
        )
        anchor_indices = nearest.reshape(-1)
        pair_distances = distances[proposal_indices, anchor_indices]
        pair_gaps = np.abs(
            anchor_scores[anchor_indices, 5] - proposal_pdms[proposal_indices]
        )
        eligible = pair_gaps >= cfg.min_score_gap
        denominator = pair_distances + cfg.hardness_eps
        near_zero = np.abs(denominator) < cfg.hardness_eps
        denominator[near_zero] = np.where(
            denominator[near_zero] < 0, -cfg.hardness_eps, cfg.hardness_eps
        )
        hardness = pair_gaps / denominator

        candidates = np.flatnonzero(eligible)
        order = candidates[
            np.lexsort(
                (
                    anchor_indices[candidates],
                    proposal_indices[candidates],
                    pair_distances[candidates],
                    -hardness[candidates],
                )
            )
        ]
        hard_indices = []
        hard_proposals = []
        hard_distances = []
        hard_gaps = []
        hard_is_gt = []
        seen = set()

        # Reserve GT-local candidates first. The online GT PDMS is used only
        # for sampling. External scoring still gathers the nearest generated
        # slot feature because the model has no separate GT proposal slot.
        if use_gt and cfg.num_local_hard > 0:
            gt_distances = trajectory_distance_matrix(
                gt_trajectory[None], anchors, yaw_weight=cfg.yaw_weight
            )[0]
            gt_nearest = _nearest_indices(
                gt_distances[None], cfg.nearest_per_gt
            )[0]
            gt_gaps = np.abs(anchor_scores[gt_nearest, 5] - gt_pdms)
            gt_eligible = gt_gaps >= cfg.min_score_gap
            gt_hardness = gt_gaps / (gt_distances[gt_nearest] + cfg.hardness_eps)
            gt_candidates = np.flatnonzero(gt_eligible)
            gt_order = gt_candidates[
                np.lexsort(
                    (
                        gt_nearest[gt_candidates],
                        gt_distances[gt_nearest[gt_candidates]],
                        -gt_hardness[gt_candidates],
                    )
                )
            ]
            for candidate_index in gt_order:
                anchor_index = int(gt_nearest[candidate_index])
                if anchor_index in seen:
                    continue
                seen.add(anchor_index)
                hard_indices.append(anchor_index)
                hard_proposals.append(int(np.argmin(distances[:, anchor_index])))
                hard_distances.append(float(gt_distances[anchor_index]))
                hard_gaps.append(float(gt_gaps[candidate_index]))
                hard_is_gt.append(True)
                if len(hard_indices) == cfg.num_gt_local_hard:
                    break

        if cfg.num_local_hard > 0:
            for pair_index in order:
                anchor_index = int(anchor_indices[pair_index])
                if anchor_index in seen:
                    continue
                seen.add(anchor_index)
                hard_indices.append(anchor_index)
                hard_proposals.append(int(proposal_indices[pair_index]))
                hard_distances.append(float(pair_distances[pair_index]))
                hard_gaps.append(float(pair_gaps[pair_index]))
                hard_is_gt.append(False)
                if len(hard_indices) == cfg.num_local_hard:
                    break

        # If a scene has fewer than the configured eligible unique anchors, keep
        # the distinction honest and replace the missing hard quota with more
        # balanced supervision.  The returned tensor still has fixed length.
        hard_indices_np = np.asarray(hard_indices, dtype=np.int64)
        hard_proposals_np = np.asarray(hard_proposals, dtype=np.int64)
        hard_distances_np = np.asarray(hard_distances, dtype=np.float32)
        hard_gaps_np = np.asarray(hard_gaps, dtype=np.float32)
        hard_is_gt_np = np.asarray(hard_is_gt, dtype=bool)
        total_count = cfg.num_local_hard + cfg.num_balanced
        balanced_count = total_count - len(hard_indices_np)
        scene_seed = cfg.seed if seed is None else int(seed)
        rng = np.random.default_rng(scene_seed)
        balanced_indices, balanced_buckets = self._sample_balanced(
            anchor_scores, balanced_count, hard_indices_np, rng
        )

        selected = np.concatenate([hard_indices_np, balanced_indices])
        selected_distances = distances[:, selected]
        balanced_nearest = np.argmin(
            selected_distances[:, len(hard_indices_np) :], axis=0
        ).astype(np.int64)
        nearest_proposals = np.concatenate([hard_proposals_np, balanced_nearest])
        balanced_distances = distances[balanced_nearest, balanced_indices]
        balanced_gaps = np.abs(
            anchor_scores[balanced_indices, 5] - proposal_pdms[balanced_nearest]
        )
        final_distances = np.concatenate([hard_distances_np, balanced_distances])
        final_gaps = np.concatenate([hard_gaps_np, balanced_gaps])
        is_hard = np.zeros(total_count, dtype=bool)
        is_hard[: len(hard_indices_np)] = True
        is_gt_hard = np.zeros(total_count, dtype=bool)
        is_gt_hard[: len(hard_indices_np)] = hard_is_gt_np
        hard_buckets = np.where(
            hard_is_gt_np, "gt_local_hard", "proposal_local_hard"
        )
        buckets = np.concatenate([hard_buckets, balanced_buckets])
        if len(np.unique(selected)) != total_count:
            raise AssertionError("anchor sampler returned duplicate indices")
        return AnchorSample(
            indices=selected,
            nearest_proposal_indices=nearest_proposals,
            distances=final_distances.astype(np.float32),
            score_gaps=final_gaps.astype(np.float32),
            is_local_hard=is_hard,
            is_gt_local_hard=is_gt_hard,
            buckets=buckets,
        )


def anchor_score_cache_path(
    metric_cache_path: Path,
    metric_cache_root: Path,
    anchor_score_root: Path,
) -> Path:
    metric_cache_path = Path(metric_cache_path)
    metric_cache_root = Path(metric_cache_root)
    try:
        relative = metric_cache_path.relative_to(metric_cache_root)
    except ValueError:
        parts = metric_cache_path.parts
        marker = metric_cache_root.name
        if marker not in parts:
            raise ValueError(
                f"Cannot map {metric_cache_path} relative to {metric_cache_root}"
            )
        marker_index = len(parts) - 1 - list(reversed(parts)).index(marker)
        relative = Path(*parts[marker_index + 1 :])
    return Path(anchor_score_root) / relative.parent / f"{relative.name}.npy"


def load_anchor_scores(path: Path, mmap_mode: Optional[str] = "r") -> np.ndarray:
    scores = np.load(Path(path), mmap_mode=mmap_mode, allow_pickle=False)
    if scores.shape != EXPECTED_ANCHOR_SCORE_SHAPE:
        raise ValueError(
            f"Expected anchor scores {EXPECTED_ANCHOR_SCORE_SHAPE}, got {scores.shape} at {path}"
        )
    if scores.dtype != np.float16:
        raise ValueError(f"Expected fp16 anchor scores, got {scores.dtype} at {path}")
    if not np.isfinite(scores).all():
        raise ValueError(f"Anchor score cache contains non-finite values: {path}")
    if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
        raise ValueError(f"Anchor score cache is outside [0,1]: {path}")
    return scores
