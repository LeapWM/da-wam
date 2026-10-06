"""E2-style local-hard plus balanced sampling for PostTraj Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


EXPECTED_SCORE_SHAPE = (8192, 6)


@dataclass(frozen=True)
class Stage2SamplerConfig:
    num_local_hard: int = 64
    num_balanced: int = 64
    nearest_per_proposal: int = 64
    min_score_gap: float = 0.2
    yaw_weight: float = 0.5
    safe_metric_min: float = 0.95
    high_final_min: float = 0.95
    medium_final_min: float = 0.3
    medium_final_max: float = 0.8
    failure_metric_max: float = 0.5
    sampling_profile: str = "balanced"
    seed: int = 20260815


@dataclass(frozen=True)
class Stage2AnchorSample:
    indices: np.ndarray
    distances: np.ndarray
    score_gaps: np.ndarray
    is_local_hard: np.ndarray
    buckets: np.ndarray

    @property
    def local_hard_count(self) -> int:
        return int(self.is_local_hard.sum())


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def trajectory_distance_matrix(
    proposals: np.ndarray,
    anchors: np.ndarray,
    yaw_weight: float = 0.5,
) -> np.ndarray:
    """Mean XY L2 plus wrapped-yaw distance, shaped ``[P,A]``."""
    proposals = np.asarray(proposals, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)
    if proposals.ndim != 3 or proposals.shape[-1] != 3:
        raise ValueError(f"proposals must be [P,T,3], got {proposals.shape}")
    if anchors.ndim != 3 or anchors.shape[1:] != proposals.shape[1:]:
        raise ValueError(
            f"anchors must be [A,{proposals.shape[1]},{proposals.shape[2]}], "
            f"got {anchors.shape}"
        )
    xy = anchors[None, :, :, :2] - proposals[:, None, :, :2]
    xy_distance = np.linalg.norm(xy, axis=-1).mean(axis=-1)
    yaw = wrap_angle(
        anchors[None, :, :, 2] - proposals[:, None, :, 2]
    )
    return xy_distance + float(yaw_weight) * np.abs(yaw).mean(axis=-1)


def _balanced_quotas(
    count: int, sampling_profile: str = "balanced"
) -> np.ndarray:
    # high-safe / medium / NOC fail / DAC fail / strict TTC-only / random
    if sampling_profile == "balanced":
        weights = np.asarray([2, 2, 1, 1, 1, 1], dtype=np.float64)
    elif sampling_profile == "safety_focus":
        # At count=96: 32 safe-high, 16 medium, 16 NOC, 8 DAC,
        # 24 strict TTC-only, and no unconditional random quota.
        weights = np.asarray([32, 16, 16, 8, 24, 0], dtype=np.float64)
    else:
        raise ValueError(
            f"unknown Stage 2 sampling profile: {sampling_profile}"
        )
    raw = int(count) * weights / weights.sum()
    quotas = np.floor(raw).astype(np.int64)
    remainder = int(count) - int(quotas.sum())
    order = np.argsort(-(raw - quotas), kind="stable")
    quotas[order[:remainder]] += 1
    return quotas


class PostTrajStage2Sampler:
    """Deterministically sample a fixed-size E2 counterfactual set."""

    def __init__(
        self,
        anchors: np.ndarray,
        config: Optional[Stage2SamplerConfig] = None,
    ) -> None:
        self.anchors = np.asarray(anchors, dtype=np.float32)
        if self.anchors.ndim != 3 or self.anchors.shape[-1] != 3:
            raise ValueError(f"anchors must be [A,T,3], got {self.anchors.shape}")
        self.config = config or Stage2SamplerConfig()

    def _sample_balanced(
        self,
        scores: np.ndarray,
        count: int,
        excluded: Sequence[int],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        available = np.ones(len(scores), dtype=bool)
        if len(excluded):
            available[np.asarray(excluded, dtype=np.int64)] = False
        safe = scores[:, [0, 1, 3]] >= cfg.safe_metric_min
        masks = (
            safe.all(axis=1) & (scores[:, 5] >= cfg.high_final_min),
            (scores[:, 5] >= cfg.medium_final_min)
            & (scores[:, 5] < cfg.medium_final_max),
            scores[:, 0] <= cfg.failure_metric_max,
            scores[:, 1] <= cfg.failure_metric_max,
            # Strict TTC-only: collision and DAC safe, TTC failed.
            (scores[:, 0] >= cfg.safe_metric_min)
            & (scores[:, 1] >= cfg.safe_metric_min)
            & (scores[:, 3] <= cfg.failure_metric_max),
            np.ones(len(scores), dtype=bool),
        )
        names = (
            "high_safe",
            "medium",
            "noc_failure",
            "dac_failure",
            "ttc_only",
            "random",
        )
        selected = []
        buckets = []
        for mask, name, quota in zip(
            masks,
            names,
            _balanced_quotas(count, cfg.sampling_profile),
        ):
            candidates = np.flatnonzero(mask & available)
            take = min(int(quota), len(candidates))
            if take:
                chosen = rng.choice(candidates, size=take, replace=False)
                selected.extend(chosen.tolist())
                buckets.extend([name] * take)
                available[chosen] = False
        remaining = int(count) - len(selected)
        if remaining:
            candidates = np.flatnonzero(available)
            if len(candidates) < remaining:
                raise RuntimeError(
                    f"cannot sample {count} unique balanced anchors"
                )
            chosen = rng.choice(candidates, size=remaining, replace=False)
            selected.extend(chosen.tolist())
            buckets.extend(["random_fill"] * remaining)
        return np.asarray(selected, dtype=np.int64), np.asarray(buckets)

    def sample(
        self,
        proposals: np.ndarray,
        proposal_final: np.ndarray,
        anchor_scores: np.ndarray,
        seed: Optional[int] = None,
    ) -> Stage2AnchorSample:
        cfg = self.config
        proposals = np.asarray(proposals, dtype=np.float32)
        proposal_final = np.asarray(proposal_final, dtype=np.float32)
        anchor_scores = np.asarray(anchor_scores, dtype=np.float32)
        if proposal_final.shape != (len(proposals),):
            raise ValueError("proposal_final must have shape [P]")
        if anchor_scores.shape != (len(self.anchors), 6):
            raise ValueError(
                f"anchor_scores must be [{len(self.anchors)},6], "
                f"got {anchor_scores.shape}"
            )
        if not all(
            np.isfinite(value).all()
            for value in (proposals, proposal_final, anchor_scores)
        ):
            raise ValueError("sampler inputs must be finite")

        distances = trajectory_distance_matrix(
            proposals, self.anchors, cfg.yaw_weight
        )
        nearest_count = min(cfg.nearest_per_proposal, len(self.anchors))
        nearest = np.argpartition(
            distances, kth=nearest_count - 1, axis=1
        )[:, :nearest_count]
        proposal_indices = np.repeat(
            np.arange(len(proposals), dtype=np.int64), nearest_count
        )
        anchor_indices = nearest.reshape(-1)
        pair_distances = distances[proposal_indices, anchor_indices]
        pair_gaps = np.abs(
            anchor_scores[anchor_indices, 5]
            - proposal_final[proposal_indices]
        )
        eligible = np.flatnonzero(pair_gaps >= cfg.min_score_gap)
        hardness = pair_gaps / (pair_distances + 1e-3)
        order = eligible[
            np.lexsort(
                (
                    anchor_indices[eligible],
                    proposal_indices[eligible],
                    pair_distances[eligible],
                    -hardness[eligible],
                )
            )
        ]
        hard_indices = []
        hard_distances = []
        hard_gaps = []
        seen = set()
        for pair_index in order:
            anchor_index = int(anchor_indices[pair_index])
            if anchor_index in seen:
                continue
            seen.add(anchor_index)
            hard_indices.append(anchor_index)
            hard_distances.append(float(pair_distances[pair_index]))
            hard_gaps.append(float(pair_gaps[pair_index]))
            if len(hard_indices) == cfg.num_local_hard:
                break

        hard_indices = np.asarray(hard_indices, dtype=np.int64)
        total_count = cfg.num_local_hard + cfg.num_balanced
        # Honest shortage fallback: do not relabel low-gap examples as hard.
        balanced_count = total_count - len(hard_indices)
        rng = np.random.default_rng(cfg.seed if seed is None else int(seed))
        balanced_indices, balanced_buckets = self._sample_balanced(
            anchor_scores, balanced_count, hard_indices, rng
        )
        selected = np.concatenate([hard_indices, balanced_indices])
        selected_distances = distances[:, selected]
        nearest_proposals = np.argmin(selected_distances, axis=0)
        final_distances = selected_distances[
            nearest_proposals, np.arange(len(selected))
        ].astype(np.float32)
        final_gaps = np.abs(
            anchor_scores[selected, 5] - proposal_final[nearest_proposals]
        ).astype(np.float32)
        is_hard = np.zeros(total_count, dtype=bool)
        is_hard[: len(hard_indices)] = True
        buckets = np.concatenate(
            [
                np.full(len(hard_indices), "local_hard"),
                balanced_buckets,
            ]
        )
        if len(selected) != total_count or len(np.unique(selected)) != total_count:
            raise AssertionError("Stage 2 sampler must return fixed unique anchors")
        return Stage2AnchorSample(
            indices=selected,
            distances=final_distances,
            score_gaps=final_gaps,
            is_local_hard=is_hard,
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
                f"cannot map {metric_cache_path} relative to {metric_cache_root}"
            )
        marker_index = len(parts) - 1 - list(reversed(parts)).index(marker)
        relative = Path(*parts[marker_index + 1 :])
    return Path(anchor_score_root) / relative.parent / f"{relative.name}.npy"


def load_anchor_scores(path: Path, mmap_mode: Optional[str] = "r") -> np.ndarray:
    scores = np.load(Path(path), mmap_mode=mmap_mode, allow_pickle=False)
    if scores.shape != EXPECTED_SCORE_SHAPE:
        raise ValueError(
            f"expected anchor scores {EXPECTED_SCORE_SHAPE}, got "
            f"{scores.shape} at {path}"
        )
    if scores.dtype != np.float16:
        raise ValueError(f"expected fp16 anchor scores, got {scores.dtype}")
    if not np.isfinite(scores).all():
        raise ValueError(f"non-finite anchor scores at {path}")
    if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
        raise ValueError(f"anchor scores outside [0,1] at {path}")
    return scores
