"""Deterministic safety-focused counterfactual sampling for Formula scorers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .hard_anchor_sampler import trajectory_distance_matrix


@dataclass(frozen=True)
class FormulaCounterfactualSamplerConfig:
    num_ttc_only: int = 24
    num_joint_unsafe: int = 8
    num_safe_rescue: int = 16
    num_safe_progress: int = 16
    num_balanced: int = 64
    safe_metric_min: float = 0.95
    failure_metric_max: float = 0.05
    safe_final_min: float = 0.90
    progress_high_min: float = 0.80
    progress_low_max: float = 0.60
    yaw_weight: float = 0.5
    seed: int = 20260803

    @property
    def total_count(self) -> int:
        return (
            self.num_ttc_only
            + self.num_joint_unsafe
            + self.num_safe_rescue
            + self.num_safe_progress
            + self.num_balanced
        )


@dataclass(frozen=True)
class FormulaCounterfactualSample:
    indices: np.ndarray
    nearest_proposal_indices: np.ndarray
    distances: np.ndarray
    buckets: np.ndarray


class FormulaCounterfactualSampler:
    """Sample a fixed 128-anchor safety curriculum without model predictions.

    The full 8192-anchor bank is filtered using cached ground-truth PDM heads.
    Within a bucket, trajectories closest to the generated set are preferred.
    Missing bucket quota is deterministically backfilled from the remaining
    anchors, so every scene has a fixed tensor shape for DDP.
    """

    def __init__(
        self,
        anchors: np.ndarray,
        config: FormulaCounterfactualSamplerConfig,
    ) -> None:
        self.anchors = np.asarray(anchors, dtype=np.float32)
        self.config = config
        if self.anchors.ndim != 3 or self.anchors.shape[-1] != 3:
            raise ValueError(
                f"anchors must be [A,T,3], got {self.anchors.shape}"
            )
        counts = (
            config.num_ttc_only,
            config.num_joint_unsafe,
            config.num_safe_rescue,
            config.num_safe_progress,
            config.num_balanced,
        )
        if any(count < 0 for count in counts):
            raise ValueError("counterfactual bucket counts must be non-negative")
        if config.total_count <= 0:
            raise ValueError("counterfactual total count must be positive")

    @staticmethod
    def _ordered_candidates(
        mask: np.ndarray,
        min_distance: np.ndarray,
        excluded: Sequence[int],
    ) -> np.ndarray:
        available = np.asarray(mask, dtype=bool).copy()
        if excluded:
            available[np.asarray(excluded, dtype=np.int64)] = False
        candidates = np.flatnonzero(available)
        if not len(candidates):
            return candidates
        order = np.lexsort((candidates, min_distance[candidates]))
        return candidates[order]

    def _take(
        self,
        mask: np.ndarray,
        count: int,
        label: str,
        min_distance: np.ndarray,
        selected: list[int],
        labels: list[str],
    ) -> None:
        if count <= 0:
            return
        candidates = self._ordered_candidates(mask, min_distance, selected)
        chosen = candidates[:count]
        selected.extend(chosen.tolist())
        labels.extend([label] * len(chosen))

    def _sample_balanced(
        self,
        scores: np.ndarray,
        count: int,
        min_distance: np.ndarray,
        selected: list[int],
        labels: list[str],
        rng: np.random.Generator,
    ) -> None:
        if count <= 0:
            return
        safe = self.config.safe_metric_min
        failure = self.config.failure_metric_max
        final = scores[:, 5]
        masks = (
            (scores[:, 0] >= safe)
            & (scores[:, 1] >= safe)
            & (scores[:, 3] >= safe)
            & (final >= self.config.safe_final_min),
            (final >= 0.3) & (final < 0.8),
            scores[:, 0] <= failure,
            scores[:, 1] <= failure,
            scores[:, 3] <= failure,
            np.ones(len(scores), dtype=bool),
        )
        names = (
            "balanced_high_safe",
            "balanced_medium",
            "balanced_noc_failure",
            "balanced_dac_failure",
            "balanced_ttc_failure",
            "balanced_random",
        )
        weights = np.asarray((2, 2, 1, 1, 1, 1), dtype=np.float64)
        raw_quota = count * weights / weights.sum()
        quota = np.floor(raw_quota).astype(np.int64)
        remainder = count - int(quota.sum())
        if remainder:
            order = np.argsort(-(raw_quota - quota), kind="stable")
            quota[order[:remainder]] += 1

        balanced_start = len(selected)
        for mask, name, bucket_count in zip(masks, names, quota):
            available = self._ordered_candidates(mask, min_distance, selected)
            take = min(int(bucket_count), len(available))
            if take:
                # Retain locality while avoiding the exact same balanced bank
                # in every epoch/scene: sample from the nearest 4x pool.
                pool = available[: min(len(available), max(take * 4, take))]
                chosen = rng.choice(pool, size=take, replace=False)
                chosen = chosen[
                    np.lexsort((chosen, min_distance[chosen]))
                ]
                selected.extend(chosen.tolist())
                labels.extend([name] * take)

        missing = count - (len(selected) - balanced_start)
        if missing:
            available_mask = np.ones(len(scores), dtype=bool)
            if selected:
                available_mask[np.asarray(selected, dtype=np.int64)] = False
            available = np.flatnonzero(available_mask)
            if len(available) < missing:
                raise RuntimeError(
                    "not enough unique anchors to backfill counterfactual batch"
                )
            chosen = rng.choice(available, size=missing, replace=False)
            selected.extend(chosen.tolist())
            labels.extend(["balanced_fill"] * missing)

    def sample(
        self,
        proposals: np.ndarray,
        anchor_scores: np.ndarray,
        *,
        seed: Optional[int] = None,
    ) -> FormulaCounterfactualSample:
        proposals = np.asarray(proposals, dtype=np.float32)
        scores = np.asarray(anchor_scores, dtype=np.float32)
        if proposals.ndim != 3 or proposals.shape[-1] != 3:
            raise ValueError(
                f"proposals must be [P,T,3], got {proposals.shape}"
            )
        if scores.shape != (len(self.anchors), 6):
            raise ValueError(
                f"anchor_scores must be [{len(self.anchors)},6], got "
                f"{scores.shape}"
            )
        if not np.isfinite(proposals).all() or not np.isfinite(scores).all():
            raise ValueError("counterfactual sampler inputs must be finite")

        distances = trajectory_distance_matrix(
            proposals, self.anchors, yaw_weight=self.config.yaw_weight
        )
        nearest_proposal = np.argmin(distances, axis=0).astype(np.int64)
        min_distance = distances[nearest_proposal, np.arange(len(self.anchors))]
        cfg = self.config
        safe = cfg.safe_metric_min
        failure = cfg.failure_metric_max
        noc_safe = scores[:, 0] >= safe
        noc_fail = scores[:, 0] <= failure
        dac_safe = scores[:, 1] >= safe
        ttc_safe = scores[:, 3] >= safe
        ttc_fail = scores[:, 3] <= failure
        fully_safe = noc_safe & dac_safe & ttc_safe

        selected: list[int] = []
        labels: list[str] = []
        self._take(
            noc_safe & ttc_fail,
            cfg.num_ttc_only,
            "ttc_only",
            min_distance,
            selected,
            labels,
        )
        self._take(
            noc_fail & ttc_fail,
            cfg.num_joint_unsafe,
            "ttc_noc_joint",
            min_distance,
            selected,
            labels,
        )
        self._take(
            fully_safe & (scores[:, 5] >= cfg.safe_final_min),
            cfg.num_safe_rescue,
            "safe_rescue",
            min_distance,
            selected,
            labels,
        )

        progress_high_count = (cfg.num_safe_progress + 1) // 2
        progress_low_count = cfg.num_safe_progress - progress_high_count
        self._take(
            fully_safe & (scores[:, 2] >= cfg.progress_high_min),
            progress_high_count,
            "safe_progress_high",
            min_distance,
            selected,
            labels,
        )
        self._take(
            fully_safe & (scores[:, 2] <= cfg.progress_low_max),
            progress_low_count,
            "safe_progress_low",
            min_distance,
            selected,
            labels,
        )

        # Any rare-scene shortage in a reserved bucket becomes balanced data.
        reserved_target = (
            cfg.num_ttc_only
            + cfg.num_joint_unsafe
            + cfg.num_safe_rescue
            + cfg.num_safe_progress
        )
        balanced_count = cfg.num_balanced + reserved_target - len(selected)
        rng = np.random.default_rng(cfg.seed if seed is None else int(seed))
        self._sample_balanced(
            scores,
            balanced_count,
            min_distance,
            selected,
            labels,
            rng,
        )

        if len(selected) != cfg.total_count:
            raise AssertionError(
                f"expected {cfg.total_count} anchors, sampled {len(selected)}"
            )
        indices = np.asarray(selected, dtype=np.int64)
        if len(np.unique(indices)) != len(indices):
            raise AssertionError("counterfactual sampler returned duplicates")
        return FormulaCounterfactualSample(
            indices=indices,
            nearest_proposal_indices=nearest_proposal[indices],
            distances=min_distance[indices].astype(np.float32),
            buckets=np.asarray(labels),
        )
