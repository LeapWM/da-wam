"""Training-free selection using model scores and causal, ego-only EC.

No human trajectory, map, traffic future, or true safety score enters selection.
The learned v1 score is a ranking signal, NOT a calibrated v2 EPDMS.
"""
import json
from pathlib import Path

import numpy as np


def predecessor_map(metadata, mappings=None):
    """metadata: token -> (log, time). Synthetic pairs MUST use official mappings."""
    previous = {}
    if mappings is None:
        by_log = {}
        for token, (log, time) in metadata.items():
            by_log.setdefault(log, []).append((time, token))
        for items in by_log.values():
            items.sort()
            for (old_time, old), (now_time, now) in zip(items, items[1:]):
                if 0 < now_time - old_time < 0.55:
                    previous[now] = old
    else:
        for now, old, branches in mappings:
            for current, prior in [(now, old)] + [tuple(pair) for pair in branches]:
                if current not in metadata:
                    continue
                if prior not in metadata:
                    raise ValueError(f"Missing predecessor {prior} for {current}")
                if current in previous and previous[current] != prior:
                    raise ValueError(f"Ambiguous predecessor for {current}")
                if metadata[current][0] != metadata[prior][0]:
                    raise ValueError("Official pair crosses logs")
                gap = metadata[current][1] - metadata[prior][1]
                if not 0 < gap < 0.55:
                    raise ValueError(f"Invalid official pair gap: {current}, {gap}")
                previous[current] = prior
    return previous


def choose(scores, ec=None, mode="bounded", weight=0.125, max_score_drop=0.02):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or len(scores) == 0:
        raise ValueError("Expected finite candidate scores")
    if not 0 <= weight <= 1 or max_score_drop < 0 or mode not in ("bounded", "momentum"):
        raise ValueError("Invalid temporal selection settings")
    base = int(scores.argmax())
    if ec is None or weight == 0:
        return base, scores.copy()
    ec = np.asarray(ec, dtype=np.float64)
    if ec.shape != scores.shape or not np.isin(ec, [0, 1]).all():
        raise ValueError("EC must be a binary value per candidate")
    adjusted = (1 - weight) * scores + weight * ec
    if mode == "bounded":
        adjusted[scores < scores[base] - max_score_drop] = -np.inf
    best = np.flatnonzero(adjusted == adjusted.max())
    # Preserve the base choice when its adjusted score ties the best.
    return (base if base in best else int(best[0])), adjusted


def pair_ec(previous_states, current_states, gap, interval=0.1):
    """Official EC, current candidates against ONE specified prior prediction."""
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        ego_is_two_frame_extended_comfort,
    )
    if not 0 < gap < 0.55:
        raise ValueError("EC requires an adjacent positive time gap")
    offset = round(gap / interval)  # same rounding as SceneAggregator
    if offset < 1 or offset >= current_states.shape[1]:
        raise ValueError("Invalid overlap")
    current = current_states[:, :-offset]
    previous = np.broadcast_to(previous_states[offset:], current.shape)
    times = np.arange(current.shape[1]) * interval
    return ego_is_two_frame_extended_comfort(current, previous, times).astype(np.float64)


class TemporalReranker:
    def __init__(self, metadata, previous, sampling, settings, audit_path):
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
        self.metadata = metadata
        self.previous = previous
        self.simulator = PDMSimulator(sampling)
        self.settings = settings
        self.states = {}
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.needed = set(previous.values())

    @classmethod
    def prepare(cls, tokens, cache_loader, cfg, worker_id, two_stage=False):
        settings = cfg.get("temporal_rerank")
        if not settings or not settings.get("enabled", False):
            return None, tokens
        # Read only metadata here, release large scene caches immediately.
        metadata = {}
        for token in sorted(tokens):
            cache = cache_loader.get_from_token(token)
            metadata[token] = (cache.log_name, cache.timepoint.time_s)
        mappings = cfg.train_test_split.reactive_all_mapping if two_stage else None
        previous = predecessor_map(metadata, mappings)
        from hydra.utils import instantiate
        obj = cls(metadata, previous, instantiate(cfg.simulator.proposal_sampling), settings,
                  Path(cfg.output_dir) / "temporal_audit" / f"{worker_id}.jsonl")
        return obj, sorted(tokens, key=lambda t: (*metadata[t], t))

    def compute(self, agent, agent_input, token, initial_ego_state, output=None):
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
        if output is None:
            output = agent.compute_candidates(agent_input)
        proposals, scores = output["proposals"][0], output["pdm_score"][0]
        if not np.isfinite(proposals).all():
            raise ValueError("Non-finite proposals")
        base = int(scores.argmax())
        np.testing.assert_array_equal(output["trajectory"][0], proposals[base])
        sampling = self.simulator.proposal_sampling
        arrays = [get_trajectory_as_array(
            transform_trajectory(Trajectory(p, agent._trajectory_sampling), initial_ego_state),
            sampling, initial_ego_state.time_point) for p in proposals]
        simulated = self.simulator.simulate_proposals(np.stack(arrays), initial_ego_state)
        prior = self.previous.get(token)
        old_states = self.states.get(prior)
        ec = None
        reason = "no_predecessor" if prior is None else "predecessor_failed"
        if old_states is not None:
            gap = self.metadata[token][1] - self.metadata[prior][1]
            ec = pair_ec(old_states, simulated, gap, sampling.interval_length)
            reason = "paired"
        selected, _ = choose(scores, ec, self.settings.mode,
                             float(self.settings.weight), float(self.settings.max_score_drop))
        if token in self.needed:
            self.states[token] = simulated[selected].copy()
        audit = dict(token=token, previous_token=prior, reason=reason,
                     base_index=base, selected_index=selected, changed=selected != base,
                     base_score=float(scores[base]), selected_score=float(scores[selected]),
                     base_ec=None if ec is None else float(ec[base]),
                     selected_ec=None if ec is None else float(ec[selected]),
                     candidate_scores=scores.tolist(), candidate_ec=None if ec is None else ec.tolist())
        with self.audit_path.open("a") as f:
            f.write(json.dumps(audit) + "\n")
        return Trajectory(proposals[selected], agent._trajectory_sampling)

    def invalidate(self, token):
        # A later frame may only use a prediction whose evaluation succeeded.
        self.states.pop(token, None)
