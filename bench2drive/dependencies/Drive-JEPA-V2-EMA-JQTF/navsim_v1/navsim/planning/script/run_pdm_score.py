from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid

import numpy as np
import torch
import torch.nn as nn
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.evaluate.pdm_score import pdm_score, get_trajectory_as_array, transform_trajectory
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def move_all_modules_to_gpu(instance):
    """
    Move all nn.Module attributes of a class instance to GPU.
    """
    for attr_name in dir(instance):
        attr = getattr(instance, attr_name)
        if isinstance(attr, nn.Module):
            setattr(instance, attr_name, attr.cuda())


def compute_trajectory_with_proposals(agent: AbstractAgent, agent_input):
    """Run one model forward and expose the selected trajectory plus all proposals."""
    if agent.requires_scene:
        raise RuntimeError("oracle diagnostic currently supports agents without requires_scene")

    features: Dict[str, torch.Tensor] = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(agent_input))
    features = {key: value.unsqueeze(0).cuda() for key, value in features.items()}

    with torch.no_grad():
        predictions = agent.forward(features)
    required = ("trajectory", "proposals", "pdm_score")
    missing = [key for key in required if key not in predictions]
    if missing:
        raise RuntimeError(f"oracle diagnostic requires predictions {required}, missing {missing}")

    trajectory = Trajectory(predictions["trajectory"].squeeze(0).cpu().numpy())
    proposals = predictions["proposals"].squeeze(0).float().cpu().numpy()
    predicted_scores = predictions["pdm_score"].squeeze(0).float().cpu().numpy()
    selection_scores = (
        predictions.get("selection_score", predictions["pdm_score"])
        .squeeze(0)
        .float()
        .cpu()
        .numpy()
    )
    if "selected_index" in predictions:
        selected_index = int(predictions["selected_index"].reshape(-1)[0].item())
    else:
        selected_index = int(np.argmax(selection_scores))
    predicted_metrics = None
    if "pred_metric_scores" in predictions:
        predicted_metrics = (
            predictions["pred_metric_scores"]
            .squeeze(0)
            .float()
            .cpu()
            .numpy()
        )
    predicted_ep = None
    if "pred_ep" in predictions:
        predicted_ep = (
            predictions["pred_ep"].squeeze(0).float().cpu().numpy()
        )
    return (
        trajectory,
        proposals,
        predicted_scores,
        selection_scores,
        selected_index,
        predicted_metrics,
        predicted_ep,
    )


def formula_noc_ttc_diagnostics(
    predicted_metrics,
    proposal_results,
    selected_index,
    predicted_ep=None,
    predicted_formula_score=None,
    *,
    prediction_threshold=0.5,
    target_threshold=0.95,
    hierarchy_margin=0.05,
    high_progress_threshold=0.9,
    confident_safety_threshold=0.95,
    final_score_threshold=0.90,
):
    """Summarize NOC/TTC combinations for one Formula candidate set."""
    if predicted_metrics is None:
        return {}
    predicted_metrics = np.asarray(predicted_metrics)
    if predicted_metrics.ndim != 2 or predicted_metrics.shape[1] != 4:
        raise ValueError(
            "Formula predicted metrics must be [P,4] in "
            "[NOC,DAC,TTC,Comfort] order, got "
            f"{predicted_metrics.shape}"
        )
    pred_noc = predicted_metrics[:, 0]
    pred_dac = predicted_metrics[:, 1]
    pred_ttc = predicted_metrics[:, 2]
    true_noc = np.asarray(proposal_results["no_at_fault_collisions"])
    true_dac = np.asarray(proposal_results["drivable_area_compliance"])
    true_ttc = np.asarray(
        proposal_results["time_to_collision_within_bound"]
    )
    true_ep = np.asarray(proposal_results["ego_progress"])
    true_score = np.asarray(proposal_results["score"])
    if any(
        values.shape != pred_noc.shape
        for values in (pred_dac, pred_ttc, true_noc, true_dac, true_ttc, true_ep)
    ):
        raise ValueError("predicted and true Formula candidate counts differ")

    if predicted_ep is not None:
        predicted_ep = np.asarray(predicted_ep)
        if predicted_ep.shape != pred_noc.shape:
            raise ValueError("predicted EP and Formula candidate counts differ")

    pred_noc_pass = pred_noc >= prediction_threshold
    pred_dac_pass = pred_dac >= prediction_threshold
    pred_ttc_pass = pred_ttc >= prediction_threshold
    true_noc_pass = true_noc >= target_threshold
    true_dac_pass = true_dac >= target_threshold
    true_ttc_pass = true_ttc >= target_threshold
    hierarchy_violation = pred_ttc > pred_noc + hierarchy_margin
    true_joint_failure = ~true_noc_pass & ~true_ttc_pass
    predicted_double_safe = pred_noc_pass & pred_ttc_pass

    predicted_fully_safe = pred_noc_pass & pred_dac_pass & pred_ttc_pass
    true_fully_safe = true_noc_pass & true_dac_pass & true_ttc_pass
    selected_true_unsafe = not bool(true_fully_safe[selected_index])
    true_safe_count = int(true_fully_safe.sum())
    selected_true_score = float(true_score[selected_index])
    best_true_safe_score = (
        float(np.max(true_score[true_fully_safe]))
        if true_safe_count
        else float("nan")
    )
    true_safe_high_progress = (
        true_fully_safe & (true_ep >= high_progress_threshold)
    )
    if true_safe_count:
        safe_indices = np.flatnonzero(true_fully_safe)
        best_true_safe_ep_index = int(
            safe_indices[np.argmax(true_ep[safe_indices])]
        )
        best_true_safe_ep = float(true_ep[best_true_safe_ep_index])
        selected_safe_progress_regret = (
            max(best_true_safe_ep - float(true_ep[selected_index]), 0.0)
            if true_fully_safe[selected_index]
            else float("nan")
        )
    else:
        best_true_safe_ep_index = -1
        best_true_safe_ep = float("nan")
        selected_safe_progress_regret = float("nan")

    result = {
        "formula_candidate_count": int(len(pred_noc)),
        "pred_both_safe_count": int((pred_noc_pass & pred_ttc_pass).sum()),
        "pred_ttc_only_count": int((pred_noc_pass & ~pred_ttc_pass).sum()),
        "pred_joint_fail_count": int((~pred_noc_pass & ~pred_ttc_pass).sum()),
        "pred_noc_only_count": int((~pred_noc_pass & pred_ttc_pass).sum()),
        "pred_hierarchy_violation_count": int(hierarchy_violation.sum()),
        "pred_noc_mean": float(pred_noc.mean()),
        "pred_ttc_mean": float(pred_ttc.mean()),
        "selected_pred_noc": float(pred_noc[selected_index]),
        "selected_pred_dac": float(pred_dac[selected_index]),
        "selected_pred_ttc": float(pred_ttc[selected_index]),
        "selected_pred_hierarchy_violation": float(
            hierarchy_violation[selected_index]
        ),
        "true_both_safe_count": int((true_noc_pass & true_ttc_pass).sum()),
        "true_ttc_only_count": int((true_noc_pass & ~true_ttc_pass).sum()),
        "true_joint_fail_count": int(true_joint_failure.sum()),
        "true_noc_only_count": int((~true_noc_pass & true_ttc_pass).sum()),
        "joint_false_safe_count": int(
            (true_joint_failure & predicted_double_safe).sum()
        ),
        "true_fully_safe_count": true_safe_count,
        "selected_true_noc_failure": float(not true_noc_pass[selected_index]),
        "selected_true_dac_failure": float(not true_dac_pass[selected_index]),
        "selected_true_ttc_failure": float(not true_ttc_pass[selected_index]),
        "selected_true_any_safety_failure": float(selected_true_unsafe),
        "selected_predicted_fully_safe": float(
            predicted_fully_safe[selected_index]
        ),
        "selected_true_ep": float(true_ep[selected_index]),
        "best_true_safe_score": best_true_safe_score,
        # Progress diagnosis.  These separate (a) a high-progress safe
        # candidate being rejected by the safety heads from (b) the scorer
        # failing to rank an already-predicted-safe candidate by progress.
        "true_safe_high_progress_count": int(
            true_safe_high_progress.sum()
        ),
        "true_safe_high_progress_pred_unsafe_count": int(
            (true_safe_high_progress & ~predicted_fully_safe).sum()
        ),
        "true_safe_high_progress_pred_safe_count": int(
            (true_safe_high_progress & predicted_fully_safe).sum()
        ),
        "scene_has_true_safe_high_progress": float(
            true_safe_high_progress.any()
        ),
        "best_true_safe_ep": best_true_safe_ep,
        "best_true_safe_ep_predicted_safe": float(
            best_true_safe_ep_index >= 0
            and predicted_fully_safe[best_true_safe_ep_index]
        ),
        "selected_true_safe_progress_regret": selected_safe_progress_regret,
        "selected_safe_progress_regret_gt_0p05": float(
            np.isfinite(selected_safe_progress_regret)
            and selected_safe_progress_regret > 0.05
        ),
        "selected_unsafe_safe_candidate_available": float(
            selected_true_unsafe and true_safe_count > 0
        ),
        "selected_unsafe_safer_higher_score_available": float(
            selected_true_unsafe
            and true_safe_count > 0
            and best_true_safe_score > selected_true_score + 1e-12
        ),
    }
    if predicted_ep is not None:
        predicted_high_progress = predicted_ep >= high_progress_threshold
        predicted_confident_safe = (
            (pred_noc >= confident_safety_threshold)
            & (pred_dac >= confident_safety_threshold)
            & (pred_ttc >= confident_safety_threshold)
        )
        false_safe_high_progress = (
            predicted_fully_safe & predicted_high_progress & ~true_fully_safe
        )
        confident_false_safe_high_progress = (
            predicted_confident_safe
            & predicted_high_progress
            & ~true_fully_safe
        )
        result.update(
            {
                "predicted_safe_high_progress_count": int(
                    (predicted_fully_safe & predicted_high_progress).sum()
                ),
                "false_safe_high_progress_count": int(
                    false_safe_high_progress.sum()
                ),
                "confident_predicted_safe_high_progress_count": int(
                    (predicted_confident_safe & predicted_high_progress).sum()
                ),
                "confident_false_safe_high_progress_count": int(
                    confident_false_safe_high_progress.sum()
                ),
                "selected_pred_ep": float(predicted_ep[selected_index]),
                "selected_predicted_high_progress": float(
                    predicted_high_progress[selected_index]
                ),
                "selected_false_safe_high_progress": float(
                    false_safe_high_progress[selected_index]
                ),
                "selected_confident_false_safe_high_progress": float(
                    confident_false_safe_high_progress[selected_index]
                ),
                "scene_has_false_safe_high_progress_candidate": float(
                    false_safe_high_progress.any()
                ),
            }
        )
    if predicted_formula_score is not None:
        predicted_formula_score = np.asarray(predicted_formula_score)
        if predicted_formula_score.shape != true_score.shape:
            raise ValueError(
                "predicted Formula score and true candidate score counts differ"
            )
        pred_final90 = predicted_formula_score >= final_score_threshold
        true_final90 = true_score >= final_score_threshold
        true_positive = pred_final90 & true_final90
        result.update(
            {
                # Candidate-level confusion-matrix counts for identifying an
                # official PDM score >= 0.90 from the predicted Formula score.
                "final90_threshold": float(final_score_threshold),
                "pred_final90_count": int(pred_final90.sum()),
                "true_final90_count": int(true_final90.sum()),
                "final90_true_positive_count": int(true_positive.sum()),
                "final90_false_positive_count": int(
                    (pred_final90 & ~true_final90).sum()
                ),
                "final90_false_negative_count": int(
                    (~pred_final90 & true_final90).sum()
                ),
                "scene_has_true_final90": float(true_final90.any()),
                "scene_has_pred_final90": float(pred_final90.any()),
                "scene_final90_retrieved": float(
                    true_positive.any()
                ),
                "selected_true_final90": float(true_final90[selected_index]),
                "selected_pred_final90": float(pred_final90[selected_index]),
            }
        )
    return result


def score_all_proposals(metric_cache, proposals, future_sampling, simulator, scorer):
    """Score all model proposals together with the same PDM machinery as NavTest."""
    initial_ego_state = metric_cache.ego_state
    pdm_states = get_trajectory_as_array(
        metric_cache.trajectory, future_sampling, initial_ego_state.time_point
    )
    proposal_states = []
    for poses in proposals:
        candidate = transform_trajectory(Trajectory(poses), initial_ego_state)
        proposal_states.append(
            get_trajectory_as_array(candidate, future_sampling, initial_ego_state.time_point)
        )

    trajectory_states = np.concatenate(
        [pdm_states[None], np.stack(proposal_states, axis=0)], axis=0
    )
    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)
    scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )

    # PDMScorer normalizes EP by the maximum valid progress among all proposals
    # passed to one call. Official NavTest passes exactly [PDM baseline, selected
    # candidate]. Reconstruct that pairwise normalization for every candidate;
    # all other metrics are proposal-local and remain safely vectorized.
    candidate_slice = slice(1, None)  # index 0 is the cached PDM trajectory
    multiplicative = scorer._multi_metrics.prod(axis=0)
    gated_raw_progress = scorer._progress_raw * multiplicative
    candidate_max_progress = np.maximum(
        gated_raw_progress[0], gated_raw_progress[candidate_slice]
    )
    candidate_progress = np.ones_like(candidate_max_progress)
    normalize = candidate_max_progress > scorer._config.progress_distance_threshold
    candidate_progress[normalize] = (
        gated_raw_progress[candidate_slice][normalize] / candidate_max_progress[normalize]
    )
    candidate_progress[
        (~normalize) & (multiplicative[candidate_slice] == 0.0)
    ] = 0.0

    candidate_weighted = scorer._weighted_metrics[:, candidate_slice].copy()
    candidate_weighted[WeightedMetricIndex.PROGRESS] = candidate_progress
    weights = scorer._config.weighted_metrics_array
    weighted_score = (candidate_weighted * weights[:, None]).sum(axis=0) / weights.sum()
    pairwise_official_scores = multiplicative[candidate_slice] * weighted_score

    return {
        "score": np.asarray(pairwise_official_scores),
        "no_at_fault_collisions": np.asarray(
            scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, candidate_slice]
        ),
        "drivable_area_compliance": np.asarray(
            scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, candidate_slice]
        ),
        "ego_progress": np.asarray(candidate_progress),
        "time_to_collision_within_bound": np.asarray(
            scorer._weighted_metrics[WeightedMetricIndex.TTC, candidate_slice]
        ),
        "comfort": np.asarray(
            scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, candidate_slice]
        ),
        "driving_direction_compliance": np.asarray(
            scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, candidate_slice]
        ),
    }


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    oracle_diagnostic = bool(cfg.get("oracle_diagnostic", False))

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()

    move_all_modules_to_gpu(agent)


    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            agent_input = scene_loader.get_agent_input_from_token(token)
            if oracle_diagnostic:
                (
                    trajectory,
                    proposals,
                    predicted_scores,
                    selection_scores,
                    selected_index,
                    predicted_metrics,
                    predicted_ep,
                ) = compute_trajectory_with_proposals(agent, agent_input)
            elif agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
            if oracle_diagnostic:
                proposal_results = score_all_proposals(
                    metric_cache,
                    proposals,
                    simulator.proposal_sampling,
                    simulator,
                    scorer,
                )
                true_scores = proposal_results["score"]
                oracle_true_score = float(np.max(true_scores))
                oracle_indices = np.flatnonzero(
                    np.isclose(true_scores, oracle_true_score, rtol=1e-9, atol=1e-12)
                )
                predicted_order = np.argsort(-selection_scores)
                predicted_ranks = np.empty_like(predicted_order)
                predicted_ranks[predicted_order] = np.arange(1, len(predicted_order) + 1)
                # Pick the highest-ranked member of the true-score tie set so
                # top-k means that any equally optimal proposal was retrieved.
                oracle_index = int(oracle_indices[np.argmin(predicted_ranks[oracle_indices])])
                oracle_predicted_rank = int(predicted_ranks[oracle_index])
                selected_true_score = float(true_scores[selected_index])
                score_row.update(
                    {
                        "selected_index": selected_index,
                        "oracle_index": oracle_index,
                        "oracle_tie_count": int(len(oracle_indices)),
                        "selected_predicted_score": float(predicted_scores[selected_index]),
                        "oracle_predicted_score": float(predicted_scores[oracle_index]),
                        "oracle_predicted_rank": oracle_predicted_rank,
                        "oracle_top1": float(oracle_predicted_rank <= 1),
                        "oracle_top3": float(oracle_predicted_rank <= 3),
                        "oracle_top5": float(oracle_predicted_rank <= 5),
                        "oracle_selected_score": selected_true_score,
                        "oracle_score": oracle_true_score,
                        "oracle_regret": oracle_true_score - selected_true_score,
                        "oracle_selected_official_delta": selected_true_score - float(pdm_result.score),
                    }
                )
                score_row.update(
                    formula_noc_ttc_diagnostics(
                        predicted_metrics,
                        proposal_results,
                        selected_index,
                        predicted_ep,
                        predicted_scores,
                    )
                )
                for metric_name, values in proposal_results.items():
                    if metric_name == "score":
                        continue
                    score_row[f"oracle_{metric_name}"] = float(values[oracle_index])
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    worker = build_worker(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    oracle_max_scenarios = int(cfg.get("oracle_max_scenarios", 0))
    if oracle_max_scenarios > 0:
        tokens_to_evaluate = tokens_to_evaluate[:oracle_max_scenarios]
        logger.info(
            "Oracle diagnostic deterministically limited to %d sorted tokens",
            len(tokens_to_evaluate),
        )
    selected_tokens = set(tokens_to_evaluate)
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": [token for token in tokens_list if token in selected_tokens],
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
        if any(token in selected_tokens for token in tokens_list)
    ]
    score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score, data_points)

    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    oracle_summary = ""
    if bool(cfg.get("oracle_diagnostic", False)):
        oracle_summary = (
            f"\n            Oracle average score: {average_row['oracle_score']}."
            f"\n            Oracle selection regret: {average_row['oracle_regret']}."
            f"\n            Oracle top-1/top-3/top-5: "
            f"{average_row['oracle_top1']}/{average_row['oracle_top3']}/{average_row['oracle_top5']}."
            f"\n            Batched-vs-official selected delta: "
            f"{average_row['oracle_selected_official_delta']}."
        )

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            {oracle_summary}
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )


if __name__ == "__main__":
    main()
