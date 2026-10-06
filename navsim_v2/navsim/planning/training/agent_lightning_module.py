from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.drive_jepa_perception_based.score_module.epdms import (
    compose_epdms,
)


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        prediction = self.agent.forward(features)
        loss = self.agent.compute_loss(features, targets, prediction)
        if isinstance(loss, dict):
            for key, value in loss.items():
                self.log(
                    f"{logging_prefix}/{key}",
                    value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=key in {"loss", "score", "best_score"},
                    sync_dist=True,
                )
            return loss["loss"]
        self.log(
            f"{logging_prefix}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        if "perception_based" not in self.agent.name().lower():
            return self._step(batch, "val")

        features, targets = batch
        predictions = self.agent.forward(features)
        selected = predictions["trajectory"][:, None]
        (
            final_score,
            best_score,
            proposal_scores,
            l2,
            trajectory_scores,
        ) = self.agent.compute_score(targets, selected)

        selected_indices = torch.argmax(predictions["pdm_score"], dim=1)
        selected_metrics = predictions["pred_metric_values"][
            torch.arange(len(selected_indices), device=selected_indices.device),
            selected_indices,
        ]
        predicted_stage1 = compose_epdms(
            selected_metrics, include_extended_comfort=False
        )
        score_error = torch.abs(
            predicted_stage1 - proposal_scores[:, 0]
        ).mean()
        mean_score = proposal_scores.mean()

        scalar_metrics = {
            "score": final_score,
            "best_score": best_score,
            "mean_score": mean_score,
            "score_error": score_error,
            "l2": l2,
            "collision": trajectory_scores[:, 0].mean(),
            "dac": trajectory_scores[:, 1].mean(),
            "tlc": trajectory_scores[:, 2].mean(),
            "ddc": trajectory_scores[:, 3].mean(),
            "progress": trajectory_scores[:, 4].mean(),
            "ttc": trajectory_scores[:, 5].mean(),
            "lk": trajectory_scores[:, 6].mean(),
            "comfort": trajectory_scores[:, 7].mean(),
        }
        for name, value in scalar_metrics.items():
            self.log(
                f"val/{name}",
                value,
                on_step=name == "score",
                on_epoch=True,
                prog_bar=name in {"score", "best_score", "score_error"},
                sync_dist=True,
            )
        return final_score

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
