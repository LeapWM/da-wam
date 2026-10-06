"""Inference-only NAVSIM v2 adapter for a NAVSIM v1 PostTraj checkpoint.

The model and scorer remain checkpoint-compatible with the v1 training code.
Only the AbstractAgent/data boundary is adapted to NAVSIM v2.
"""

from typing import Any, Dict

import torch

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.drive_jepa_perception_based.drive_jepa_features import (
    DriveJEPAFeatureBuilder,
)
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory

from .drive_jepa_config import DriveJEPAConfig
from .drive_jepa_model import DriveJEPAModel


class DriveJEPAPostTrajV2Agent(AbstractAgent):
    """Run the v1 PostTraj network under NAVSIM v2's evaluator contract."""

    def __init__(
        self,
        config: DriveJEPAConfig,
        lr: float = 1e-4,
        checkpoint_path: str = "",
        cache_data: bool = False,
    ) -> None:
        super().__init__(config.trajectory_sampling)
        if cache_data:
            raise ValueError("PostTraj v2 is inference-only and cannot cache data")
        if not config.posttraj_future_enabled:
            raise ValueError("PostTraj v2 requires posttraj_future_enabled=True")
        self._config = config
        self._lr = float(lr)
        self._checkpoint_path = checkpoint_path
        self._pad_model = DriveJEPAModel(config)

    def name(self) -> str:
        return "drive_jepa_posttraj_v1_scorer_navsim_v2"

    def initialize(self) -> None:
        if not self._checkpoint_path:
            raise ValueError("A trained PostTraj checkpoint is required")
        checkpoint: Dict[str, Any] = torch.load(
            self._checkpoint_path,
            map_location="cpu",
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        remapped = {
            key.replace("agent._pad_model", "_pad_model", 1): value
            for key, value in state_dict.items()
            if key.startswith("agent._pad_model")
        }
        if not remapped:
            raise KeyError(
                "checkpoint has no agent._pad_model parameters: "
                f"{self._checkpoint_path}"
            )
        # Strict loading is intentional: the v2 result must use the exact v1
        # PostTraj scorer rather than silently initializing new parameters.
        self.load_state_dict(remapped, strict=True)

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig(
            cam_f0=[2, 3],
            cam_l0=[3],
            cam_l1=[],
            cam_l2=[],
            cam_r0=[3],
            cam_r1=[],
            cam_r2=[],
            cam_b0=[3],
            lidar_pc=[],
        )

    def get_feature_builders(self):
        return [DriveJEPAFeatureBuilder(config=self._config)]

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """Run inference on the module device under both v2.0 and v2.2 APIs."""
        output = self.compute_candidates(agent_input)
        return Trajectory(output["trajectory"][0], self._trajectory_sampling)

    def compute_candidates(self, agent_input: AgentInput):
        """Expose the unchanged proposal pool and learned scores in one forward."""
        self.eval()
        device = next(self.parameters()).device
        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        features = {
            key: value.unsqueeze(0).to(device)
            for key, value in features.items()
        }
        with torch.no_grad():
            output = self.forward(features)
        return {
            key: output[key].detach().cpu().numpy()
            for key in ("trajectory", "proposals", "pdm_score")
        }

    def forward(self, features: Dict[str, torch.Tensor]):
        return self._pad_model(features)
