"""Isolated EMA-JQTF building blocks.

The legacy Drive-JEPA future/scorer paths remain available.  These modules are
selected only when ``future_integration == "ema_jqtf"``.
"""

from .candidate_scorer import EMAJQTFCandidateScorer
from .joint_future import JointQueryFutureHead
from .target_pair import build_future_target_camera_pair

__all__ = [
    "EMAJQTFCandidateScorer",
    "JointQueryFutureHead",
    "build_future_target_camera_pair",
]
