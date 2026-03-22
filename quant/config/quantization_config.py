from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PTQConfig:
    """Hyper-parameters for VideoMamba PTQ."""

    num_groups: int = 4
    cls_token_position: str = "auto"  # auto | first | center | none
    temporal_steps: Optional[int] = None
    alpha: float = 0.5
    beta: float = 0.25
    gamma: float = 0.25
    tau_percentile: float = 90.0
    quick_high_ratio_threshold: float = 0.30
    high_bit: int = 8
    low_bit: int = 4
    eps: float = 1e-6
    lambda_min: float = 0.10
    lambda_max: float = 0.95
    default_lambda: float = 0.50
    default_bit: int = 8


@dataclass
class CalibrationResult:
    """Per-block calibration stats for sensitivity scoring."""

    block_stats: Dict[int, Dict[str, Any]]
    config: PTQConfig


@dataclass
class QuickEvalResult:
    """Static block bit allocation from quick-eval stage."""

    block_bits: Dict[int, int]
    block_high_ratio: Dict[int, float]
