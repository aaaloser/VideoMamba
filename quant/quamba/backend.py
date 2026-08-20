from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[2]

if str(_REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_REPO_ROOT))

from quamba import (  # noqa: E402
    QuambaCalibrationResult,
    QuambaPTQBridgeSession as QuambaPTQSession,
    apply_quamba_bridge_ptq,
    build_uniform_block_bits,
    calibrate_quamba_bridge,
)


def run_quamba_calibration(
    model: nn.Module,
    calib_loader: Iterable[Any],
    device: torch.device,
    max_calib_batches: int = 16,
    a_bits: int = 8,
    percentile_alpha: float = 0.9995,
    calib_size: Optional[int] = None,
    calib_batch_size: Optional[int] = None,
) -> QuambaCalibrationResult:
    return calibrate_quamba_bridge(
        model=model,
        calib_loader=calib_loader,
        device=device,
        max_calib_batches=max_calib_batches,
        a_bits=a_bits,
        percentile_alpha=percentile_alpha,
        calib_size=calib_size,
        calib_batch_size=calib_batch_size,
    )


def apply_quamba_ptq(
    model: nn.Module,
    block_bits: Dict[Any, int],
    default_bit: int = 4,
    calibration: Optional[QuambaCalibrationResult] = None,
) -> QuambaPTQSession:
    return apply_quamba_bridge_ptq(
        model=model,
        block_bits=block_bits,
        calibration=calibration,
        default_bit=default_bit,
    )
