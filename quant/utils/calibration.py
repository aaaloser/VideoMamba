from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn

from quant.config import CalibrationResult, PTQConfig, QuickEvalResult
from quant.utils.helpers import (
    collect_layer_group_metrics,
    safe_float_cpu,
    score_metrics,
)


def calibrate_videomamba_ptq(
    model: nn.Module,
    calib_loader: Iterable[Any],
    device: torch.device,
    max_calib_batches: int = 16,
    cfg: Optional[PTQConfig] = None,
) -> CalibrationResult:
    """Phase-2 calibration: collect ranges and tau/lambda for each Mamba block."""

    cfg = cfg or PTQConfig()
    layer_metrics = collect_layer_group_metrics(
        model=model,
        data_loader=calib_loader,
        device=device,
        max_batches=max_calib_batches,
        cfg=cfg,
    )

    block_stats: Dict[int, Dict[str, Any]] = {}
    for layer_idx, metrics in layer_metrics.items():
        r_vals = safe_float_cpu([m["R"] for m in metrics])
        spa_vals = safe_float_cpu([m["E_spa"] for m in metrics])
        temp_vals = safe_float_cpu([m["E_temp"] for m in metrics])
        if len(r_vals) == 0:
            continue

        ranges = {
            "R": [float(r_vals.min()), float(r_vals.max())],
            "E_spa": [float(spa_vals.min()), float(spa_vals.max())],
            "E_temp": [float(temp_vals.min()), float(temp_vals.max())],
        }
        stat = {"ranges": ranges}

        scores = [score_metrics(m, stat, cfg) for m in metrics]
        tau = float(np.percentile(np.asarray(scores, dtype=np.float32), cfg.tau_percentile))

        high_scores = [s for s in scores if s > tau]
        lambda_est = float(np.mean(high_scores)) if high_scores else tau
        lambda_est = float(np.clip(lambda_est, cfg.lambda_min, cfg.lambda_max))

        block_stats[layer_idx] = {
            "ranges": ranges,
            "tau": tau,
            "lambda": lambda_est,
            "num_group_samples": int(len(metrics)),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
        }

    return CalibrationResult(block_stats=block_stats, config=cfg)


def quick_eval_allocate_block_bits(
    model: nn.Module,
    quick_loader: Iterable[Any],
    calibration: CalibrationResult,
    device: torch.device,
    max_quick_batches: int = 8,
    cfg: Optional[PTQConfig] = None,
) -> QuickEvalResult:
    """Phase-3 quick eval: estimate high-sensitivity ratio and assign block bits."""

    cfg = cfg or calibration.config
    layer_metrics = collect_layer_group_metrics(
        model=model,
        data_loader=quick_loader,
        device=device,
        max_batches=max_quick_batches,
        cfg=cfg,
    )

    block_high_ratio: Dict[int, float] = {}
    block_bits: Dict[int, int] = {}

    for layer_idx, metrics in layer_metrics.items():
        block_stat = calibration.block_stats.get(layer_idx)
        if block_stat is None or len(metrics) == 0:
            block_high_ratio[layer_idx] = 0.0
            block_bits[layer_idx] = cfg.low_bit
            continue

        tau = float(block_stat["tau"])
        scores = [score_metrics(m, block_stat, cfg) for m in metrics]
        high_ratio = float(np.mean([1.0 if s > tau else 0.0 for s in scores]))

        block_high_ratio[layer_idx] = high_ratio
        block_bits[layer_idx] = cfg.high_bit if high_ratio >= cfg.quick_high_ratio_threshold else cfg.low_bit

    return QuickEvalResult(block_bits=block_bits, block_high_ratio=block_high_ratio)
