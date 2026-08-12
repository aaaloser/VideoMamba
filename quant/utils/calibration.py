from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn

from quant.config import CalibrationResult, PTQConfig, QuickEvalResult
from quant.utils.helpers import (
    collect_layer_group_metrics,
    collect_layer_lambda_error,
    safe_float_cpu,
    score_metrics,
)


def calibrate_videomamba_ptq(
    model: nn.Module,
    calib_loader: Iterable[Any],
    device: torch.device,
    calib_size: Optional[int] = None,
    calib_batch_size: Optional[int] = None,
    cfg: Optional[PTQConfig] = None,
    max_calib_batches: Optional[int] = None,
) -> CalibrationResult:
    """Phase-2 calibration: collect ranges and tau/lambda for each Mamba block.

    ``calib_size`` is the total number of calibration samples; loader batches
    are split into ``calib_batch_size`` sub-batches per forward to keep peak
    activation memory low. ``max_calib_batches`` is kept as a deprecated legacy
    alias (number of loader batches).
    """

    cfg = cfg or PTQConfig()
    if calib_size is not None:
        layer_metrics, layer_lambda, _ = collect_layer_group_metrics(
            model=model,
            data_loader=calib_loader,
            device=device,
            cfg=cfg,
            calib_size=calib_size,
            calib_batch_size=calib_batch_size,
        )
    else:
        legacy_batches = max_calib_batches if max_calib_batches is not None else 16
        layer_metrics, layer_lambda, _ = collect_layer_group_metrics(
            model=model,
            data_loader=calib_loader,
            device=device,
            max_batches=legacy_batches,
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

        lambda_samples = layer_lambda.get(layer_idx)
        if lambda_samples:
            lambda_est = float(np.median(np.asarray(lambda_samples, dtype=np.float32)))
        else:
            lambda_est = float(cfg.default_lambda)
        lambda_est = float(np.clip(lambda_est, -1.0, 1.0))

        block_stats[layer_idx] = {
            "ranges": ranges,
            "tau": tau,
            "lambda": lambda_est,
            "lambda_error": lambda_est,
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
    max_quick_batches: int = 50,
    cfg: Optional[PTQConfig] = None,
) -> QuickEvalResult:
    """Phase-3 quick eval (TimeSformer step-3): pure-statistics runtime routing.

    For each quick-eval batch, the model runs FP32 forward (routed_input =
    original input, no actual quantization). Per-group sensitivity scores are
    computed using the static calibration profiles (ranges + tau); groups with
    score > tau are counted as high-sensitivity. The block-level average
    high-sensitivity ratio (avg_ratio_b) drives rank-based bit allocation
    (step-4). Also reports quick_top1 (FP32 accuracy on quick-eval batches)
    and per-block call counts (calls_b).
    """

    cfg = cfg or calibration.config
    layer_metrics, _, predictions = collect_layer_group_metrics(
        model=model,
        data_loader=quick_loader,
        device=device,
        max_batches=max_quick_batches,
        cfg=cfg,
        return_predictions=True,
    )

    # quick_top1: FP32 model accuracy on quick-eval batches (step-3-6)
    quick_top1: Optional[float] = None
    if predictions:
        correct = sum(
            (output.argmax(dim=-1) == label).sum().item()
            for output, label in predictions
        )
        total = sum(label.numel() for _, label in predictions)
        quick_top1 = 100.0 * correct / max(total, 1)

    block_high_ratio: Dict[int, float] = {}
    block_calls: Dict[int, int] = {}

    for layer_idx, metrics in layer_metrics.items():
        block_stat = calibration.block_stats.get(layer_idx)
        num_groups = max(cfg.num_groups, 1)
        calls_b = max(1, len(metrics) // num_groups)
        block_calls[layer_idx] = calls_b

        if block_stat is None or len(metrics) == 0:
            block_high_ratio[layer_idx] = 0.0
            continue

        tau = float(block_stat["tau"])
        scores = [score_metrics(m, block_stat, cfg) for m in metrics]
        high_ratio = float(np.mean([1.0 if s > tau else 0.0 for s in scores]))

        block_high_ratio[layer_idx] = high_ratio

    block_bits, block_rank, n_high = allocate_block_bits(block_high_ratio, cfg)
    return QuickEvalResult(
        block_bits=block_bits,
        block_high_ratio=block_high_ratio,
        block_rank=block_rank,
        n_high=n_high,
        quick_top1=quick_top1,
        block_calls=block_calls,
    )


def allocate_block_bits_by_rank(
    block_high_ratio: Dict[int, float],
    cfg: PTQConfig,
):
    """TimeSformer step-4: rank-based bit allocation.

    Sort blocks by avg high-sensitivity ratio descending; the top
    ``high_block_fraction`` fraction receives ``high_bit``, the rest receive
    ``low_bit``. Insensitive to the absolute scale of the calibration
    threshold, avoiding all-8/all-4 extremes.
    """
    ordered = sorted(block_high_ratio.items(), key=lambda kv: (-kv[1], kv[0]))
    n_total = len(ordered)
    n_high = int(round(n_total * cfg.high_block_fraction))
    n_high = max(0, min(n_high, n_total))

    block_bits: Dict[int, int] = {}
    block_rank: Dict[int, int] = {}
    for rank, (layer_idx, _ratio) in enumerate(ordered):
        block_rank[layer_idx] = rank
        block_bits[layer_idx] = cfg.high_bit if rank < n_high else cfg.low_bit
    return block_bits, block_rank, n_high


def allocate_block_bits_by_threshold(
    block_high_ratio: Dict[int, float],
    cfg: PTQConfig,
):
    """Legacy threshold-based allocation (ratio >= threshold -> high_bit)."""
    block_bits: Dict[int, int] = {}
    for layer_idx, ratio in block_high_ratio.items():
        block_bits[layer_idx] = cfg.high_bit if ratio >= cfg.quick_high_ratio_threshold else cfg.low_bit
    return block_bits, {}, 0


def allocate_block_bits(
    block_high_ratio: Dict[int, float],
    cfg: PTQConfig,
):
    """Dispatch block bit allocation by ``cfg.bit_allocation_mode``."""
    mode = cfg.bit_allocation_mode.lower()
    if mode == "rank":
        return allocate_block_bits_by_rank(block_high_ratio, cfg)
    if mode == "threshold":
        return allocate_block_bits_by_threshold(block_high_ratio, cfg)
    raise ValueError(f"Unsupported bit_allocation_mode: {cfg.bit_allocation_mode}")
