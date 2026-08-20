from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from quant.quant_layers.quantization_ops import quant_dequant_symmetric
from quant.utils.helpers import extract_video_tensor, iter_mamba_mixers


class _PerTensorMinmaxObserver:
    def __init__(self, n_bits: int = 8, clip_ratio: float = 1.0) -> None:
        self.n_bits = n_bits
        self.clip_ratio = clip_ratio
        self.w_max: Optional[torch.Tensor] = None

    def update(self, x: torch.Tensor) -> None:
        cur = x.detach().float().abs().amax().clamp(min=1e-5)
        if self.w_max is None:
            self.w_max = cur
        else:
            self.w_max = torch.maximum(self.w_max, cur)

    def scale(self) -> torch.Tensor:
        if self.w_max is None:
            return torch.tensor(1.0, dtype=torch.float32)
        qmax = float((1 << (self.n_bits - 1)) - 1)
        return (self.w_max * self.clip_ratio / qmax).to(torch.float32).clamp(min=1e-6)


class _PerTensorPercentileObserver:
    def __init__(self, n_bits: int = 8, clip_ratio: float = 1.0, alpha: float = 0.9995) -> None:
        self.n_bits = n_bits
        self.clip_ratio = clip_ratio
        self.alpha = alpha
        self.w_max: Optional[torch.Tensor] = None

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float().reshape(-1)
        if x.numel() == 0:
            return
        cur = torch.quantile(x.abs(), self.alpha).clamp(min=1e-5)
        if self.w_max is None:
            self.w_max = cur
        else:
            self.w_max = torch.maximum(self.w_max, cur)

    def scale(self) -> torch.Tensor:
        if self.w_max is None:
            return torch.tensor(1.0, dtype=torch.float32)
        qmax = float((1 << (self.n_bits - 1)) - 1)
        return (self.w_max * self.clip_ratio / qmax).to(torch.float32).clamp(min=1e-6)


@dataclass
class QuambaCalibrationResult:
    act_scales: Dict[int, Dict[str, torch.Tensor]]
    block_stats: Dict[int, Dict[str, Any]]


@dataclass
class QuambaPTQBridgeSession:
    model: nn.Module
    weight_backup: Dict[Tuple[int, str], torch.Tensor]

    def close(self) -> None:
        for (layer_idx, name), tensor in self.weight_backup.items():
            mixer = None
            for idx, candidate in iter_mamba_mixers(self.model):
                if idx == layer_idx:
                    mixer = candidate
                    break
            if mixer is None:
                continue
            module_name, param_name = name.split(".", 1)
            module = getattr(mixer, module_name, None)
            if module is None:
                continue
            target = getattr(module, param_name, None)
            if target is None or not torch.is_tensor(target):
                continue
            target.data.copy_(tensor)


def build_uniform_block_bits(model: nn.Module, default_bit: int = 4) -> Dict[int, int]:
    return {layer_idx: int(default_bit) for layer_idx, _ in iter_mamba_mixers(model)}


def _resolve_block_bit(block_bits: Dict[Any, int], layer_idx: int, default_bit: int) -> int:
    if layer_idx in block_bits:
        return int(block_bits[layer_idx])
    key = str(layer_idx)
    if key in block_bits:
        return int(block_bits[key])
    return int(default_bit)


def _iter_calib_tensors(
    calib_loader: Iterable[Any],
    max_calib_batches: int = 16,
    calib_size: Optional[int] = None,
    calib_batch_size: Optional[int] = None,
) -> Iterable[torch.Tensor]:
    seen = 0
    for bidx, batch in enumerate(calib_loader):
        x = extract_video_tensor(batch)

        if calib_size is not None and calib_size > 0:
            sub_bs = int(calib_batch_size) if calib_batch_size and calib_batch_size > 0 else x.shape[0]
            for start in range(0, x.shape[0], sub_bs):
                if seen >= calib_size:
                    return
                chunk = x[start:start + sub_bs]
                if seen + chunk.shape[0] > calib_size:
                    chunk = chunk[:calib_size - seen]
                yield chunk
                seen += int(chunk.shape[0])
        else:
            if bidx >= max_calib_batches:
                break
            yield x


@torch.no_grad()
def calibrate_quamba_bridge(
    model: nn.Module,
    calib_loader: Iterable[Any],
    device: torch.device,
    max_calib_batches: int = 16,
    a_bits: int = 8,
    percentile_alpha: float = 0.9995,
    calib_size: Optional[int] = None,
    calib_batch_size: Optional[int] = None,
) -> QuambaCalibrationResult:
    model.eval()

    module_names = ["in_proj", "x_proj", "dt_proj", "out_proj"]
    obs_map: Dict[int, Dict[str, Any]] = {}
    hook_handles: List[Any] = []

    for layer_idx, mixer in iter_mamba_mixers(model):
        obs_map[layer_idx] = {}
        for name in module_names:
            m = getattr(mixer, name, None)
            if m is None:
                continue

            in_obs: Any
            if name in ("x_proj",):
                in_obs = _PerTensorPercentileObserver(n_bits=a_bits, alpha=percentile_alpha)
            else:
                in_obs = _PerTensorMinmaxObserver(n_bits=a_bits)
            out_obs = _PerTensorMinmaxObserver(n_bits=a_bits)
            obs_map[layer_idx][f"{name}:input"] = in_obs
            obs_map[layer_idx][f"{name}:output"] = out_obs

            def _hook(_m: nn.Module, inputs: Tuple[Any, ...], outputs: Any, n: str = name, lidx: int = layer_idx) -> None:
                x = inputs[0] if isinstance(inputs, tuple) and len(inputs) > 0 else inputs
                y = outputs[0] if isinstance(outputs, tuple) and len(outputs) > 0 else outputs
                if torch.is_tensor(x):
                    obs_map[lidx][f"{n}:input"].update(x)
                if torch.is_tensor(y):
                    obs_map[lidx][f"{n}:output"].update(y)

            hook_handles.append(m.register_forward_hook(_hook))

    for videos in _iter_calib_tensors(
        calib_loader,
        max_calib_batches=max_calib_batches,
        calib_size=calib_size,
        calib_batch_size=calib_batch_size,
    ):
        videos = videos.to(device, non_blocking=True)
        _ = model(videos)

    for h in hook_handles:
        h.remove()

    act_scales: Dict[int, Dict[str, torch.Tensor]] = {}
    block_stats: Dict[int, Dict[str, Any]] = {}
    for layer_idx, obs in obs_map.items():
        act_scales[layer_idx] = {}
        for name, observer in obs.items():
            act_scales[layer_idx][name] = observer.scale().cpu()

        x_scale = float(act_scales[layer_idx].get("x_proj:input", torch.tensor(1.0)).item())
        out_scale = float(act_scales[layer_idx].get("out_proj:input", torch.tensor(1.0)).item())
        block_stats[layer_idx] = {
            "method": "quamba_bridge_calib",
            "x_proj_input_scale": x_scale,
            "out_proj_input_scale": out_scale,
        }

    return QuambaCalibrationResult(act_scales=act_scales, block_stats=block_stats)


@torch.no_grad()
def apply_quamba_bridge_ptq(
    model: nn.Module,
    block_bits: Dict[Any, int],
    calibration: Optional[QuambaCalibrationResult] = None,
    default_bit: int = 4,
) -> QuambaPTQBridgeSession:
    del calibration  # The current bridge uses calibration for reporting only.

    target_names = [
        "in_proj.weight",
        "in_proj.bias",
        "conv1d.weight",
        "conv1d.bias",
        "x_proj.weight",
        "x_proj.bias",
        "dt_proj.weight",
        "dt_proj.bias",
        "out_proj.weight",
        "out_proj.bias",
        "conv1d_b.weight",
        "conv1d_b.bias",
        "x_proj_b.weight",
        "x_proj_b.bias",
        "dt_proj_b.weight",
        "dt_proj_b.bias",
    ]

    weight_backup: Dict[Tuple[int, str], torch.Tensor] = {}
    for layer_idx, mixer in iter_mamba_mixers(model):
        bits = _resolve_block_bit(block_bits, layer_idx, default_bit)
        for name in target_names:
            module_name, param_name = name.split(".", 1)
            module = getattr(mixer, module_name, None)
            if module is None:
                continue
            tensor = getattr(module, param_name, None)
            if tensor is None or not torch.is_tensor(tensor):
                continue
            if not torch.is_floating_point(tensor):
                continue

            weight_backup[(layer_idx, name)] = tensor.data.detach().clone()
            per_channel = bool(param_name == "weight" and tensor.ndim >= 2)
            tensor.data.copy_(
                quant_dequant_symmetric(
                    tensor.data,
                    bits=bits,
                    per_channel=per_channel,
                    channel_dim=0,
                )
            )

    return QuambaPTQBridgeSession(model=model, weight_backup=weight_backup)
