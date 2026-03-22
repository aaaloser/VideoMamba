from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
from torch import nn


def quant_dequant_symmetric(
    x: torch.Tensor,
    bits: int,
    per_channel: bool = False,
    channel_dim: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    bits = max(2, int(bits))
    qmax = float((1 << (bits - 1)) - 1)
    qmin = -qmax - 1

    if per_channel and x.ndim > 1:
        reduce_dims = [d for d in range(x.ndim) if d != channel_dim]
        scale = x.detach().abs().amax(dim=reduce_dims, keepdim=True) / qmax
    else:
        scale = x.detach().abs().amax() / qmax

    scale = torch.clamp(scale, min=eps)
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q * scale


def resolve_block_bit(block_bits: Dict[Any, int], layer_idx: int, default_bit: int) -> int:
    for key in (layer_idx, str(layer_idx), f"layers.{layer_idx}"):
        if key in block_bits:
            return int(block_bits[key])
    return int(default_bit)


def fake_quantize_mamba_weights_(
    model: nn.Module,
    iter_mamba_mixers: Iterable[Tuple[int, nn.Module]],
    block_bits: Dict[Any, int],
    default_bit: int = 8,
) -> Dict[Tuple[int, str], torch.Tensor]:
    """In-place fake quantization for Mamba block weights. Returns a restore backup."""

    backup: Dict[Tuple[int, str], torch.Tensor] = {}
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

    for layer_idx, mixer in iter_mamba_mixers:
        bits = resolve_block_bit(block_bits, layer_idx, default_bit)
        named_params = dict(mixer.named_parameters())
        for name in target_names:
            if name not in named_params:
                continue
            p = named_params[name]
            if not torch.is_floating_point(p.data):
                continue
            backup[(layer_idx, name)] = p.data.detach().clone()
            per_channel = p.ndim >= 2 and name.endswith("weight")
            q = quant_dequant_symmetric(
                p.data,
                bits=bits,
                per_channel=per_channel,
                channel_dim=0,
            )
            p.data.copy_(q)

    return backup


def restore_mamba_weights_(
    iter_mamba_mixers: Iterable[Tuple[int, nn.Module]],
    backup: Dict[Tuple[int, str], torch.Tensor],
) -> None:
    if len(backup) == 0:
        return

    mixers = {idx: mixer for idx, mixer in iter_mamba_mixers}
    for (layer_idx, name), original in backup.items():
        mixer = mixers.get(layer_idx)
        if mixer is None:
            continue
        params = dict(mixer.named_parameters())
        if name in params:
            params[name].data.copy_(original)
