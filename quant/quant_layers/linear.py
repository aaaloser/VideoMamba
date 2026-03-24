from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def pack_int4_signed(q_int8: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 tensor [-8, 7] into uint8 (two values per byte)."""
    q = q_int8.to(torch.int16)
    q = torch.clamp(q, -8, 7)
    q_u = (q + 8).to(torch.uint8)

    flat = q_u.flatten()
    if flat.numel() % 2 == 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)], dim=0)

    lo = flat[0::2] & 0x0F
    hi = (flat[1::2] & 0x0F) << 4
    packed = lo | hi
    return packed


def unpack_int4_signed(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack uint8 packed int4 to int8 signed tensor in [-8, 7]."""
    p = packed.flatten().to(torch.uint8)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    q_u = torch.empty(p.numel() * 2, dtype=torch.uint8, device=p.device)
    q_u[0::2] = lo
    q_u[1::2] = hi
    q_u = q_u[:numel]
    q = q_u.to(torch.int16) - 8
    return q.to(torch.int8)


class QuantizedLinear(nn.Module):
    """Weight-only quantized linear layer with on-the-fly dequantization."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        out_features: int,
        in_features: int,
        bits: int,
        packed_int4: bool,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bits = int(bits)
        self.packed_int4 = bool(packed_int4)

        self.register_buffer("qweight", packed_weight.contiguous())
        self.register_buffer("scale", scale.contiguous())
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.contiguous())

    def _dequant_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if self.bits <= 4 and self.packed_int4:
            q = unpack_int4_signed(self.qweight, self.out_features * self.in_features)
            q = q.view(self.out_features, self.in_features)
        else:
            q = self.qweight.view(self.out_features, self.in_features).to(torch.int8)

        qf = q.to(dtype)
        w = qf * self.scale.to(dtype)
        return w

    @property
    def weight(self) -> torch.Tensor:
        # Keep compatibility with code paths that directly read `.weight`
        # (e.g. mamba_simple uses `self.in_proj.weight @ x`).
        if self.qweight.is_cuda and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        else:
            target_dtype = self.scale.dtype
        return self._dequant_weight(target_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequant_weight(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


def quantize_linear_weight(
    weight: torch.Tensor,
    bits: int,
    per_channel: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Quantize linear weight to signed integer codes with symmetric scales."""
    bits = max(2, int(bits))
    qmax = int((1 << (bits - 1)) - 1)
    qmin = -qmax - 1

    if per_channel:
        scale = weight.detach().abs().amax(dim=1, keepdim=True) / float(qmax)
    else:
        scale = weight.detach().abs().amax() / float(qmax)

    scale = torch.clamp(scale, min=eps)
    q = torch.clamp(torch.round(weight / scale), qmin, qmax).to(torch.int8)
    return q, scale, qmin, qmax
