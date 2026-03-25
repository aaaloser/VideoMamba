from __future__ import annotations

from typing import Any, Dict

from torch import nn

from .helpers import iter_mamba_mixers
from .runtime import apply_weight_only_quantized_projections_


def build_uniform_block_bits(model: nn.Module, bit: int) -> Dict[int, int]:
    """Build a uniform block->bit mapping for all VideoMamba mixers."""
    uniform_bit = int(bit)
    return {layer_idx: uniform_bit for layer_idx, _ in iter_mamba_mixers(model)}


def apply_uniform_global_weight_only(
    model: nn.Module,
    bit: int,
    pack_int4: bool = True,
) -> Dict[str, Any]:
    """Apply uniform real weight-only quantization to all mixer in/out projections."""
    uniform_bit = int(bit)
    block_bits = build_uniform_block_bits(model, uniform_bit)
    summary = apply_weight_only_quantized_projections_(
        model=model,
        block_bits=block_bits,
        default_bit=uniform_bit,
        pack_int4=pack_int4,
    )
    return {
        "block_bits": block_bits,
        "summary": summary,
    }
