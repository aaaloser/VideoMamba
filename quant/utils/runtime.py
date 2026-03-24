from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from quant.config import CalibrationResult, PTQConfig
from quant.quant_layers import (
    fake_quantize_mamba_weights_ as _fake_quantize_mamba_weights,
    quantize_symmetric_to_int,
    quant_dequant_symmetric,
    resolve_block_bit,
    restore_mamba_weights_ as _restore_mamba_weights,
)
from quant.utils.helpers import (
    group_boundaries,
    infer_temporal_steps,
    iter_mamba_mixers,
    merge_cls_and_tokens,
    reshape_to_spatiotemporal,
    split_cls_and_tokens,
)


def fake_quantize_mamba_weights_(
    model: nn.Module,
    block_bits: Dict[Any, int],
    default_bit: int = 8,
) -> Dict[Tuple[int, str], torch.Tensor]:
    return _fake_quantize_mamba_weights(
        model=model,
        iter_mamba_mixers=iter_mamba_mixers(model),
        block_bits=block_bits,
        default_bit=default_bit,
    )


def restore_mamba_weights_(model: nn.Module, backup: Dict[Tuple[int, str], torch.Tensor]) -> None:
    _restore_mamba_weights(iter_mamba_mixers=iter_mamba_mixers(model), backup=backup)


def export_quantized_mamba_checkpoint(
    model: nn.Module,
    block_bits: Dict[Any, int],
    save_path: str,
    default_bit: int = 8,
    include_non_quantized_state: bool = True,
) -> Dict[str, Any]:
    """Export quantized model weights with integer storage and scale metadata.

    Integer storage convention:
    - 8-bit tensors are stored as int8 codes.
    - <=4-bit tensors are also stored in int8 container, with `bits` metadata.
    """
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

    quantized_tensors: Dict[str, Dict[str, Any]] = {}
    quantized_keys: set[str] = set()

    for layer_idx, mixer in iter_mamba_mixers(model):
        bits = resolve_block_bit(block_bits, layer_idx, default_bit)
        named_params = dict(mixer.named_parameters())
        for name in target_names:
            if name not in named_params:
                continue

            p = named_params[name]
            if not torch.is_floating_point(p.data):
                continue

            per_channel = p.ndim >= 2 and name.endswith("weight")
            q_int, scale, qmin, qmax = quantize_symmetric_to_int(
                p.data,
                bits=bits,
                per_channel=per_channel,
                channel_dim=0,
            )

            full_key = f"layers.{layer_idx}.mixer.{name}"
            quantized_keys.add(full_key)
            quantized_tensors[full_key] = {
                "q": q_int.detach().cpu(),
                "scale": scale.detach().cpu().to(torch.float32),
                "bits": int(bits),
                "qmin": int(qmin),
                "qmax": int(qmax),
                "per_channel": bool(per_channel),
                "channel_dim": 0,
                "shape": list(p.shape),
            }

    non_quantized_state: Dict[str, torch.Tensor] = {}
    if include_non_quantized_state:
        for key, tensor in model.state_dict().items():
            if key not in quantized_keys:
                non_quantized_state[key] = tensor.detach().cpu()

    payload: Dict[str, Any] = {
        "format": "videomamba_ptq_int_v1",
        "block_bits": {str(k): int(v) for k, v in block_bits.items()},
        "quantized_tensors": quantized_tensors,
        "non_quantized_state": non_quantized_state,
    }

    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    torch.save(payload, save_path)
    return payload


class PTQRuntimeActivationHook:
    """Phase-4 runtime activation fake-quant with error forwarding / residual reuse."""

    def __init__(
        self,
        model: nn.Module,
        block_bits: Dict[Any, int],
        calibration: CalibrationResult,
        cfg: Optional[PTQConfig] = None,
    ):
        self.model = model
        self.block_bits = block_bits
        self.calibration = calibration
        self.cfg = cfg or calibration.config
        self._handles: List[Any] = []
        self._state: Dict[int, Dict[str, Optional[torch.Tensor]]] = {}

    def _reset_state(self) -> None:
        self._state.clear()
        for idx, _ in iter_mamba_mixers(self.model):
            self._state[idx] = {"carry": None, "anchor": None}

    def _layer_lambda(self, layer_idx: int) -> float:
        stat = self.calibration.block_stats.get(layer_idx, {})
        return float(stat.get("lambda", self.cfg.default_lambda))

    def _apply_group_quantization(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        bits: int,
    ) -> torch.Tensor:
        state = self._state.setdefault(layer_idx, {"carry": None, "anchor": None})

        cls_token, x_tokens, cls_idx = split_cls_and_tokens(hidden_states, self.cfg.cls_token_position)
        t_steps = infer_temporal_steps(self.model, x_tokens.shape[1], self.cfg)
        x_st, _ = reshape_to_spatiotemporal(x_tokens, t_steps)

        lam = self._layer_lambda(layer_idx)
        groups = group_boundaries(x_st.shape[1], self.cfg.num_groups)
        out_groups: List[torch.Tensor] = []

        if bits >= 8:
            carry = state.get("carry")
            for start, end in groups:
                xg = x_st[:, start:end, :, :]
                xq = quant_dequant_symmetric(xg, bits=bits)
                err = xg - xq
                if carry is not None and carry.shape == xg.shape:
                    xq = xq + lam * carry
                out_groups.append(xq)
                carry = err.detach()
            state["carry"] = carry
            state["anchor"] = None
        elif bits <= 4:
            anchor = state.get("anchor")
            for start, end in groups:
                xg = x_st[:, start:end, :, :]
                if anchor is None or anchor.shape != xg.shape:
                    anchor = torch.zeros_like(xg)
                delta = xg - anchor
                delta_q = quant_dequant_symmetric(delta, bits=bits)
                yg = anchor + delta_q
                out_groups.append(yg)
                anchor = yg.detach()
            state["anchor"] = anchor
            state["carry"] = None
        else:
            for start, end in groups:
                xg = x_st[:, start:end, :, :]
                out_groups.append(quant_dequant_symmetric(xg, bits=bits))
            state["carry"] = None
            state["anchor"] = None

        xq_st = torch.cat(out_groups, dim=1)
        xq_tokens = xq_st.view(x_tokens.shape)
        return merge_cls_and_tokens(cls_token, xq_tokens, cls_idx)

    def attach(self) -> None:
        self.detach()
        self._reset_state()

        def _model_pre_hook(_module: nn.Module, _inputs: Tuple[Any, ...]) -> None:
            self._reset_state()

        self._handles.append(self.model.register_forward_pre_hook(_model_pre_hook))

        for layer_idx, mixer in iter_mamba_mixers(self.model):
            def _make_pre_hook(idx: int):
                def _pre_hook(_module: nn.Module, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
                    if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
                        return inputs
                    bits = resolve_block_bit(self.block_bits, idx, self.cfg.default_bit)
                    hidden_states = inputs[0]
                    xq = self._apply_group_quantization(idx, hidden_states, bits)
                    return (xq,) + inputs[1:]

                return _pre_hook

            self._handles.append(mixer.register_forward_pre_hook(_make_pre_hook(layer_idx)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._state.clear()


class VideoMambaPTQSession:
    """Utility holder to manage fake-quant deployment lifecycle."""

    def __init__(
        self,
        model: nn.Module,
        weight_backup: Dict[Tuple[int, str], torch.Tensor],
        runtime_hook: PTQRuntimeActivationHook,
    ):
        self.model = model
        self.weight_backup = weight_backup
        self.runtime_hook = runtime_hook

    def close(self) -> None:
        self.runtime_hook.detach()
        restore_mamba_weights_(self.model, self.weight_backup)


def apply_videomamba_ptq(
    model: nn.Module,
    block_bits: Dict[Any, int],
    calibration: CalibrationResult,
    cfg: Optional[PTQConfig] = None,
) -> VideoMambaPTQSession:
    """Phase-4 entrypoint: apply weight fake-quant and activation runtime hook."""

    cfg = cfg or calibration.config
    backup = fake_quantize_mamba_weights_(model, block_bits, default_bit=cfg.default_bit)
    runtime_hook = PTQRuntimeActivationHook(model, block_bits, calibration, cfg)
    runtime_hook.attach()
    return VideoMambaPTQSession(model=model, weight_backup=backup, runtime_hook=runtime_hook)
