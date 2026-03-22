from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from quant.config import CalibrationResult, PTQConfig
from quant.quant_layers import (
    fake_quantize_mamba_weights_ as _fake_quantize_mamba_weights,
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
