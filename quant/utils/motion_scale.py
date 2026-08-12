"""Step-5 motion-aware per-output-channel scale optimization (TimeSformer route).

For each quantized in_proj, capture its calibration input (the mixer
hidden_states), compute the FP output Y_fp = W @ X, derive per-frame motion
weights, then search 32 candidate scales (0.5x..1.5x of per-channel MinMax)
and pick the one minimizing motion-weighted reconstruction error per channel.

out_proj input lives inside the Mamba SSM core (accessed via ``.weight`` in
``F.linear``, not a module call) so it cannot be captured with hooks; it falls
back to per-channel MinMax. This is a deliberate adaptation of the TimeSformer
method to VideoMamba's architecture.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from quant.config import PTQConfig
from quant.quant_layers import QuantizedLinear, pack_int4_signed, resolve_block_bit
from quant.utils.helpers import (
    extract_video_tensor,
    infer_temporal_steps,
    iter_calib_tensors,
    iter_mamba_mixers,
    resolve_cls_index,
)


def _minmax_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    lo, hi = float(x.min().item()), float(x.max().item())
    if abs(hi - lo) < eps:
        return torch.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


def compute_motion_frame_weights(
    y: torch.Tensor,
    eta: float = 0.5,
    tau_m: float = 0.7,
    rho: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Frame weights from motion intensity and redundancy (TimeSformer 5-3..5-5).

    y: [B, T, S, O] -> w_t: [T], sum to 1.
    """
    B, T, S, O = y.shape
    if T <= 1:
        return torch.ones(T, device=y.device, dtype=y.dtype) / max(T, 1)

    motion = (y[:, 1:] - y[:, :-1]).abs().mean(dim=(0, 2, 3))  # [T-1]
    motion = _minmax_norm(motion, eps)

    favg = y.mean(dim=(0, 2))  # [T, O]
    cos = F.cosine_similarity(favg[:-1], favg[1:], dim=-1)  # [T-1]
    redun = (cos + 1.0) / 2.0

    score = motion - eta * redun  # [T-1]
    score_t = torch.empty(T, device=y.device, dtype=y.dtype)
    score_t[0] = score[0]  # first frame reuses first pair (= second frame's score)
    score_t[1:] = score

    uniform = torch.full((T,), 1.0 / T, device=y.device, dtype=y.dtype)
    sm = F.softmax(score_t / tau_m, dim=0)
    w = rho * uniform + (1.0 - rho) * sm
    return w / w.sum()


def compute_motion_aware_scale(
    weight: torch.Tensor,
    activation: torch.Tensor,
    t_steps: int,
    bits: int = 4,
    n_candidates: int = 32,
    eta: float = 0.5,
    tau_m: float = 0.7,
    rho: float = 0.2,
    eps: float = 1e-8,
    cls_token_position: str = "auto",
) -> torch.Tensor:
    """Per-output-channel scale minimizing motion-weighted reconstruction error.

    weight: [out, in] fp32; activation: [B, L, in] fp32 (in_proj input).
    Returns best_scale: [out] fp32.
    """
    weight = weight.detach().float()
    activation = activation.detach().float()
    out, in_feat = weight.shape
    B, L, _ = activation.shape

    cls_idx = resolve_cls_index(L, cls_token_position)
    if cls_idx is not None:
        x = torch.cat([activation[:, :cls_idx], activation[:, cls_idx + 1:]], dim=1)
        n_tokens = L - 1
    else:
        x = activation
        n_tokens = L

    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax - 1

    # degenerate -> T=1 (uniform weight, standard L2 search)
    if t_steps is None or t_steps <= 1 or n_tokens <= 0 or n_tokens % t_steps != 0:
        t_eff, s_eff = 1, n_tokens
    else:
        t_eff, s_eff = t_steps, n_tokens // t_steps

    x = x[:, : t_eff * s_eff].contiguous()  # [B, n_tokens, in]
    y_fp = F.linear(x, weight).view(B, t_eff, s_eff, out)  # [B, T, S, O]
    w_t = compute_motion_frame_weights(y_fp, eta, tau_m, rho, eps)  # [T]

    base = weight.abs().amax(dim=1).clamp(min=eps) / float(qmax)  # [out]
    factors = torch.linspace(0.5, 1.5, n_candidates, device=weight.device, dtype=weight.dtype)
    cand_scales = base.unsqueeze(1) * factors.unsqueeze(0)  # [out, n_c]

    weighted_err = torch.zeros(n_candidates, out, device=weight.device, dtype=torch.float32)
    for ci in range(n_candidates):
        s = cand_scales[:, ci]  # [out]
        wq = torch.clamp(torch.round(weight / s.unsqueeze(1)), qmin, qmax)
        wdq = wq * s.unsqueeze(1)  # [out, in]
        yq = F.linear(x, wdq).view(B, t_eff, s_eff, out)  # [B, T, S, O]
        err = (y_fp - yq).abs().mean(dim=(0, 2))  # [T, out]
        weighted_err[ci] = (w_t.unsqueeze(1) * err).sum(dim=0)  # [out]

    best_ci = weighted_err.argmin(dim=0)  # [out]
    best_scale = cand_scales[torch.arange(out, device=weight.device), best_ci]  # [out]
    return best_scale.float()


def capture_in_proj_activations(
    model: nn.Module,
    data_loader: Iterable[Any],
    device: torch.device,
    max_batches: Optional[int] = None,
    cfg: Optional[PTQConfig] = None,
    calib_size: Optional[int] = None,
    calib_batch_size: Optional[int] = None,
) -> Dict[int, torch.Tensor]:
    """Capture in_proj input (= mixer hidden_states) per layer via forward pre-hook.

    Either ``max_batches`` (legacy: number of loader batches) or ``calib_size``
    (total samples, split into ``calib_batch_size`` sub-batches per forward)
    must be given.
    """
    activations: Dict[int, list] = {}
    hook_handles = []

    def _make_hook(layer_idx: int):
        def _hook(module: nn.Module, inputs: Tuple[Any, ...]) -> None:
            if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
                return
            activations.setdefault(layer_idx, []).append(inputs[0].detach())

        return _hook

    if calib_size is not None:
        batch_iter: Iterable[Tuple[torch.Tensor, Optional[torch.Tensor]]] = iter_calib_tensors(
            data_loader, device, calib_size, calib_batch_size
        )
    elif max_batches is not None:
        batch_iter = (
            (extract_video_tensor(batch).to(device, non_blocking=True), None)
            for bidx, batch in enumerate(data_loader)
            if bidx < max_batches
        )
    else:
        raise ValueError("capture_in_proj_activations requires either max_batches or calib_size")

    model.eval()
    with torch.no_grad():
        for idx, _mixer in iter_mamba_mixers(model):
            hook_handles.append(_mixer.register_forward_pre_hook(_make_hook(idx)))
        for video, _label in batch_iter:
            _ = model(video)

    for h in hook_handles:
        h.remove()

    out_act: Dict[int, torch.Tensor] = {}
    for layer_idx, tensors in activations.items():
        out_act[layer_idx] = torch.cat(tensors, dim=0).to(device)
    return out_act


def compute_motion_aware_in_proj_scales(
    model: nn.Module,
    block_bits: Dict[Any, int],
    activations: Dict[int, torch.Tensor],
    cfg: PTQConfig,
    default_bit: int = 8,
    n_candidates: int = 32,
    eta: float = 0.5,
    tau_m: float = 0.7,
    rho: float = 0.2,
    eps: float = 1e-8,
) -> Dict[int, torch.Tensor]:
    """Precompute per-layer in_proj motion-aware scales.

    Returns a dict ``{layer_idx: scale_cpu[out]}`` (CPU tensors) so the result
    can be broadcast across DDP ranks without device mismatch. Layers without a
    captured activation are skipped (the apply step will MinMax-fallback them).
    """
    scales: Dict[int, torch.Tensor] = {}
    for layer_idx, mixer in iter_mamba_mixers(model):
        bits = resolve_block_bit(block_bits, layer_idx, default_bit)
        hs = activations.get(layer_idx)
        proj = getattr(mixer, "in_proj", None)
        if proj is None or not isinstance(proj, nn.Linear) or hs is None:
            continue
        cls_idx = resolve_cls_index(hs.shape[1], cfg.cls_token_position)
        n_tokens = hs.shape[1] - (1 if cls_idx is not None else 0)
        t_steps = infer_temporal_steps(model, n_tokens, cfg)
        scale = compute_motion_aware_scale(
            proj.weight.data,
            hs,
            t_steps,
            bits=bits,
            n_candidates=n_candidates,
            eta=eta,
            tau_m=tau_m,
            rho=rho,
            eps=eps,
            cls_token_position=cfg.cls_token_position,
        )
        scales[layer_idx] = scale.detach().to("cpu")
    return scales


def apply_motion_aware_weight_only_(
    model: nn.Module,
    block_bits: Dict[Any, int],
    activations: Optional[Dict[int, torch.Tensor]] = None,
    cfg: Optional[PTQConfig] = None,
    default_bit: int = 8,
    pack_int4: bool = True,
    precomputed_in_proj_scales: Optional[Dict[int, torch.Tensor]] = None,
    n_candidates: int = 32,
    eta: float = 0.5,
    tau_m: float = 0.7,
    rho: float = 0.2,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """Replace in_proj with motion-aware QuantizedLinear; out_proj with MinMax.

    in_proj scale priority:
      1. ``precomputed_in_proj_scales`` (dict of CPU tensors, e.g. from DDP broadcast)
      2. ``activations`` + ``cfg`` (recompute via 32-candidate search)
      3. MinMax fallback (per-channel weight absmax / qmax)

    out_proj always uses MinMax (its input lives inside the Mamba SSM core and
    cannot be captured with hooks).
    """
    by_bit: Dict[int, int] = {}
    replaced: list = []
    motion_applied: list = []

    for layer_idx, mixer in iter_mamba_mixers(model):
        bits = resolve_block_bit(block_bits, layer_idx, default_bit)
        qmax = (1 << (bits - 1)) - 1
        qmin = -qmax - 1
        hs = activations.get(layer_idx) if activations is not None else None

        for proj_name in ("in_proj", "out_proj"):
            proj = getattr(mixer, proj_name, None)
            if proj is None or not isinstance(proj, nn.Linear):
                continue

            if proj_name == "in_proj":
                pre = None
                if precomputed_in_proj_scales is not None:
                    pre = precomputed_in_proj_scales.get(layer_idx)
                if pre is not None:
                    scale = pre.to(device=proj.weight.device)
                    motion_applied.append(f"layers.{layer_idx}.mixer.in_proj")
                elif hs is not None and cfg is not None:
                    cls_idx = resolve_cls_index(hs.shape[1], cfg.cls_token_position)
                    n_tokens = hs.shape[1] - (1 if cls_idx is not None else 0)
                    t_steps = infer_temporal_steps(model, n_tokens, cfg)
                    scale = compute_motion_aware_scale(
                        proj.weight.data,
                        hs,
                        t_steps,
                        bits=bits,
                        n_candidates=n_candidates,
                        eta=eta,
                        tau_m=tau_m,
                        rho=rho,
                        eps=eps,
                        cls_token_position=cfg.cls_token_position,
                    )
                    motion_applied.append(f"layers.{layer_idx}.mixer.in_proj")
                else:
                    scale = proj.weight.data.detach().abs().amax(dim=1).clamp(min=eps) / float(qmax)
            else:
                scale = proj.weight.data.detach().abs().amax(dim=1).clamp(min=eps) / float(qmax)

            scale = scale.view(-1, 1).float()
            q = torch.clamp(torch.round(proj.weight.data.float() / scale), qmin, qmax).to(torch.int8)
            if bits <= 4 and pack_int4:
                q_store = pack_int4_signed(q)
                packed = True
            else:
                q_store = q.contiguous()
                packed = False

            q_layer = QuantizedLinear(
                packed_weight=q_store,
                scale=scale,
                bias=proj.bias.data if proj.bias is not None else None,
                out_features=proj.out_features,
                in_features=proj.in_features,
                bits=bits,
                packed_int4=packed,
            )
            setattr(mixer, proj_name, q_layer)
            by_bit[bits] = by_bit.get(bits, 0) + 1
            replaced.append(f"layers.{layer_idx}.mixer.{proj_name}")

    return {
        "num_replaced": len(replaced),
        "by_bit": {str(k): v for k, v in sorted(by_bit.items(), key=lambda x: x[0])},
        "replaced_keys": replaced,
        "motion_applied": motion_applied,
    }
