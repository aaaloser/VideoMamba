from __future__ import annotations

from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from quant.config import PTQConfig


def extract_video_tensor(batch: Any) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (list, tuple)) and len(batch) > 0:
        if isinstance(batch[0], torch.Tensor):
            return batch[0]
    if isinstance(batch, MutableMapping):
        for key in ("video", "videos", "inputs", "input", "data"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError("Unable to extract video tensor from dataloader batch.")


def iter_mamba_mixers(model: nn.Module) -> Iterable[Tuple[int, nn.Module]]:
    if not hasattr(model, "layers"):
        raise AttributeError("Expected model to expose `layers` (VideoMamba).")
    for idx, layer in enumerate(model.layers):
        mixer = getattr(layer, "mixer", None)
        if mixer is None:
            continue
        if all(hasattr(mixer, n) for n in ("in_proj", "conv1d", "x_proj", "dt_proj", "out_proj")):
            yield idx, mixer


def safe_float_cpu(values: List[float]) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def infer_temporal_steps(model: nn.Module, seqlen_wo_cls: int, cfg: PTQConfig) -> int:
    if cfg.temporal_steps is not None:
        return max(int(cfg.temporal_steps), 1)
    temporal_pos = getattr(model, "temporal_pos_embedding", None)
    if isinstance(temporal_pos, torch.Tensor) and temporal_pos.ndim == 3:
        t = int(temporal_pos.shape[1])
        if t > 0 and seqlen_wo_cls % t == 0:
            return t
    return 1


def resolve_cls_index(seqlen: int, cls_token_position: str) -> Optional[int]:
    mode = cls_token_position.lower()
    if mode == "none":
        return None
    if mode == "first":
        return 0
    if mode == "center":
        return seqlen // 2
    if mode == "auto":
        return 0
    raise ValueError(f"Unsupported cls_token_position: {cls_token_position}")


def split_cls_and_tokens(
    x: torch.Tensor, cls_token_position: str
) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[int]]:
    cls_idx = resolve_cls_index(x.shape[1], cls_token_position)
    if cls_idx is None:
        return None, x, None
    cls_token = x[:, cls_idx : cls_idx + 1, :]
    x_tokens = torch.cat((x[:, :cls_idx, :], x[:, cls_idx + 1 :, :]), dim=1)
    return cls_token, x_tokens, cls_idx


def merge_cls_and_tokens(
    cls_token: Optional[torch.Tensor],
    x_tokens: torch.Tensor,
    cls_idx: Optional[int],
) -> torch.Tensor:
    if cls_token is None or cls_idx is None:
        return x_tokens
    if cls_idx == 0:
        return torch.cat((cls_token, x_tokens), dim=1)
    if cls_idx >= x_tokens.shape[1]:
        return torch.cat((x_tokens, cls_token), dim=1)
    return torch.cat((x_tokens[:, :cls_idx, :], cls_token, x_tokens[:, cls_idx:, :]), dim=1)


def group_boundaries(t_steps: int, num_groups: int) -> List[Tuple[int, int]]:
    num_groups = max(1, min(num_groups, t_steps))
    base = t_steps // num_groups
    rem = t_steps % num_groups
    bounds: List[Tuple[int, int]] = []
    start = 0
    for g in range(num_groups):
        size = base + (1 if g < rem else 0)
        end = start + size
        if end > start:
            bounds.append((start, end))
        start = end
    return bounds


def reshape_to_spatiotemporal(
    x_tokens: torch.Tensor, t_steps: int
) -> Tuple[torch.Tensor, int]:
    bsz, l_tokens, dim = x_tokens.shape
    t_steps = max(1, t_steps)
    if l_tokens % t_steps != 0:
        t_steps = 1
    s_steps = l_tokens // t_steps
    return x_tokens.view(bsz, t_steps, s_steps, dim), s_steps


def branch_delta(
    x_branch: torch.Tensor,
    conv1d: nn.Conv1d,
    x_proj: nn.Linear,
    dt_proj: nn.Linear,
) -> torch.Tensor:
    x_conv_in = x_branch.transpose(1, 2)
    x_conv = F.conv1d(
        x_conv_in,
        weight=conv1d.weight,
        bias=conv1d.bias,
        padding=conv1d.padding[0],
        groups=conv1d.groups,
    )
    x_conv = x_conv[..., : x_branch.shape[1]]
    x_act = F.silu(x_conv).transpose(1, 2)
    x_dbl = F.linear(x_act, x_proj.weight, x_proj.bias)
    dt_lowrank = x_dbl[..., : dt_proj.in_features]
    return F.softplus(F.linear(dt_lowrank, dt_proj.weight, dt_proj.bias))


def estimate_delta_tensor(mixer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    xz = F.linear(hidden_states, mixer.in_proj.weight, mixer.in_proj.bias)
    x, _ = xz.chunk(2, dim=-1)

    delta_f = branch_delta(x, mixer.conv1d, mixer.x_proj, mixer.dt_proj)
    if not getattr(mixer, "bimamba", False):
        return delta_f

    if not all(hasattr(mixer, n) for n in ("conv1d_b", "x_proj_b", "dt_proj_b")):
        return delta_f

    delta_b = branch_delta(
        torch.flip(x, dims=[1]), mixer.conv1d_b, mixer.x_proj_b, mixer.dt_proj_b
    )
    delta_b = torch.flip(delta_b, dims=[1])
    return 0.5 * (delta_f + delta_b)


def normalize(value: float, vmin: float, vmax: float, eps: float) -> float:
    if abs(vmax - vmin) < eps:
        return 0.5
    return float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))


def score_metrics(metrics: Dict[str, float], block_stat: Dict[str, Any], cfg: PTQConfig) -> float:
    ranges = block_stat["ranges"]
    r_n = normalize(metrics["R"], ranges["R"][0], ranges["R"][1], cfg.eps)
    spa_n = normalize(metrics["E_spa"], ranges["E_spa"][0], ranges["E_spa"][1], cfg.eps)
    temp_n = normalize(metrics["E_temp"], ranges["E_temp"][0], ranges["E_temp"][1], cfg.eps)
    return cfg.alpha * r_n + cfg.beta * (1.0 - spa_n) + cfg.gamma * (1.0 - temp_n)


def group_metrics_for_layer(
    hidden_states: torch.Tensor,
    mixer: nn.Module,
    t_steps: int,
    cfg: PTQConfig,
) -> List[Dict[str, float]]:
    hs = hidden_states.detach().float()

    cls_token, x_tokens, _ = split_cls_and_tokens(hs, cfg.cls_token_position)
    del cls_token

    delta_seq = estimate_delta_tensor(mixer, hs)
    _, delta_tokens, _ = split_cls_and_tokens(delta_seq, cfg.cls_token_position)

    x_st, _ = reshape_to_spatiotemporal(x_tokens, t_steps)
    d_st, _ = reshape_to_spatiotemporal(delta_tokens, t_steps)

    bounds = group_boundaries(x_st.shape[1], cfg.num_groups)
    out: List[Dict[str, float]] = []
    for gidx, (start, end) in enumerate(bounds):
        xg = x_st[:, start:end, :, :]
        dg = d_st[:, start:end, :, :]

        x_anchor = xg.mean(dim=(1, 2), keepdim=True)
        rg = (xg - x_anchor).pow(2).sum(dim=-1).sqrt().mean().item()

        p_spa = torch.softmax(dg, dim=2)
        e_spa = (-(p_spa * (p_spa + cfg.eps).log2()).sum(dim=2)).mean().item()

        p_temp = torch.softmax(dg, dim=1)
        e_temp = (-(p_temp * (p_temp + cfg.eps).log2()).sum(dim=1)).mean().item()

        out.append({"group": float(gidx), "R": rg, "E_spa": e_spa, "E_temp": e_temp})

    return out


def collect_layer_group_metrics(
    model: nn.Module,
    data_loader: Iterable[Any],
    device: torch.device,
    max_batches: int,
    cfg: PTQConfig,
) -> Dict[int, List[Dict[str, float]]]:
    layer_metrics: Dict[int, List[Dict[str, float]]] = {}
    hook_handles: List[Any] = []

    def _make_hook(layer_idx: int, mixer: nn.Module):
        def _hook(module: nn.Module, inputs: Tuple[Any, ...]) -> None:
            if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
                return
            hs = inputs[0]
            cls_idx = resolve_cls_index(hs.shape[1], cfg.cls_token_position)
            tokens = hs.shape[1] - (1 if cls_idx is not None else 0)
            t_steps = infer_temporal_steps(model, tokens, cfg)
            stats = group_metrics_for_layer(hs, mixer, t_steps, cfg)
            layer_metrics.setdefault(layer_idx, []).extend(stats)

        return _hook

    model.eval()
    with torch.no_grad():
        for idx, mixer in iter_mamba_mixers(model):
            hook_handles.append(mixer.register_forward_pre_hook(_make_hook(idx, mixer)))

        for bidx, batch in enumerate(data_loader):
            if bidx >= max_batches:
                break
            video = extract_video_tensor(batch).to(device, non_blocking=True)
            _ = model(video)

    for h in hook_handles:
        h.remove()

    return layer_metrics
