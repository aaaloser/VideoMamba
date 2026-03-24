#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _resolve_fp_state(fp_obj: Any) -> tuple[Optional[Dict[str, torch.Tensor]], str]:
    if not isinstance(fp_obj, dict):
        return None, f"unexpected_type:{type(fp_obj)}"

    for key in ("model", "module", "state_dict"):
        if key in fp_obj and isinstance(fp_obj[key], dict):
            return fp_obj[key], key

    if any(torch.is_tensor(v) for v in fp_obj.values()):
        return fp_obj, "<root_dict>"

    return None, "unresolved"


def _f(x: float) -> str:
    return f"{x:.6e}"


def build_report(quant_ckpt: Dict[str, Any], fp_ckpt: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    out["quant_top_keys"] = sorted(list(quant_ckpt.keys()))
    out["quant_format"] = quant_ckpt.get("format")

    quantized_tensors: Dict[str, Dict[str, Any]] = quant_ckpt.get("quantized_tensors", {})
    non_quantized_state: Dict[str, torch.Tensor] = quant_ckpt.get("non_quantized_state", {})
    block_bits: Dict[str, int] = quant_ckpt.get("block_bits", {})

    out["quant_num_quantized_tensors"] = len(quantized_tensors)
    out["quant_num_non_quantized_tensors"] = len(non_quantized_state)
    out["quant_num_blocks"] = len(block_bits)

    if quantized_tensors:
        bit_counter = Counter(int(v.get("bits", -1)) for v in quantized_tensors.values())
        q_numel = sum(v["q"].numel() for v in quantized_tensors.values())
        q_code_min = min(int(v["q"].min().item()) for v in quantized_tensors.values())
        q_code_max = max(int(v["q"].max().item()) for v in quantized_tensors.values())
        s_min = min(float(v["scale"].min().item()) for v in quantized_tensors.values())
        s_max = max(float(v["scale"].max().item()) for v in quantized_tensors.values())

        out["quant_bit_distribution"] = dict(sorted(bit_counter.items()))
        out["quant_numel"] = int(q_numel)
        out["quant_code_range"] = [q_code_min, q_code_max]
        out["quant_scale_range"] = [s_min, s_max]

        per_bit = defaultdict(lambda: {
            "tensors": 0,
            "q_code_min": 10**9,
            "q_code_max": -(10**9),
            "scale_min": float("inf"),
            "scale_max": 0.0,
        })
        for _, v in quantized_tensors.items():
            bit = int(v.get("bits", -1))
            q = v["q"]
            s = v["scale"]
            d = per_bit[bit]
            d["tensors"] += 1
            d["q_code_min"] = min(d["q_code_min"], int(q.min().item()))
            d["q_code_max"] = max(d["q_code_max"], int(q.max().item()))
            d["scale_min"] = min(d["scale_min"], float(s.min().item()))
            d["scale_max"] = max(d["scale_max"], float(s.max().item()))
        out["quant_per_bit_stats"] = {str(k): v for k, v in sorted(per_bit.items(), key=lambda x: x[0])}
    else:
        out["quant_bit_distribution"] = {}
        out["quant_numel"] = 0

    ns_numel = sum(t.numel() for t in non_quantized_state.values() if torch.is_tensor(t))
    out["non_quant_numel"] = int(ns_numel)
    out["total_numel_from_quant_ckpt"] = int(out["quant_numel"] + ns_numel)

    fp_state, fp_state_key = _resolve_fp_state(fp_ckpt)
    out["fp_state_key"] = fp_state_key

    if fp_state is not None:
        fp_num_tensors = len(fp_state)
        fp_numel = sum(v.numel() for v in fp_state.values() if torch.is_tensor(v))
        fp_min = min(float(v.min().item()) for v in fp_state.values() if torch.is_tensor(v))
        fp_max = max(float(v.max().item()) for v in fp_state.values() if torch.is_tensor(v))

        out["fp_num_tensors"] = int(fp_num_tensors)
        out["fp_numel"] = int(fp_numel)
        out["fp_value_range"] = [fp_min, fp_max]

        q_keys = set(quantized_tensors.keys())
        fp_keys = set(fp_state.keys())
        overlap = q_keys & fp_keys

        out["key_overlap_count"] = len(overlap)
        out["q_only_examples"] = sorted(list(q_keys - fp_keys))[:10]
        out["fp_only_examples"] = sorted(list(fp_keys - q_keys))[:10]
    else:
        out["fp_num_tensors"] = -1
        out["fp_numel"] = -1
        out["fp_value_range"] = []
        out["key_overlap_count"] = -1
        out["q_only_examples"] = []
        out["fp_only_examples"] = []

    return out


def render_report_text(report: Dict[str, Any], quant_path: Path, fp_path: Path) -> str:
    lines: list[str] = []
    lines.append("=== Checkpoint Compare Report ===")
    lines.append(f"quant_ckpt: {quant_path}")
    lines.append(f"fp_ckpt   : {fp_path}")
    lines.append("")

    lines.append("[Quantized Checkpoint]")
    lines.append(f"- keys: {report['quant_top_keys']}")
    lines.append(f"- format: {report['quant_format']}")
    lines.append(f"- quantized_tensors: {report['quant_num_quantized_tensors']}")
    lines.append(f"- non_quantized_tensors: {report['quant_num_non_quantized_tensors']}")
    lines.append(f"- blocks: {report['quant_num_blocks']}")
    lines.append(f"- bit_distribution: {report['quant_bit_distribution']}")
    if report.get("quant_code_range"):
        lines.append(f"- q_code_range: {report['quant_code_range']}")
        lines.append(
            f"- q_scale_range: [{_f(report['quant_scale_range'][0])}, {_f(report['quant_scale_range'][1])}]"
        )
    lines.append(f"- quant_numel: {report['quant_numel']}")
    lines.append(f"- non_quant_numel: {report['non_quant_numel']}")
    lines.append(f"- total_numel(quant_ckpt): {report['total_numel_from_quant_ckpt']}")
    lines.append("")

    lines.append("[Per-Bit Stats]")
    per_bit = report.get("quant_per_bit_stats", {})
    if per_bit:
        for bit, stat in per_bit.items():
            lines.append(
                f"- bit={bit}: tensors={stat['tensors']}, q_range=[{stat['q_code_min']},{stat['q_code_max']}], "
                f"scale_range=[{_f(stat['scale_min'])},{_f(stat['scale_max'])}]"
            )
    else:
        lines.append("- <empty>")
    lines.append("")

    lines.append("[Full-Precision Checkpoint]")
    lines.append(f"- state_key: {report['fp_state_key']}")
    lines.append(f"- num_tensors: {report['fp_num_tensors']}")
    lines.append(f"- numel: {report['fp_numel']}")
    if report.get("fp_value_range"):
        lines.append(
            f"- value_range: [{_f(report['fp_value_range'][0])}, {_f(report['fp_value_range'][1])}]"
        )
    lines.append("")

    lines.append("[Key Alignment]")
    lines.append(f"- overlap_count: {report['key_overlap_count']}")
    lines.append(f"- q_only_examples: {report['q_only_examples']}")
    lines.append(f"- fp_only_examples: {report['fp_only_examples']}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot compare quantized and fp checkpoints.")
    parser.add_argument(
        "--quant_path",
        type=str,
        default="/data/liyifan24/VideoMamba/output_pth/videomamba_middle_mask_eval_f16_res224_ptq_h8_l4.pth",
    )
    parser.add_argument(
        "--fp_path",
        type=str,
        default="/data/liyifan24/VideoMamba/pretrain_model/videomamba_m16_k400_mask_ft_f16_res224.pth",
    )
    parser.add_argument("--save_json", type=str, default="")
    args = parser.parse_args()

    quant_path = Path(args.quant_path)
    fp_path = Path(args.fp_path)

    quant_ckpt = torch.load(str(quant_path), map_location="cpu")
    fp_ckpt = torch.load(str(fp_path), map_location="cpu")

    report = build_report(quant_ckpt, fp_ckpt)
    text = render_report_text(report, quant_path, fp_path)
    print(text)

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[Saved] {save_path}")


if __name__ == "__main__":
    main()
