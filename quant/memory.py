"""memory.py —— 量化后的内存占用估算。

论文 Table III 用“相对 FP32 的显存节省”衡量收益，这里按同样的口径计算：

    权重显存 = Σ (参数量 × 位宽 / 8)
    状态显存 = 层数 × H × N/2 × 2（实部+虚部）× 位宽 / 8   （单样本）

注意：
    - FP32 基线按 4 Byte/参数；
    - 偏置不量化（本实现里保留 FP32），占比 < 0.1%；
    - 激活的峰值显存与序列长度 L、batch 有关，这里只统计“常驻权重 + 状态”。
"""

import torch
import torch.nn as nn

from .config import QuantConfig


#############################
# 单个参数的位宽
#############################
def bits_for_param(name: str, cfg: QuantConfig) -> int:
    """按参数名判断分组，返回其量化位宽。"""
    # SSM 参数
    if ".kernel." in name:
        if name.endswith(("log_A_real", "A_imag", "A_bar_real", "A_bar_imag", "log_dt")):
            return cfg.bits_a
        if name.endswith(("B_bar_real", "B_bar_imag")):
            return cfg.bits_b
        if name.endswith(("C", "C_real", "C_imag")):
            return cfg.bits_c
    # 直通项 D
    if name.endswith(".D") or name == "D":
        return cfg.bits_d
    # 其他（encoder / decoder / norm / conv）
    return cfg.bits_other


def group_for_param(name: str) -> str:
    """给参数分组，便于结果表里分别统计。"""
    if ".kernel." in name:
        if name.endswith(("log_A_real", "A_imag", "A_bar_real", "A_bar_imag", "log_dt")):
            return "A"
        if name.endswith(("B_bar_real", "B_bar_imag")):
            return "B"
        if name.endswith(("C", "C_real", "C_imag")):
            return "C"
    if name.endswith(".D") or name == "D":
        return "D"
    if "norms" in name:
        return "norm"
    if "output_linear" in name:
        return "conv"
    return "other"


#############################
# 估算
#############################
def average_state_bits(cfg: QuantConfig, length: int = None) -> float:
    """考虑按序列位置分段后的平均状态位宽。

    未配置 `segment_bits` 时直接返回 bits_state；配置了但未给 length 时，
    用所有分段里最大的 end 作为序列长度（None 视作 0，即只有显式长度段参与）。
    """
    if not getattr(cfg, "segment_bits", None):
        return float(cfg.bits_state)

    if length is None:
        ends = [int(s["end"]) for s in cfg.segment_bits if s.get("end") is not None]
        length = max(ends) if ends else 0

    if length <= 0:
        return float(cfg.bits_state)

    total = 0.0
    covered = 0
    for start, end, _, bits_state in cfg.segments_for_length(length):
        total += (end - start) * bits_state
        covered += end - start

    return total / covered if covered else float(cfg.bits_state)


@torch.no_grad()
def estimate_memory(model: nn.Module, cfg: QuantConfig = None, length: int = None) -> dict:
    """估算权重与状态的内存占用（Byte）。

    `length` 用于计算分段位宽下的平均状态开销；不传则取分段的最大 end。
    """
    cfg = cfg or getattr(model, "cfg", QuantConfig())

    fp_bytes = 0
    quant_bytes = 0
    groups = {}

    for name, p in model.named_parameters():
        n = p.numel()
        bits = bits_for_param(name, cfg)
        g = group_for_param(name)

        entry = groups.setdefault(
            g, {"params": 0, "fp_bytes": 0, "quant_bytes": 0, "bits": bits}
        )
        entry["params"] += n
        entry["fp_bytes"] += n * 4
        entry["quant_bytes"] += n * bits / 8.0

        fp_bytes += n * 4
        quant_bytes += n * bits / 8.0

    # 状态缓存（单样本）：每层 H × N/2 个复数
    n_layers = len(model.backbone.s4_layers)
    H = model.backbone.s4_layers[0].h
    N_half = model.backbone.s4_layers[0].n // 2
    state_elems = n_layers * H * N_half * 2

    state_fp = state_elems * 4
    state_bits = average_state_bits(cfg, length)
    state_q = state_elems * state_bits / 8.0 if cfg.quantize_state else state_fp

    savings = 100.0 * (1.0 - quant_bytes / fp_bytes) if fp_bytes else 0.0

    return {
        "weights": {
            "fp32_bytes": int(fp_bytes),
            "quant_bytes": int(quant_bytes),
            "fp32_mib": fp_bytes / 2 ** 20,
            "quant_mib": quant_bytes / 2 ** 20,
            "savings_pct": savings,
            "compression": (fp_bytes / quant_bytes) if quant_bytes else float("inf"),
        },
        "state_per_sample": {
            "fp32_bytes": int(state_fp),
            "quant_bytes": int(state_q),
            "savings_pct": 100.0 * (1.0 - state_q / state_fp) if state_fp else 0.0,
        },
        "groups": groups,
        "bits": {
            "a": cfg.bits_a, "b": cfg.bits_b, "c": cfg.bits_c, "d": cfg.bits_d,
            "state": cfg.bits_state, "act": cfg.bits_act, "other": cfg.bits_other,
            "state_effective": state_bits,
            "segment_bits": cfg.segment_bits,
        },
    }


def format_memory(mem: dict) -> str:
    """把估算结果格式化成一行，便于打表。"""
    w = mem["weights"]
    return (
        f"weights {w['fp32_mib']:.2f} MiB → {w['quant_mib']:.2f} MiB "
        f"(-{w['savings_pct']:.2f}%, {w['compression']:.2f}x) | "
        f"state/sample {mem['state_per_sample']['fp32_bytes']}B → "
        f"{mem['state_per_sample']['quant_bytes']}B"
    )
