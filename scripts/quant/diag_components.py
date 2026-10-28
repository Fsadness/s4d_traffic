#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""diag_components.py —— PTQ 组件消融，用来定位掉点来源。

论文 Fig.2 的思路：单独量化某一个组件，其余保持高精度（16-bit）。
本脚本在此之上再加一档“只用 qGELU、不量化”，用于区分
“量化本身”和“激活函数替换”各自的贡献。

用法：
    python scripts/diag_components.py --ckpt-dir <实验目录> --max-samples 32
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quant import QuantConfig                              # noqa: E402
from quant.calibrate import calibrate_model                # noqa: E402
from quant.data import load_experiment_config, make_loader  # noqa: E402
from quant.evaluate import evaluate, evaluate_quantization_off  # noqa: E402
from quant.model import build_quant_model                  # noqa: E402


#############################
# 变体定义
#############################
def base16(**flags):
    """所有位宽 16-bit 的底配，再按需改单个组件。"""
    return QuantConfig(
        bits_a=16, bits_b=16, bits_c=16, bits_d=16,
        bits_state=16, bits_act=16, bits_other=16,
        **flags,
    )


def build_variants():
    """返回 [(名称, 说明, QuantConfig)]。"""
    off = dict(quantize_weights=False, quantize_state=False, quantize_activations=False)

    variants = []

    # 0) 纯 FP（原始 GELU、全部不量化）
    variants.append(("fp_gelu", "FP32 + GELU（基线）", base16(use_qgelu=False, **off)))

    # 1) 只换 qGELU
    variants.append(("qgelu_only", "FP32 + qGELU", base16(use_qgelu=True, **off)))

    # 2) 只量化非 SSM 权重（8-bit）
    cfg = base16(use_qgelu=True, quantize_state=False, quantize_activations=False)
    cfg.bits_other = 8
    variants.append(("weights8", "非 SSM 权重 8-bit", cfg))

    # 3) 只量化激活（8-bit）
    cfg = base16(use_qgelu=True, quantize_weights=False, quantize_state=False)
    cfg.bits_act = 8
    variants.append(("act8", "激活 8-bit", cfg))

    # 4) 只量化状态（8-bit）
    cfg = base16(use_qgelu=True, quantize_weights=False, quantize_activations=False)
    cfg.bits_state = 8
    variants.append(("state8", "状态 x 8-bit", cfg))

    # 5) 只量化 Ā（8-bit）
    cfg = base16(use_qgelu=True, **off)
    cfg.bits_a = 8
    variants.append(("A8", "Ā 8-bit", cfg))

    # 6) 只量化 B̄/C/D（4-bit）
    cfg = base16(use_qgelu=True, **off)
    cfg.bits_b = cfg.bits_c = cfg.bits_d = 4
    variants.append(("BCD4", "B̄/C/D 4-bit", cfg))

    # 7) 统一 16-bit 全量化
    variants.append(("uniform16", "全部 16-bit", base16(use_qgelu=True)))

    # 8) 统一 8-bit 全量化
    variants.append((
        "uniform8", "全部 8-bit",
        QuantConfig(
            bits_a=8, bits_b=8, bits_c=8, bits_d=8,
            bits_state=8, bits_act=8, bits_other=8, use_qgelu=True,
        ),
    ))

    return variants


#############################
# 主流程
#############################
def main():
    parser = argparse.ArgumentParser(description="PTQ 组件消融")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--calib-batches", type=int, default=1)
    parser.add_argument("--gelu", action="store_true",
                        help="所有变体保留原始 GELU（只观察量化影响）")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    exp_cfg = load_experiment_config(args.ckpt_dir)
    ckpt = torch.load(os.path.join(args.ckpt_dir, "best.pth"), map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt

    test_loader = make_loader(
        exp_cfg, split=args.split, data_dir=args.data_dir,
        batch_size=args.batch_size, num_workers=2,
        max_samples=args.max_samples, seed=0,
    )
    calib_loader = make_loader(
        exp_cfg, split="valid", data_dir=args.data_dir,
        batch_size=args.batch_size, num_workers=2,
        max_samples=args.max_samples, seed=1,
    )

    print("=" * 92)
    print(f"{'变体':<12} {'acc':>8} {'loss':>8}   说明")
    print("=" * 92)

    for name, desc, qcfg in build_variants():
        # 可选：整组变体都保留原始 GELU，用于把“qGELU 影响”从量化影响里剥离
        if args.gelu:
            qcfg.use_qgelu = False
            desc = desc + "（保留 GELU）"

        model = build_quant_model(sd, qcfg=qcfg, verbose=False).to(device)

        needs_calib = qcfg.quantize_state or qcfg.quantize_activations
        if needs_calib:
            calibrate_model(model, calib_loader, device=device,
                            max_batches=args.calib_batches, verbose=False)

        if name == "fp_gelu":
            stats = evaluate_quantization_off(model, test_loader, device=device)
        else:
            stats = evaluate(model, test_loader, device=device)

        print(f"{name:<12} {stats['acc']:>7.2f}% {stats['loss']:>8.4f}   {desc}")

    print("=" * 92)


if __name__ == "__main__":
    main()
