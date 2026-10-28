#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ptq_eval.py —— 训练后量化（PTQ）评估。

对应论文 III 节：把训练好的卷积模式 S4D 转到递推模式，直接对离散参数做伪量化，
观察不同位宽 / 粒度 / 对称性 / 分段策略下的精度下降。

每种量化策略都从 **同一份 FP 权重** 重新构建 + 校准，结果互不影响。

用法：
    # 论文异构方案（W4A6, Ā8, x8）
    python scripts/quant/ptq_eval.py --ckpt-dir <实验目录> --preset paper

    # 统一位宽扫描
    python scripts/quant/ptq_eval.py --ckpt-dir <实验目录> --sweep 16,8,6,4

    # 按序列位置的分段位宽：前 800 步 8-bit，其余 4-bit
    python scripts/quant/ptq_eval.py --ckpt-dir <实验目录> \
        --segment-bits "0:800:8,800::4"

    # 从策略文件读取（推荐：策略可复用、可记录）
    python scripts/quant/ptq_eval.py --ckpt-dir <实验目录> \
        --quant-config configs/quant/segment_8_4.yaml
"""

import argparse
import json
import os
import sys
import time

import torch

# 仓库根目录（scripts/quant/ -> repo）
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quant import QuantConfig, paper_config, uniform_config          # noqa: E402
from quant.calibrate import calibrate_model                         # noqa: E402
from quant.cli import add_quant_args, apply_overrides, build_config  # noqa: E402
from quant.data import (                                            # noqa: E402
    infer_sequence_length,
    load_experiment_config,
    make_loader,
)
from quant.evaluate import evaluate, evaluate_quantization_off      # noqa: E402
from quant.memory import estimate_memory, format_memory             # noqa: E402
from quant.model import build_quant_model                           # noqa: E402


#############################
# 参数
#############################
def parse_args():
    parser = argparse.ArgumentParser(description="S4D 训练后量化（PTQ）评估")

    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="训练实验目录（含 best.pth / config.yaml）")
    parser.add_argument("--ckpt-name", type=str, default="best.pth", help="权重文件名")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="覆盖 config.yaml 里的 data_dir")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--max-samples", type=int, default=256,
                        help="评估子集大小（0 表示全量）")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    add_quant_args(parser)
    parser.add_argument("--sweep", type=str, default=None,
                        help="uniform 位宽扫描，例如 16,8,6,4")

    # 校准
    parser.add_argument("--calib-batches", type=int, default=4)
    parser.add_argument("--calib-split", type=str, default="valid")

    parser.add_argument("--out", type=str, default=None, help="结果 JSON 路径")
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


#############################
# 构造量化配置
#############################
def build_configs(args):
    """返回 [(标签, QuantConfig), ...]。"""
    if args.sweep:
        bits_list = [int(b) for b in args.sweep.split(",") if b.strip()]
        configs = []
        for bits in bits_list:
            cfg = uniform_config(bits, param_mode=args.param_mode or "continuous")
            configs.append((f"uniform-A{bits}", apply_overrides(cfg, args)))
        return configs

    return [build_config(args, default_param_mode="continuous")]


#############################
# 主流程
#############################
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    # ---- 读实验配置与权重 ----
    train_cfg = load_experiment_config(args.ckpt_dir)
    seq_len = infer_sequence_length(train_cfg)
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    fp_state_dict = ckpt["model"] if "model" in ckpt else ckpt

    configs = build_configs(args)

    # ---- DataLoader（评估 + 校准）----
    print(f"==> 数据集: {train_cfg.dataset} | split={args.split} | max_samples={args.max_samples}")
    test_loader = make_loader(
        train_cfg, split=args.split, data_dir=args.data_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_samples=(args.max_samples or None), seed=args.seed,
    )
    calib_loader = make_loader(
        train_cfg, split=args.calib_split, data_dir=args.data_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_samples=args.calib_batches * args.batch_size, seed=args.seed + 1,
    )

    results = {
        "experiment": os.path.abspath(args.ckpt_dir),
        "dataset": train_cfg.dataset,
        "split": args.split,
        "max_samples": args.max_samples,
        "device": device,
        "runs": [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ---- FP32 递推基线 ----
    # 同一套递推实现、关闭伪量化、保留原始 GELU，用于确认"卷积 → 递推"本身不掉点。
    fp_model = build_quant_model(
        fp_state_dict, qcfg=QuantConfig(use_qgelu=False), verbose=False
    ).to(device)
    fp_stats = evaluate_quantization_off(fp_model, test_loader, device=device)
    fp_mem = estimate_memory(fp_model, QuantConfig(), length=seq_len)
    results["fp32_recurrent"] = {"eval": fp_stats, "memory": fp_mem}

    print("\n" + "=" * 96)
    print(f"FP32 递推基线: acc={fp_stats['acc']:.2f}% | loss={fp_stats['loss']:.4f} | "
          f"{fp_stats['latency_ms']:.2f} ms/sample")
    print("=" * 96)

    # ---- 逐个量化策略：每次都从 FP 权重重新构建 ----
    for label, qcfg in configs:
        t0 = time.time()

        model = build_quant_model(fp_state_dict, qcfg=qcfg, verbose=False).to(device)
        calibrate_model(model, calib_loader, device=device,
                        max_batches=args.calib_batches, verbose=False)

        stats = evaluate(model, test_loader, device=device)
        mem = estimate_memory(model, qcfg, length=seq_len)

        run = {
            "label": label,
            "quant_config": qcfg.to_dict(),
            "describe": qcfg.describe(),
            "eval": stats,
            "memory": mem,
            "elapsed_s": time.time() - t0,
        }
        results["runs"].append(run)

        print(f"\n[{label}] {qcfg.describe()}")
        print(f"  acc={stats['acc']:.2f}% | macro-F1={stats['macro_f1']:.2f}% | "
              f"loss={stats['loss']:.4f} | latency={stats['latency_ms']:.2f} ms/sample "
              f"| n={stats['n_samples']}")
        print(f"  {format_memory(mem)}")

    # ---- 保存 ----
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[Saved] {args.out}")

    return results


if __name__ == "__main__":
    main()
