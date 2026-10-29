#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qat_train.py —— 量化感知训练（QAT）。

对应论文 V 节：
    1. 先用 PTQ 量化好的模型做起点（每次都从 FP 权重重新构建/校准）；
    2. 切到递推模式，用 STE 继续训练若干 epoch（论文 10 epoch、lr 1e-4）；
    3. 三种参数化：continuous / discrete / frozen；
    4. 训练后重新评估，并保存量化 checkpoint。

不同量化策略（含按序列位置的分段位宽）各自独立跑一次 PTQ + QAT，
通过 `--quant-config` 或 `--segment-bits` 指定，产物目录用 `--tag` 区分。

用法：
    python scripts/quant/qat_train.py --ckpt-dir <实验目录> --param-mode discrete --epochs 2

    # 分段策略：前 800 token 8-bit，其余 4-bit
    python scripts/quant/qat_train.py --ckpt-dir <实验目录> \
        --quant-config configs/quant/segment_8_4.yaml --tag seg84 --epochs 2

    # 快速冒烟
    python scripts/quant/qat_train.py --ckpt-dir <实验目录> --epochs 1 \
        --max-train-batches 1 --max-eval-samples 32 --batch-size 4
"""

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quant.calibrate import calibrate_model                       # noqa: E402
from quant.cli import add_quant_args, build_config                # noqa: E402
from quant.data import (                                          # noqa: E402
    infer_sequence_length,
    load_experiment_config,
    make_loader,
    make_stratified_loader,
)
from quant.evaluate import evaluate                               # noqa: E402
from quant.memory import estimate_memory, format_memory           # noqa: E402
from quant.model import build_quant_model, save_quant_checkpoint  # noqa: E402
from quant.qat import train_qat                                   # noqa: E402


#############################
# 参数
#############################
def parse_args():
    parser = argparse.ArgumentParser(description="S4D 量化感知训练（QAT）")

    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--ckpt-name", type=str, default="best.pth")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="输出目录（默认 <ckpt-dir>/quant_qat_<tag>）")
    parser.add_argument("--tag", type=str, default=None,
                        help="策略标签，用于区分不同量化配置的产物")
    parser.add_argument("--device", type=str, default="cuda")

    add_quant_args(parser)

    # 训练
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1000.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="评估用 batch（默认与 --batch-size 相同；评估无反向，可放大加速）")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="每个 epoch 最多训练多少个 batch（冒烟用）")
    parser.add_argument("--max-eval-samples", type=int, default=256,
                        help="评估子集大小（0 表示全量）")
    parser.add_argument("--eval-every-epoch", action="store_true",
                        help="每个 epoch 结束后在全量测试集上评估一次（记录恢复曲线）")
    parser.add_argument("--calib-batches", type=int, default=4)
    parser.add_argument("--train-split", type=str, default="train",
                        choices=["train", "valid"],
                        help="冒烟测试时可用 valid 避免扫描整个训练集")
    parser.add_argument("--train-fraction", type=float, default=None,
                        help="按类别分层抽样训练集的比例（如 0.05 / 0.1）；None 表示全量")

    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


#############################
# 主流程
#############################
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    # ---- 量化策略 ----
    label, qcfg = build_config(args, default_param_mode="discrete")
    qcfg.grad_clip = args.grad_clip

    tag = args.tag or qcfg.param_mode
    out_dir = args.out_dir or os.path.join(args.ckpt_dir, f"quant_qat_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 实验配置与 FP 权重 ----
    train_cfg = load_experiment_config(args.ckpt_dir)
    seq_len = infer_sequence_length(train_cfg)
    ckpt = torch.load(os.path.join(args.ckpt_dir, args.ckpt_name), map_location="cpu")
    fp_state_dict = ckpt["model"] if "model" in ckpt else ckpt

    print(f"==> 实验: {args.ckpt_dir}")
    print(f"==> 策略: {label} | {qcfg.describe()}")
    print(f"==> 输出: {out_dir}")

    # ---- 数据 ----
    eval_batch_size = args.eval_batch_size or args.batch_size

    eval_loader = make_loader(
        train_cfg, split="test", data_dir=args.data_dir,
        batch_size=eval_batch_size, num_workers=args.num_workers,
        max_samples=(args.max_eval_samples or None), seed=args.seed,
    )
    calib_loader = make_loader(
        train_cfg, split="valid", data_dir=args.data_dir,
        batch_size=eval_batch_size, num_workers=args.num_workers,
        max_samples=args.calib_batches * eval_batch_size, seed=args.seed + 1,
    )
    # 训练中监控用 valid（避免用测试集做逐 epoch 观测）；最终指标仍在 test 上算
    monitor_loader = None
    if args.eval_every_epoch:
        monitor_loader = make_loader(
            train_cfg, split="valid", data_dir=args.data_dir,
            batch_size=eval_batch_size, num_workers=args.num_workers,
            max_samples=(args.max_eval_samples or None), seed=args.seed,
        )
    train_loader_kwargs = dict(
        cfg=train_cfg, split=args.train_split, data_dir=args.data_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        seed=args.seed,
    )

    if args.train_fraction:
        train_loader, subset_stats = make_stratified_loader(
            fraction=args.train_fraction, **train_loader_kwargs
        )
        print(f"==> 训练子集: fraction={args.train_fraction:.3f} | "
              f"n={subset_stats['total']} | per_class={subset_stats['per_class']}")
    else:
        train_loader = make_loader(shuffle=True, **train_loader_kwargs)
        subset_stats = {"fraction": 1.0, "total": None, "per_class": None}

    # ---- 构建 + 校准（PTQ 起点）----
    model = build_quant_model(fp_state_dict, qcfg=qcfg).to(device)
    calibrate_model(model, calib_loader, device=device, max_batches=args.calib_batches)

    before = evaluate(model, eval_loader, device=device)
    print(f"\n[PTQ] acc={before['acc']:.2f}% | loss={before['loss']:.4f}")

    # ---- QAT ----
    t0 = time.time()
    history = train_qat(
        model,
        train_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=args.grad_clip,
        max_batches_per_epoch=args.max_train_batches,
        val_loader=monitor_loader,
        max_val_batches=None,
    )
    elapsed = time.time() - t0

    after = evaluate(model, eval_loader, device=device)
    mem = estimate_memory(model, qcfg, length=seq_len)

    print(f"\n[QAT] acc={after['acc']:.2f}% | loss={after['loss']:.4f} | "
          f"耗时 {elapsed:.1f}s")
    print(f"[mem] {format_memory(mem)}")

    # ---- 保存 ----
    save_quant_checkpoint(
        model,
        os.path.join(out_dir, "quant_best.pth"),
        meta={
            "source_experiment": os.path.abspath(args.ckpt_dir),
            "dataset": train_cfg.dataset,
            "strategy": label,
            "ptq_acc": before["acc"],
            "qat_acc": after["acc"],
        },
    )

    report = {
        "experiment": os.path.abspath(args.ckpt_dir),
        "dataset": train_cfg.dataset,
        "strategy": label,
        "quant_config": qcfg.to_dict(),
        "describe": qcfg.describe(),
        "ptq": before,
        "qat": after,
        "history": history,
        "memory": mem,
        "elapsed_s": elapsed,
        "train_subset": subset_stats,
        "args": vars(args),
    }
    with open(os.path.join(out_dir, "qat_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved] {out_dir}/quant_best.pth")
    print(f"[Saved] {out_dir}/qat_report.json")

    return report


if __name__ == "__main__":
    main()
