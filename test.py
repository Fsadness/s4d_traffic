"""
test.py — 实验模型评估脚本
用法:
    python test.py <experiment_dir>                     # 整段前向（conv 核，默认）
    python test.py <experiment_dir> --kernel recurrent  # 整段前向（recurrent 核）
    python test.py <experiment_dir> --mode stream       # 逐步递推（可接早退策略）

实验文件夹需包含: best.pth, config.yaml, train.log, checkpoints/

评估指标:
  - 单次推理平均时间 (Average Latency, ms/sample)
  - 总体 & 每类别: Accuracy, Precision, Recall, F1
  - 归一化混淆矩阵图
  - 平均置信度, 置信度裕度, 交叉熵损失
  - stream 模式额外记录：实际步数、早退比例、所用策略

输出:
  - <exp_dir>/evaluation/            (full + conv)
  - <exp_dir>/evaluation_recurrent/  (full + recurrent)
  - <exp_dir>/evaluation_stream/     (stream)
"""

import os
import json
import time
import argparse
import importlib

import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

from tqdm.auto import tqdm

# 项目内部模块 — 数据、指标与 checkpoint 加载均与其他入口共用
from datasets.dataloader import build_loaders
from datasets.specs import infer_d_input
from engine.checkpoint import load_fp_checkpoint
from engine.metrics import (
    compute_metrics,
    normalized_confusion,
    plot_confusion_matrix,
    write_classification_report,
)
from policies import build_exit_policy


###############################################################################
# 参数解析
###############################################################################

parser = argparse.ArgumentParser(description="Evaluate a trained S4D model")
parser.add_argument("exp_dir", type=str, help="实验文件夹路径 (含 best.pth / config.yaml)")
parser.add_argument("--mode", type=str, default="full", choices=["full", "stream"],
                    help="full: 整段前向；stream: 逐步递推（固定 recurrent 核，可接早退策略）")
parser.add_argument("--batch_size", type=int, default=None, help="覆盖 config 中的 batch_size")
parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
parser.add_argument("--device", type=str, default=None, help="cuda / cpu, 默认自动检测")
parser.add_argument("--kernel", type=str, default="conv", choices=["conv", "recurrent"],
                    help="full 模式下使用的核（stream 模式固定 recurrent）")
parser.add_argument("--max_samples", type=int, default=None,
                    help="每个 split 随机抽样至多 N 个样本（固定 seed=0，便于复现；None 表示全量）")
args = parser.parse_args()

exp_dir = os.path.abspath(args.exp_dir)
assert os.path.isdir(exp_dir), f"实验文件夹不存在: {exp_dir}"

best_pth = os.path.join(exp_dir, "best.pth")
config_path = os.path.join(exp_dir, "config.yaml")
assert os.path.isfile(best_pth), f"best.pth 不存在: {best_pth}"
assert os.path.isfile(config_path), f"config.yaml 不存在: {config_path}"

if args.mode == "stream":
    eval_dir = os.path.join(exp_dir, "evaluation_stream")
elif args.kernel == "recurrent":
    eval_dir = os.path.join(exp_dir, "evaluation_recurrent")
else:
    eval_dir = os.path.join(exp_dir, "evaluation")
os.makedirs(eval_dir, exist_ok=True)


###############################################################################
# 加载 config
###############################################################################

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 使用 argparse namespace 兼容 build_loaders / build_transform 的 args 访问方式
cfg = argparse.Namespace(**config)

# 覆盖项
if args.batch_size is not None:
    cfg.batch_size = args.batch_size
cfg.num_workers = args.num_workers

# 模式强制为 finetune（test 只在 finetune 模式下有意义）
cfg.mode = "finetune"

# 兼容旧版 config: 如果没有 tokenizer 字段，则默认使用 pixel
if not hasattr(cfg, 'tokenizer'):
    cfg.tokenizer = "pixel"

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
# 构建数据集（仅需 testloader，由共用加载器完成）
###############################################################################

print(f"==> Loading dataset: {cfg.dataset} ...")
test_bundle = build_loaders(cfg, splits=("test",), max_samples=args.max_samples)["test"]
testloader = test_bundle.loader
num_classes = test_bundle.num_classes
class_names = test_bundle.class_names
print(f"    Classes: {num_classes} | Test samples: {len(testloader.dataset)}")


###############################################################################
# 计算 d_input（与训练共用同一份推导）
###############################################################################

d_input = infer_d_input(cfg)


###############################################################################
# 构建模型
###############################################################################

# stream 模式固定使用合并后的分类网络（streaming=True，循环核）；
# full 模式沿用 config 里的模型，核由 --kernel 决定。
if args.mode == "stream":
    model_name = "s4d_classification"
    kernel_type = "recurrent"
else:
    model_name = cfg.model
    kernel_type = args.kernel

print(f"==> Building model: {model_name} (mode={args.mode}, kernel={kernel_type}) ...")
model_module = importlib.import_module(f"models.networks.{model_name}")
ModelClass = model_module.Model

model_kwargs = dict(
    d_input=d_input,
    d_output=num_classes,
    d_model=cfg.d_model,
    n_layers=cfg.n_layers,
    dropout=cfg.dropout,
    prenorm=cfg.prenorm,
    lr=cfg.lr,
    kernel_type=kernel_type,
)

policy = None
if args.mode == "stream":
    # 当前阶段策略为空实现：始终返回 False，即"不早退"
    policy = build_exit_policy("none")
    model_kwargs.update(streaming=True, early_stop_strategy=policy, debug=True)

model = ModelClass(**model_kwargs)
print(f"    d_input: {d_input} | d_output: {num_classes}")
if policy is not None:
    print(f"    exit policy: {policy.describe()}")

# 加载权重：load -> eval -> prepare_for_inference（顺序由 engine.checkpoint 保证）
print(f"==> Loading weights: {best_pth}")
model, checkpoint = load_fp_checkpoint(model, best_pth, device=device, strict=True)

if device == "cuda":
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()


###############################################################################
# 推理 & 收集结果
###############################################################################

all_targets = []
all_preds = []
all_probs = []
all_losses = []
all_margins = []
latencies = []
stream_steps = []

print("==> Running inference on test set ...")
with torch.no_grad():
    for inputs, targets in tqdm(testloader, desc="Testing"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # 计时
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if args.mode == "stream":
            outputs, info = model(inputs)
            if info and "stop_step" in info:
                stream_steps.extend(info["stop_step"].detach().cpu().tolist())
        else:
            outputs = model(inputs)

        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        batch_size = inputs.size(0)
        per_sample_latency = (t1 - t0) / batch_size  # 秒/sample
        latencies.extend([per_sample_latency] * batch_size)

        # 概率 & 预测
        probs = torch.softmax(outputs, dim=1)
        max_probs, preds = probs.max(dim=1)

        # 置信度裕度: top1 - top2（在这里顺带算，避免再跑一遍数据集）
        sorted_probs, _ = probs.sort(dim=1, descending=True)
        margins = (sorted_probs[:, 0] - sorted_probs[:, 1]).cpu().numpy()

        # 逐样本交叉熵损失
        batch_losses = nn.functional.cross_entropy(outputs, targets, reduction="none")

        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(max_probs.cpu().numpy().tolist())
        all_losses.extend(batch_losses.cpu().numpy().tolist())
        all_margins.extend(margins.tolist())

all_targets = np.array(all_targets)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_losses = np.array(all_losses)
all_margins = np.array(all_margins)
latencies = np.array(latencies)


###############################################################################
# 计算指标
###############################################################################

print("==> Computing metrics ...")

# 仅 full 模式保留原有的 profiler 输出（stream 模式逐时间步，profiler 代价过高）
if args.mode == "full":
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:

        outputs = model(inputs)

    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=20
    ))

avg_margin = float(all_margins.mean())

result = compute_metrics(
    all_targets,
    all_preds,
    all_probs,
    all_losses,
    latencies,
    class_names,
    avg_margin,
)
overall = result["overall"]
per_class = result["per_class"]


###############################################################################
# 流式统计（早退比例 / 实际步数）
###############################################################################

streaming = None
if args.mode == "stream" and stream_steps:
    steps = np.asarray(stream_steps, dtype=np.int64) + 1
    total_steps = int(getattr(model, "_total_steps", 0) or steps.max())
    streaming = {
        "policy": policy.describe(),
        "total_steps": total_steps,
        "avg_steps": round(float(steps.mean()), 2),
        "early_exit_ratio": round(float((steps < total_steps).mean()), 4),
    }


###############################################################################
# 打印摘要
###############################################################################

print("\n" + "=" * 60)
print(f"  Experiment : {exp_dir}")
print(f"  Dataset    : {cfg.dataset}")
print(f"  Model      : {model_name} ({args.mode})")
print(f"  Device     : {device}")
print("=" * 60)
print(f"  Accuracy           : {overall['accuracy']:.3f}%")
print(f"  Macro Precision    : {overall['macro_precision']:.3f}%")
print(f"  Macro Recall       : {overall['macro_recall']:.3f}%")
print(f"  Macro F1           : {overall['macro_f1']:.3f}%")
print(f"  Weighted F1        : {overall['weighted_f1']:.3f}%")
print(f"  Avg Loss (CE)      : {overall['avg_loss']:.4f}")
print(f"  Avg Confidence     : {overall['avg_confidence']:.4f}")
print(f"  Avg Margin         : {overall['avg_margin']:.4f}")
print(f"  Avg Latency        : {overall['avg_latency_ms']:.3f} ms/sample")
print(f"  Throughput         : {overall['throughput_samples_per_sec']:.1f} samples/sec")
if streaming is not None:
    print("-" * 60)
    print(f"  Exit Policy        : {streaming['policy']}")
    print(f"  Avg Steps          : {streaming['avg_steps']:.1f} / {streaming['total_steps']}")
    print(f"  Early Exit Ratio   : {streaming['early_exit_ratio'] * 100:.2f}%")
print("=" * 60)


###############################################################################
# 保存 metrics.json
###############################################################################

metrics = {
    "experiment_dir": exp_dir,
    "dataset": cfg.dataset,
    "model": model_name,
    "tokenizer": cfg.tokenizer,
    "checkpoint": "best.pth",
    "device": device,
    "mode": args.mode,
    "kernel": kernel_type,
    "test_samples": int(len(all_targets)),
    "num_classes": int(num_classes),
    "overall": overall,
    "per_class": per_class,
}
if streaming is not None:
    metrics["streaming"] = streaming

metrics_path = os.path.join(eval_dir, "metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"[Saved] {metrics_path}")


###############################################################################
# 保存 classification_report.csv
###############################################################################

csv_path = os.path.join(eval_dir, "classification_report.csv")
write_classification_report(csv_path, all_targets, all_preds, class_names)
print(f"[Saved] {csv_path}")


###############################################################################
# 保存 confusion_matrix.png
###############################################################################

_, cm_normalized = normalized_confusion(all_targets, all_preds, num_classes)
cm_path = os.path.join(eval_dir, "confusion_matrix.png")
plot_confusion_matrix(cm_normalized, class_names, cm_path)


###############################################################################
# 完成
###############################################################################

print(f"\n==> All results saved to: {eval_dir}")
print("Done.")
