"""
test.py — 实验模型评估脚本
用法: python test.py <experiment_dir>
实验文件夹需包含: best.pth, config.yaml, train.log, checkpoints/

评估指标:
  - 单次推理平均时间 (Average Latency, ms/sample)
  - 总体 & 每类别: Accuracy, Precision, Recall, F1
  - 归一化混淆矩阵图
  - 平均置信度, 置信度裕度, 交叉熵损失

输出 (保存至 <experiment_dir>/evaluation/):
  - metrics.json
  - classification_report.csv
  - confusion_matrix.png
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

# 项目内部模块 — 数据与指标均与训练 / 量化评估共用
from datasets.dataloader import build_loaders
from datasets.specs import infer_d_input
from engine.metrics import (
    compute_metrics,
    normalized_confusion,
    plot_confusion_matrix,
    write_classification_report,
)


###############################################################################
# 参数解析
###############################################################################

parser = argparse.ArgumentParser(description="Evaluate a trained S4D model")
parser.add_argument("exp_dir", type=str, help="实验文件夹路径 (含 best.pth / config.yaml)")
parser.add_argument("--batch_size", type=int, default=None, help="覆盖 config 中的 batch_size")
parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
parser.add_argument("--device", type=str, default=None, help="cuda / cpu, 默认自动检测")
parser.add_argument("--kernel", type=str, default="conv", choices=["conv", "recurrent"], help="覆盖 config 中的 kernel_type")
parser.add_argument("--max_samples", type=int, default=None, help="只评估前 N 个测试样本（快速冒烟，None 表示全量）")
args = parser.parse_args()

exp_dir = os.path.abspath(args.exp_dir)
assert os.path.isdir(exp_dir), f"实验文件夹不存在: {exp_dir}"

best_pth = os.path.join(exp_dir, "best.pth")
config_path = os.path.join(exp_dir, "config.yaml")
assert os.path.isfile(best_pth), f"best.pth 不存在: {best_pth}"
assert os.path.isfile(config_path), f"config.yaml 不存在: {config_path}"

if args.kernel == "recurrent":
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

print(f"==> Building model: {cfg.model} ...")
model_module = importlib.import_module(f"models.networks.{cfg.model}")
ModelClass = model_module.Model

model = ModelClass(
    d_input=d_input,
    d_output=num_classes,
    d_model=cfg.d_model,
    n_layers=cfg.n_layers,
    dropout=cfg.dropout,
    prenorm=cfg.prenorm,
    lr=cfg.lr,
    kernel_type=args.kernel,
)
print(f"    d_input: {d_input} | d_output: {num_classes}")
# 加载权重
print(f"==> Loading weights: {best_pth}")
checkpoint = torch.load(best_pth, map_location=device)
model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()

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
latencies = []

print("==> Running inference on test set ...")
with torch.no_grad():
    for inputs, targets in tqdm(testloader, desc="Testing"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # 计时
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

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

        # 逐样本交叉熵损失
        batch_losses = nn.functional.cross_entropy(outputs, targets, reduction="none")

        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(max_probs.cpu().numpy().tolist())
        all_losses.extend(batch_losses.cpu().numpy().tolist())

all_targets = np.array(all_targets)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_losses = np.array(all_losses)
latencies = np.array(latencies)


###############################################################################
# 计算指标
###############################################################################

print("==> Computing metrics ...")

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


# 置信度裕度: top1 概率 - top2 概率（需要重新计算 top2）
# 为节省显存，这里用逐 batch 重新计算
all_margins = []
with torch.no_grad():
    for inputs, targets in tqdm(testloader, desc="Computing margins"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        sorted_probs, _ = probs.sort(dim=1, descending=True)
        margins = (sorted_probs[:, 0] - sorted_probs[:, 1]).cpu().numpy()
        all_margins.extend(margins.tolist())
all_margins = np.array(all_margins)
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
# 打印摘要
###############################################################################

print("\n" + "=" * 60)
print(f"  Experiment : {exp_dir}")
print(f"  Dataset    : {cfg.dataset}")
print(f"  Model      : {cfg.model}")
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
print("=" * 60)


###############################################################################
# 保存 metrics.json
###############################################################################

metrics = {
    "experiment_dir": exp_dir,
    "dataset": cfg.dataset,
    "model": cfg.model,
    "tokenizer": cfg.tokenizer,
    "checkpoint": "best.pth",
    "device": device,
    "test_samples": int(len(all_targets)),
    "num_classes": int(num_classes),
    "overall": overall,
    "per_class": per_class,
}

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
