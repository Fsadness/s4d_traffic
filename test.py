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
import sys
import json
import time
import argparse
import importlib

import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# 项目内部模块 — 与 train_log_patch.py 保持一致
from datasets.dataloader import build_dataloader


###############################################################################
# 参数解析
###############################################################################

parser = argparse.ArgumentParser(description="Evaluate a trained S4D model")
parser.add_argument("exp_dir", type=str, help="实验文件夹路径 (含 best.pth / config.yaml)")
parser.add_argument("--batch_size", type=int, default=None, help="覆盖 config 中的 batch_size")
parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
parser.add_argument("--device", type=str, default=None, help="cuda / cpu, 默认自动检测")
parser.add_argument("--kernel", type=str, default="conv", choices=["conv", "recurrent"], help="覆盖 config 中的 kernel_type")
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

# 使用 argparse namespace 兼容 build_dataloader / build_transform 的 args 访问方式
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
# 构建数据集（仅需 testloader）
###############################################################################

from datasets.transforms import build_transform
from datasets.dataloader import SafeImageFolder

def load_test_only(args):
    """只加载 test 集，跳过 train/val，避免双重 PIL.Image.verify() 遍历。"""
    transform = build_transform(args)
    data_dir = os.path.join(args.data_dir, args.dataset, 'dataset_sampled')
    test_dir = os.path.join(data_dir, 'test')
    testset = SafeImageFolder(root=test_dir, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    num_classes = len(testset.classes)
    class_names = testset.classes
    return testloader, num_classes, class_names

print(f"==> Loading dataset: {cfg.dataset} ...")
testloader, num_classes, class_names = load_test_only(cfg)
print(f"    Classes: {num_classes} | Test samples: {len(testloader.dataset)}")


###############################################################################
# 计算 d_input（与 train 保持一致）
###############################################################################

if cfg.tokenizer == "pixel":
    d_input = 1
elif cfg.tokenizer == "patch":
    d_input = cfg.patch_size * cfg.patch_size
elif cfg.tokenizer == "compressed_patch":
    d_input = cfg.compressed_size * cfg.compressed_size
elif cfg.tokenizer in ("packet", "log_compressed_packet"):
    d_input = 5
else:
    raise ValueError(f"Unsupported tokenizer: {cfg.tokenizer}")


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

# --- 总体指标 ---
overall_accuracy = accuracy_score(all_targets, all_preds) * 100  # %
macro_precision = precision_score(all_targets, all_preds, average="macro", zero_division=0) * 100
macro_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0) * 100
macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100
weighted_precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0) * 100
weighted_recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0) * 100
weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0) * 100

avg_loss = float(all_losses.mean())
avg_confidence = float(all_probs.mean())

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:

    outputs=model(inputs)

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

avg_latency_ms = float(latencies.mean() * 1000)  # ms/sample
throughput = float(1.0 / latencies.mean())        # samples/sec

# --- 每类别指标 ---
per_class_precision = precision_score(all_targets, all_preds, average=None, zero_division=0) * 100
per_class_recall = recall_score(all_targets, all_preds, average=None, zero_division=0) * 100
per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0) * 100
support = np.bincount(all_targets, minlength=num_classes)

# --- 混淆矩阵 ---
cm = confusion_matrix(all_targets, all_preds)
# 归一化（按行，即每类真实样本数）
cm_normalized = cm.astype(np.float64)
row_sums = cm_normalized.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # 避免除零
cm_normalized = cm_normalized / row_sums


###############################################################################
# 打印摘要
###############################################################################

print("\n" + "=" * 60)
print(f"  Experiment : {exp_dir}")
print(f"  Dataset    : {cfg.dataset}")
print(f"  Model      : {cfg.model}")
print(f"  Device     : {device}")
print("=" * 60)
print(f"  Accuracy           : {overall_accuracy:.3f}%")
print(f"  Macro Precision    : {macro_precision:.3f}%")
print(f"  Macro Recall       : {macro_recall:.3f}%")
print(f"  Macro F1           : {macro_f1:.3f}%")
print(f"  Weighted F1        : {weighted_f1:.3f}%")
print(f"  Avg Loss (CE)      : {avg_loss:.4f}")
print(f"  Avg Confidence     : {avg_confidence:.4f}")
print(f"  Avg Margin         : {avg_margin:.4f}")
print(f"  Avg Latency        : {avg_latency_ms:.3f} ms/sample")
print(f"  Throughput         : {throughput:.1f} samples/sec")
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
    "overall": {
        "accuracy": round(overall_accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_precision, 4),
        "weighted_recall": round(weighted_recall, 4),
        "weighted_f1": round(weighted_f1, 4),
        "avg_loss": round(avg_loss, 6),
        "avg_confidence": round(avg_confidence, 6),
        "avg_margin": round(avg_margin, 6),
        "avg_latency_ms": round(avg_latency_ms, 4),
        "throughput_samples_per_sec": round(throughput, 2),
    },
    "per_class": {
        class_names[i]: {
            "precision": round(float(per_class_precision[i]), 4),
            "recall": round(float(per_class_recall[i]), 4),
            "f1": round(float(per_class_f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(num_classes)
    },
}

metrics_path = os.path.join(eval_dir, "metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"[Saved] {metrics_path}")


###############################################################################
# 保存 classification_report.csv
###############################################################################

# 使用 sklearn 的 classification_report 获取完整文本，然后解析为 CSV
report_dict = classification_report(
    all_targets, all_preds,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

import csv

csv_path = os.path.join(eval_dir, "classification_report.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["class", "precision", "recall", "f1-score", "support"])
    for cls_name in class_names:
        row = report_dict[cls_name]
        writer.writerow([
            cls_name,
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['f1-score']:.4f}",
            int(row["support"]),
        ])
    # macro avg
    row = report_dict["macro avg"]
    writer.writerow([
        "macro avg",
        f"{row['precision']:.4f}",
        f"{row['recall']:.4f}",
        f"{row['f1-score']:.4f}",
        int(row["support"]),
    ])
    # weighted avg
    row = report_dict["weighted avg"]
    writer.writerow([
        "weighted avg",
        f"{row['precision']:.4f}",
        f"{row['recall']:.4f}",
        f"{row['f1-score']:.4f}",
        int(row["support"]),
    ])
print(f"[Saved] {csv_path}")


###############################################################################
# 保存 confusion_matrix.png
###############################################################################

def plot_confusion_matrix(cm_norm, class_names, save_path, figsize=None, cmap="Blues"):
    """绘制归一化混淆矩阵并保存。

    针对大类别数做了内存保护：
    - figsize 上限 24 英寸，避免图像尺寸爆炸（181 类时原代码会产生 81 英寸 ≈ 2.65 亿像素）
    - DPI 随类别数自适应降低
    - 超过 40 类时不显示刻度标签，避免重叠
    - 超过 30 类时不逐格写数值，避免创建数万个 Text 对象
    """
    n = len(class_names)

    # 动态调整尺寸 — 上限 24 英寸
    if figsize is None:
        base = max(8, min(24, n * 0.45))
        figsize = (base, base)

    # DPI 随类别数自适应：小矩阵用 200，大矩阵降到 100
    dpi = 200 if n <= 60 else 100

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, aspect="auto")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 标签
    tick_marks = np.arange(n)
    if n <= 40:
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=90, fontsize=max(4, 10 - n // 8))
        ax.set_yticklabels(class_names, fontsize=max(4, 10 - n // 8))
    else:
        # 类别太多时只显示部分刻度
        step = max(1, n // 20)
        subset_ticks = tick_marks[::step]
        ax.set_xticks(subset_ticks)
        ax.set_yticks(subset_ticks)
        ax.set_xticklabels([class_names[i] for i in subset_ticks], rotation=90, fontsize=6)
        ax.set_yticklabels([class_names[i] for i in subset_ticks], fontsize=6)

    # 在格子中填数值（超过 30 类时跳过，避免创建数万个 Text 对象导致 OOM）
    if n <= 30:
        thresh = cm_norm.max() / 2.0
        for i in range(n):
            for j in range(n):
                val = cm_norm[i, j]
                if val > 0:
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=max(3, 8 - n // 6),
                    )

    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Normalized Confusion Matrix", fontsize=14)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path}")


cm_path = os.path.join(eval_dir, "confusion_matrix.png")
plot_confusion_matrix(cm_normalized, class_names, cm_path)


###############################################################################
# 完成
###############################################################################

print(f"\n==> All results saved to: {eval_dir}")
print("Done.")
