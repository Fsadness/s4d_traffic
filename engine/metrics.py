"""评估指标、分类报告与混淆矩阵。

评估逻辑（`test.py`、`test_inference.py` 以及后续的量化评估）统一调用本模块，
避免每个入口各写一份 sklearn 调用与绘图代码。

约定：
- 输入都是逐样本的 numpy 数组（targets / preds / probs / losses / latencies）
- `compute_metrics` 返回的结构与 `metrics.json` 中的 `overall` / `per_class` 一致
- 混淆矩阵按行归一化（每类真实样本数）
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(targets, preds, probs, losses, latencies, class_names, avg_margin):
    """计算总体与逐类指标，返回 {"overall": {...}, "per_class": {...}}。"""
    num_classes = len(class_names)
    # 固定类别标签，保证小样本/缺类时指标与混淆矩阵的维度仍然稳定
    labels = np.arange(num_classes)

    overall_accuracy = accuracy_score(targets, preds) * 100  # %
    macro_precision = precision_score(targets, preds, labels=labels, average="macro", zero_division=0) * 100
    macro_recall = recall_score(targets, preds, labels=labels, average="macro", zero_division=0) * 100
    macro_f1 = f1_score(targets, preds, labels=labels, average="macro", zero_division=0) * 100
    weighted_precision = precision_score(targets, preds, labels=labels, average="weighted", zero_division=0) * 100
    weighted_recall = recall_score(targets, preds, labels=labels, average="weighted", zero_division=0) * 100
    weighted_f1 = f1_score(targets, preds, labels=labels, average="weighted", zero_division=0) * 100

    avg_loss = float(losses.mean())
    avg_confidence = float(probs.mean())
    avg_latency_ms = float(latencies.mean() * 1000)  # ms/sample
    throughput = float(1.0 / latencies.mean())       # samples/sec

    per_class_precision = precision_score(targets, preds, labels=labels, average=None, zero_division=0) * 100
    per_class_recall = recall_score(targets, preds, labels=labels, average=None, zero_division=0) * 100
    per_class_f1 = f1_score(targets, preds, labels=labels, average=None, zero_division=0) * 100
    support = np.bincount(targets, minlength=num_classes)

    return {
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


def normalized_confusion(targets, preds, num_classes=None):
    """返回 (原始混淆矩阵, 行归一化混淆矩阵)。

    传入 num_classes 时固定标签范围，样本里缺类也能保持 N×N 形状。
    """
    labels = None if num_classes is None else np.arange(num_classes)
    cm = confusion_matrix(targets, preds, labels=labels)
    cm_normalized = cm.astype(np.float64)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    cm_normalized = cm_normalized / row_sums
    return cm, cm_normalized


def write_classification_report(path, targets, preds, class_names):
    """写逐类 P/R/F1 + macro/weighted 平均的 CSV，返回路径。"""
    labels = np.arange(len(class_names))
    report_dict = classification_report(
        targets, preds,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    with open(path, "w", newline="", encoding="utf-8") as f:
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
        for avg_name in ("macro avg", "weighted avg"):
            row = report_dict[avg_name]
            writer.writerow([
                avg_name,
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1-score']:.4f}",
                int(row["support"]),
            ])

    return path


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
