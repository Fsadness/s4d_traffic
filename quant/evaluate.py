"""evaluate.py —— 量化模型评估（精度 / 延迟 / 吞吐）。

只做“单次前向”的精度与延迟（递推模式，反映 edge 部署形态），
指标计算复用主仓的 `engine.metrics.compute_metrics`，与 `test.py` 口径一致。
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from .repo import ensure_repo_path

ensure_repo_path()

from engine.metrics import compute_metrics  # noqa: E402


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: str = "cuda",
    max_batches: int = None,
    measure_latency: bool = True,
    verbose: bool = False,
) -> dict:
    """在给定 DataLoader 上评估。

    Returns:
        dict(acc, loss, n_samples, n_batches, latency_ms, throughput,
             macro_f1, macro_precision, macro_recall, avg_confidence, avg_margin)
    """
    model.eval()

    targets_all = []
    preds_all = []
    probs_all = []
    losses_all = []
    margins_all = []
    latencies = []

    n_batches = 0
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = model(inputs)

        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        probs = torch.softmax(logits, dim=1)
        max_probs, preds = probs.max(dim=1)
        sorted_probs, _ = probs.sort(dim=1, descending=True)
        margins = (sorted_probs[:, 0] - sorted_probs[:, 1]).cpu().numpy()

        batch_losses = F.cross_entropy(logits, targets, reduction="none")

        batch_size = inputs.size(0)
        if measure_latency:
            latencies.extend([(t1 - t0) / batch_size] * batch_size)
        else:
            latencies.extend([float("nan")] * batch_size)

        targets_all.extend(targets.cpu().tolist())
        preds_all.extend(preds.cpu().tolist())
        probs_all.extend(max_probs.cpu().tolist())
        losses_all.extend(batch_losses.cpu().tolist())
        margins_all.extend(margins.tolist())

        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    targets_all = np.asarray(targets_all)
    preds_all = np.asarray(preds_all)
    probs_all = np.asarray(probs_all)
    losses_all = np.asarray(losses_all)
    margins_all = np.asarray(margins_all)
    latencies = np.asarray(latencies)

    num_classes = int(getattr(model, "d_output", 0)) or int(
        max(targets_all.max(), preds_all.max()) + 1
    )
    class_names = [str(i) for i in range(num_classes)]

    result = compute_metrics(
        targets_all,
        preds_all,
        probs_all,
        losses_all,
        latencies,
        class_names,
        float(margins_all.mean()),
    )
    overall = result["overall"]

    if verbose:
        print(
            f"[eval] samples={len(targets_all)} acc={overall['accuracy']:.2f}% "
            f"loss={overall['avg_loss']:.4f}"
        )

    return {
        "acc": overall["accuracy"],
        "loss": overall["avg_loss"],
        "n_samples": int(len(targets_all)),
        "n_batches": n_batches,
        "latency_ms": overall["avg_latency_ms"],
        "throughput": overall["throughput_samples_per_sec"],
        "macro_f1": overall["macro_f1"],
        "macro_precision": overall["macro_precision"],
        "macro_recall": overall["macro_recall"],
        "avg_confidence": overall["avg_confidence"],
        "avg_margin": overall["avg_margin"],
    }


@torch.no_grad()
def evaluate_quantization_off(model, loader, device="cuda", max_batches=None) -> dict:
    """临时关闭伪量化，得到“同一结构的 FP32 递推基线”。

    用于确认“递推实现本身”没有引入精度损失（应与卷积模式几乎一致）。
    """
    model.set_quantization(False)
    try:
        return evaluate(model, loader, device=device, max_batches=max_batches)
    finally:
        model.set_quantization(True)
