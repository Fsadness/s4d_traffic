"""qat.py —— 量化感知训练（QAT / QAFT）。

论文做法（V 节）：
    - 从 PTQ 后的模型出发，在 **递推模式** 下继续训练 10 个 epoch
    - 学习率 1e-4（比预训练的 1e-3 小一个数量级）
    - 前向用伪量化模拟整数运算，反向用 STE 直通
    - 梯度逐元素裁剪到 [-1000, 1000]，防止递推中的梯度爆炸
    - 三种参数化：continuous / discrete / frozen

本模块只负责训练循环本身；模型构建与校准在 model.py / calibrate.py。
"""

import torch
import torch.nn.functional as F


def clamp_gradients(model, grad_clip: float):
    """逐元素裁剪梯度（论文的 [-1000, +1000]）。

    注意：不是按范数裁剪。递推展开后个别维度梯度可能极大，
    逐元素裁剪更贴近论文描述，也更适合定点硬件的训练模拟。
    """
    if grad_clip <= 0:
        return
    for p in model.parameters():
        if p.grad is not None:
            p.grad.clamp_(-grad_clip, grad_clip)


def train_qat(
    model,
    train_loader,
    device: str = "cuda",
    epochs: int = 10,
    lr: float = 1e-4,
    grad_clip: float = 1000.0,
    max_batches_per_epoch: int = None,
    val_loader=None,
    max_val_batches: int = None,
    log_fn=print,
) -> list:
    """递推模式下的 QAT 微调。

    Args:
        model: 已校准的 QuantS4DClassification（量化开关已打开）
        train_loader: 训练集 DataLoader
        epochs: 论文用 10
        lr: 论文用 1e-4
        grad_clip: 论文用 1000
        max_batches_per_epoch: 限制每个 epoch 的 batch 数（快速验证用）
        val_loader: 可选，每个 epoch 结束后评估
        log_fn: 日志函数

    Returns:
        history: 每个 epoch 的 dict 列表
    """
    model.train()
    model.set_quantization(True)     # 训练时也必须伪量化，否则不是 QAT
    model.set_observing(False)       # 静态校准已固定，不再更新范围

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0)

    # 递推 QAT 每个 batch 都很慢，给一个进度条（含 ETA）便于观察
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None

    history = []

    for epoch in range(epochs):
        total = 0
        correct = 0
        loss_sum = 0.0
        n_batches = 0

        iterator = train_loader
        if tqdm is not None:
            iterator = tqdm(
                train_loader,
                desc=f"QAT epoch {epoch + 1}/{epochs}",
                leave=False,
                total=(max_batches_per_epoch or len(train_loader)),
            )

        for inputs, targets in iterator:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clamp_gradients(model, grad_clip)   # 论文的梯度裁剪
            optimizer.step()

            loss_sum += loss.item() * targets.numel()
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.numel()
            n_batches += 1

            if tqdm is not None:
                iterator.set_postfix(
                    loss=f"{loss_sum / total:.4f}",
                    acc=f"{100.0 * correct / total:.2f}%",
                )

            if max_batches_per_epoch is not None and n_batches >= max_batches_per_epoch:
                break

        record = {
            "epoch": epoch,
            "train_loss": loss_sum / total,
            "train_acc": 100.0 * correct / total,
            "n_samples": total,
        }

        if val_loader is not None:
            from .evaluate import evaluate

            val = evaluate(model, val_loader, device=device, max_batches=max_val_batches)
            record.update({"val_acc": val["acc"], "val_loss": val["loss"]})
            model.train()

        history.append(record)
        log_fn(
            f"[QAT] epoch {epoch + 1}/{epochs} | "
            f"loss {record['train_loss']:.4f} | acc {record['train_acc']:.2f}%"
            + (f" | val {record['val_acc']:.2f}%" if "val_acc" in record else "")
        )

    return history
