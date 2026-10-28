"""Checkpoint 读写、推理加载与训练恢复。

把三件容易写错的事收敛到一处：

1. **加载顺序**：S4D 的卷积核在 `eval()` 时会缓存离散参数，缓存必须在
   `load_state_dict` 之后重建。实测顺序错误时 conv→recurrent 的 logits
   相差约 0.18，正确顺序约 1e-7。
2. **字段一致**：所有训练入口都通过 `build_state` 写同一组键
   （model / optimizer / scheduler / acc / epoch / early_stop_counter）。
3. **目录推断**：`--resume` 可能指向 `checkpoints/epoch_xxx.pth`，需要回到实验根目录。
"""

import os

import torch


def resolve_experiment_dir(resume_path: str) -> str:
    """从 checkpoint 路径推断实验目录（兼容 checkpoints/ 子目录）。"""
    path = os.path.abspath(resume_path)
    directory = os.path.dirname(path)
    if os.path.basename(directory) == "checkpoints":
        return os.path.dirname(directory)
    return directory


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device)


def build_state(model, optimizer=None, scheduler=None, **extra):
    """构造训练 checkpoint；字段与历史 run 保持一致。"""
    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    state.update(extra)
    return state


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)
    return path


def load_fp_checkpoint(model, path, device="cpu", strict=True):
    """加载 FP 权重并切到推理态。

    顺序固定为 load -> to(device) -> eval -> prepare_for_inference()；
    最后一步负责用新权重重建离散参数缓存，不能省略。

    Returns:
        (model, checkpoint)
    """
    checkpoint = load_checkpoint(path, device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    model.load_state_dict(state_dict, strict=strict)
    model = model.to(device)
    model.eval()

    if hasattr(model, "prepare_for_inference"):
        model.prepare_for_inference()

    return model, checkpoint


def restore_training_state(checkpoint, model, optimizer=None, scheduler=None):
    """从训练 checkpoint 恢复模型/优化器/调度器。

    Returns:
        (start_epoch, best_acc, early_stop_counter)
    """
    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_acc = float(checkpoint.get("acc", 0.0))
    early_stop_counter = int(checkpoint.get("early_stop_counter", 0))

    return start_epoch, best_acc, early_stop_counter
