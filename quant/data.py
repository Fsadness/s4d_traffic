"""data.py —— 量化实验的数据加载（复用主仓 datasets 与 engine.config）。

与训练/评估共用同一套实现：
    - 配置读取：`engine.config.load_experiment_config`
    - 数据集与 transform：`datasets.dataloader.build_dataset` / `build_loader`

因此量化评估看到的样本、预处理和主仓 `test.py` 完全一致；
`max_samples` 使用固定 seed 抽样，保证不同位宽/策略比较的是同一批样本。
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .repo import ensure_repo_path

ensure_repo_path()

from datasets.dataloader import build_dataset, build_loader  # noqa: E402
from datasets.specs import make_input_spec  # noqa: E402
from engine.config import load_experiment_config, with_overrides  # noqa: E402


def infer_sequence_length(cfg, default_size: int = 40) -> int:
    """按 tokenizer 推导序列长度（用于显存估算里的分段位宽加权）。"""
    size = int(getattr(cfg, "img_size", default_size) or default_size)
    return int(make_input_spec(cfg).seq_len(size, size))


def stratified_indices(targets, fraction: float, seed: int = 0):
    """按类别分层抽样：每类取 round(n_c * fraction) 个（至少 1 个）。"""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction 必须在 (0, 1] 内，收到 {fraction}")

    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    chosen = []

    for label in np.unique(targets):
        class_indices = np.where(targets == label)[0]
        count = max(1, int(round(len(class_indices) * fraction)))
        count = min(count, len(class_indices))
        chosen.extend(rng.choice(class_indices, size=count, replace=False).tolist())

    return sorted(chosen)


def class_counts(targets, indices):
    """返回 {label: count}，便于日志确认分层是否合理。"""
    targets = np.asarray(targets)
    return {
        int(label): int((targets[indices] == label).sum())
        for label in np.unique(targets)
    }


def make_dataset(
    cfg,
    split: str = "test",
    data_dir: str = None,
    max_samples: int = None,
    seed: int = 0,
):
    """构建数据集（按需子集）。"""
    cfg = with_overrides(cfg, data_dir=data_dir, mode="finetune")
    return build_dataset(cfg, split, max_samples=max_samples, seed=seed)


def make_loader(
    cfg,
    split: str = "test",
    data_dir: str = None,
    batch_size: int = None,
    num_workers: int = None,
    max_samples: int = None,
    seed: int = 0,
    shuffle: bool = None,
) -> DataLoader:
    """构建 DataLoader（与主仓共用 build_loader 的 batch/worker 语义）。"""
    if shuffle is None:
        shuffle = split == "train"

    cfg = with_overrides(
        cfg,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        mode="finetune",
    )
    dataset = build_dataset(cfg, split, max_samples=max_samples, seed=seed)
    return build_loader(cfg, dataset, training=shuffle)


def make_stratified_loader(
    cfg,
    split: str = "train",
    fraction: float = 0.1,
    data_dir: str = None,
    batch_size: int = None,
    num_workers: int = None,
    seed: int = 0,
    shuffle: bool = True,
):
    """按类别分层抽取 `fraction` 的训练子集，返回 (loader, stats)。

    用于 QAT 的"小比例数据能否恢复精度"实验：每类按比例抽样，
    避免随机子集在少数类上抽样过少导致结论失真。
    """
    cfg = with_overrides(
        cfg,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        mode="finetune",
    )
    dataset = build_dataset(cfg, split, max_samples=None, seed=seed)
    indices = stratified_indices(dataset.targets, fraction, seed=seed)
    subset = Subset(dataset, indices)

    stats = {
        "fraction": fraction,
        "total": len(indices),
        "per_class": class_counts(dataset.targets, indices),
    }
    return build_loader(cfg, subset, training=shuffle), stats


__all__ = [
    "load_experiment_config",
    "make_dataset",
    "make_loader",
    "make_stratified_loader",
    "infer_sequence_length",
    "stratified_indices",
    "class_counts",
]
