"""data.py —— 量化实验的数据加载（复用主仓 datasets 与 engine.config）。

与训练/评估共用同一套实现：
    - 配置读取：`engine.config.load_experiment_config`
    - 数据集与 transform：`datasets.dataloader.build_dataset` / `build_loader`

因此量化评估看到的样本、预处理和主仓 `test.py` 完全一致；
`max_samples` 使用固定 seed 抽样，保证不同位宽/策略比较的是同一批样本。
"""

import torch
from torch.utils.data import DataLoader

from .repo import ensure_repo_path

ensure_repo_path()

from datasets.dataloader import build_dataset, build_loader  # noqa: E402
from datasets.specs import make_input_spec  # noqa: E402
from engine.config import load_experiment_config, with_overrides  # noqa: E402


def infer_sequence_length(cfg, default_size: int = 40) -> int:
    """按 tokenizer 推导序列长度（用于显存估算里的分段位宽加权）。"""
    size = int(getattr(cfg, "img_size", default_size) or default_size)
    return int(make_input_spec(cfg).seq_len(size, size))


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


__all__ = ["load_experiment_config", "make_dataset", "make_loader", "infer_sequence_length"]
