"""数据加载：目录约定、样本校验与 DataLoader 构建的唯一实现。

- `SafeImageFolder`：过滤损坏文件，并把校验结果缓存到 `.cache/manifests/`
- `build_dataset` / `build_loaders`：训练、评估、量化、扫描脚本共用
- `build_dataloader`：兼容旧入口 `(train, val, test, num_classes)` 的薄包装
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

from datasets.manifest import dataset_fingerprint, load_cached_valid, save_cached_valid
from datasets.transforms import build_transform


def _is_valid_image(path):
    """Check if file exists and is a valid image."""
    if not os.path.isfile(path):
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


class SafeImageFolder(ImageFolder):
    """ImageFolder that filters out missing/corrupt files to prevent DataLoader crashes.

    校验结果按数据目录缓存在 `.cache/manifests/`（见 `datasets/manifest.py`），
    数据未变动时不会重复做 `PIL.Image.verify()`。
    """

    def __init__(self, root, transform=None, target_transform=None, cache=True):
        super().__init__(root, transform=transform, target_transform=target_transform)

        original_count = len(self.samples)
        valid_paths = None
        fingerprint = None

        if cache:
            try:
                fingerprint = dataset_fingerprint(self.samples)
                cached = load_cached_valid(self.root, fingerprint)
                if cached is not None:
                    wanted = set(cached)
                    valid_paths = [
                        path for path, _ in self.samples
                        if os.path.relpath(path, self.root) in wanted
                    ]
            except OSError:
                valid_paths = None

        if valid_paths is None:
            valid_paths = [path for path, _ in self.samples if _is_valid_image(path)]
            if cache and fingerprint is not None:
                try:
                    save_cached_valid(
                        self.root,
                        fingerprint,
                        [os.path.relpath(path, self.root) for path in valid_paths],
                    )
                except OSError:
                    pass

        valid_set = set(valid_paths)
        valid = [(path, target) for path, target in self.samples if path in valid_set]

        if len(valid) < original_count:
            print(
                f"[SafeImageFolder] {root}: filtered "
                f"{original_count - len(valid)} invalid files, {len(valid)} remaining"
            )

        self.samples = valid
        self.targets = [target for _, target in valid]
        self.imgs = self.samples


@dataclass
class SplitData:
    """一个 split 的数据集、DataLoader 与类别信息。"""

    split: str
    dataset: Dataset
    loader: DataLoader
    num_classes: int
    class_names: List[str]


def _unwrap(dataset: Dataset) -> Dataset:
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def class_names_of(dataset: Dataset) -> List[str]:
    return list(_unwrap(dataset).classes)


def split_dir(args, split: str) -> str:
    if getattr(args, "mode", "finetune") == "pretrain":
        # 预训练只用 <dataset>/train，没有 dataset_sampled 划分
        return os.path.join(args.data_dir, args.dataset, "train")
    return os.path.join(args.data_dir, args.dataset, "dataset_sampled", split)


def build_dataset(
    args,
    split: str,
    transform=None,
    max_samples: Optional[int] = None,
    seed: int = 0,
    cache: bool = True,
) -> Dataset:
    """构建单个 split 的数据集；`max_samples` 用于快速冒烟/扫描。"""
    transform = transform or build_transform(args)
    dataset = SafeImageFolder(split_dir(args, split), transform=transform, cache=cache)

    if max_samples is not None and 0 < max_samples < len(dataset):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
        dataset = Subset(dataset, indices)

    return dataset


def build_loader(args, dataset: Dataset, training: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.num_workers,
    )


def build_loaders(
    args,
    splits=None,
    max_samples: Optional[int] = None,
    seed: int = 0,
    cache: bool = True,
) -> Dict[str, SplitData]:
    """构建若干 split，返回 {split: SplitData}。"""
    if splits is None:
        if getattr(args, "mode", "finetune") == "pretrain":
            splits = ("train",)
        else:
            splits = ("train", "valid", "test")

    transform = build_transform(args)
    bundles: Dict[str, SplitData] = {}
    for split in splits:
        dataset = build_dataset(
            args,
            split,
            transform=transform,
            max_samples=max_samples,
            seed=seed,
            cache=cache,
        )
        bundles[split] = SplitData(
            split=split,
            dataset=dataset,
            loader=build_loader(args, dataset, training=(split == "train")),
            num_classes=len(class_names_of(dataset)),
            class_names=class_names_of(dataset),
        )
    return bundles


def build_dataloader(args):
    """向后兼容旧入口。

    返回 (trainloader, valloader, testloader, num_classes)；
    预训练模式返回 (trainloader, None, None, 1)。
    """
    bundles = build_loaders(args)

    if getattr(args, "mode", "finetune") == "pretrain":
        return bundles["train"].loader, None, None, 1

    return (
        bundles["train"].loader,
        bundles["valid"].loader,
        bundles["test"].loader,
        bundles["train"].num_classes,
    )
