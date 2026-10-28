"""训练入口（唯一实现）。

根目录的 `train.py` / `train_log.py` / `train_log_patch.py` 都是指向这里的薄壳，
保留文件名是为了让历史命令与队列脚本继续可用。

相对历史脚本新增：
- `--seed`：固定随机性（初始化 / shuffle / dropout），便于控制变量实验
- `--deterministic`：把 cudnn 切到确定性模式（更慢，但可逐位复现）
- `--kernel_type`：训练时选择 conv / recurrent 核
- 每次运行写 `meta.json`（git commit、argv、数据集规模、设备）
"""

import argparse
import importlib
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml

# 允许 `python scripts/train.py` 直接运行（否则 sys.path[0] 是 scripts/）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.dataloader import build_loaders
from datasets.specs import infer_d_input
from engine.checkpoint import (
    load_checkpoint,
    resolve_experiment_dir,
    restore_training_state,
)
from engine.logging import setup_logger, write_meta
from engine.trainer import Trainer
from optim.optimizer import setup_optimizer


def build_parser():
    parser = argparse.ArgumentParser(description="PyTorch Traffic Training")

    # Dataset
    parser.add_argument("--data_dir", default="/mnt/e/datasets/NetMamba", type=str, help="数据集根目录")
    parser.add_argument("--dataset", default="ISCXVPN2016", type=str, help="Dataset")
    parser.add_argument("--img_size", default=40, type=int, help="img_size (40x40)")

    # Dataloader
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers to use for dataloader")
    parser.add_argument("--max_samples", default=None, type=int, help="每个 split 随机抽样至多 N 个样本（快速冒烟，None 表示全量）")

    # Model
    parser.add_argument("--model", default="s4d_classification", type=str, help="Model name (should match a file in models/networks)")
    parser.add_argument("--n_layers", default=4, type=int, help="Number of layers")
    parser.add_argument("--d_model", default=128, type=int, help="Model dimension")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout")
    parser.add_argument("--prenorm", action="store_true", help="Prenorm")
    parser.add_argument("--kernel_type", default="conv", choices=["conv", "recurrent"], help="S4D kernel 类型（推理时可与训练不同）")

    # Tokenizer
    parser.add_argument("--tokenizer", type=str, default="pixel", choices=["pixel", "patch", "compressed_patch", "packet", "log_compressed_packet"], help="Tokenizer")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size")
    parser.add_argument("--compressed_size", type=int, default=2, help="Compressed patch size")
    parser.add_argument("--log_compressed_segment", type=int, default=64, help="Log-compressed segment size")

    # Training
    parser.add_argument("--epochs", default=100, type=int, help="Training epochs")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay")
    parser.add_argument("--mode", default="finetune", choices=["pretrain", "finetune"], help="Model training mode")

    # Reproducibility
    parser.add_argument("--seed", default=0, type=int, help="随机种子（初始化 / shuffle / dropout）")
    parser.add_argument("--deterministic", action="store_true", help="cudnn 确定性模式（更慢，可逐位复现）")

    # Checkpoint
    parser.add_argument("--resume", "-r", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--pretrained_path", default="", type=str, help="Pretrained model path")
    parser.add_argument("--save_dir", default="./checkpoint", type=str, help="Directory to save checkpoints")
    parser.add_argument("--save_freq", default=10, type=int, help="Save checkpoint every N epochs")

    # Early stopping
    parser.add_argument("--use_early_stop", default=False, action="store_true", help="Enable early stopping")
    parser.add_argument("--early_stop_patience", default=10, type=int, help="Early stopping patience")
    parser.add_argument("--early_stop_delta", default=0.001, type=float, help="Early stopping delta")

    # Config
    parser.add_argument("--config", "-c", default=None, type=str, help="Config file path")

    return parser


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False


def apply_config(args, parser):
    """用配置文件覆盖仍处于默认值的参数（命令行显式指定的优先）。"""
    if args.config is None:
        return

    with open(args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    for key, value in config.items():
        if hasattr(args, key):
            if getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
        else:
            print(f'Warning: Unknown config key "{key}"')


def build_model(args, d_input, num_classes):
    print(f"==> Building model: {args.model}")
    model_module = importlib.import_module(f"models.networks.{args.model}")
    ModelClass = model_module.Model

    if args.mode == "pretrain":
        model = ModelClass(
            d_input=d_input,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            prenorm=args.prenorm,
            lr=args.lr,
            kernel_type=args.kernel_type,
        )
        criterion = nn.MSELoss()
    else:
        model = ModelClass(
            d_input=d_input,
            d_output=num_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            prenorm=args.prenorm,
            lr=args.lr,
            kernel_type=args.kernel_type,
        )
        criterion = nn.CrossEntropyLoss()

    return model, criterion


def load_pretrained(model, path, device):
    print(f"==> Loading pretrained weights: {path}")
    checkpoint = load_checkpoint(path, device)
    pretrained_dict = checkpoint["model"]
    model_dict = model.state_dict()

    # 去掉分类头 / 重建头，其余按同名覆盖
    pretrained_dict = {
        key: value
        for key, value in pretrained_dict.items()
        if (
            key in model_dict
            and "head" not in key
            and "decoder" not in key
            and "reconstruction_head" not in key
        )
    }

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("==> Pretrained weights loaded")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    ###########################################################################
    # 实验目录（续训时从 checkpoint 推断）
    ###########################################################################
    if args.resume:
        exp_dir = resolve_experiment_dir(args.resume)
        args.config = os.path.join(exp_dir, "config.yaml")
    else:
        exp_dir = None

    ###########################################################################
    # 配置覆盖
    ###########################################################################
    apply_config(args, parser)

    ###########################################################################
    # 目录、配置、日志
    ###########################################################################
    if exp_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(args.save_dir, args.model, timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    if not args.resume:
        with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(vars(args), f, sort_keys=False)

    logger = setup_logger(os.path.join(exp_dir, "train.log"))
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
    else:
        logger.info("=" * 80)
        logger.info(f"Experiment: {exp_dir}")
        for key, value in vars(args).items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 80)

    ###########################################################################
    # 随机性
    ###########################################################################
    seed_everything(args.seed, args.deterministic)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ###########################################################################
    # 数据
    ###########################################################################
    print(f"==> Preparing {args.dataset} data..")
    loaders = build_loaders(args, max_samples=args.max_samples)

    trainloader = loaders["train"].loader
    valloader = loaders["valid"].loader if "valid" in loaders else None
    testloader = loaders["test"].loader if "test" in loaders else None
    num_classes = loaders["train"].num_classes

    print(f"Data shape: {trainloader.dataset[0][0].shape}")
    d_input = infer_d_input(args)

    ###########################################################################
    # 模型 / 优化器
    ###########################################################################
    model, criterion = build_model(args, d_input, num_classes)
    model = model.to(device)

    if device == "cuda" and not args.deterministic:
        cudnn.benchmark = True

    optimizer, scheduler = setup_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
    )

    ###########################################################################
    # 恢复 / 预训练权重
    ###########################################################################
    start_epoch = 0
    best_acc = 0.0
    early_stop_counter = 0

    if args.mode == "finetune":
        if args.resume:
            print("==> Resuming from checkpoint..")
            assert os.path.exists(args.resume), "Checkpoint file not found"
            checkpoint = load_checkpoint(args.resume, device)
            start_epoch, best_acc, early_stop_counter = restore_training_state(
                checkpoint, model, optimizer, scheduler
            )
            print(
                f'==> Resumed from checkpoint: {args.resume} | '
                f'Epoch: {checkpoint["epoch"]}'
            )
        elif args.pretrained_path and os.path.exists(args.pretrained_path):
            load_pretrained(model, args.pretrained_path, device)
        else:
            print("==> Training from scratch")
    else:
        print("==> Pretraining from scratch")

    ###########################################################################
    # 训练
    ###########################################################################
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        args=args,
        exp_dir=exp_dir,
        logger=logger,
    )
    trainer.start_epoch = start_epoch
    trainer.best_acc = best_acc
    trainer.early_stop_counter = early_stop_counter

    write_meta(
        exp_dir,
        args,
        datasets={split: len(bundle.dataset) for split, bundle in loaders.items()},
        extra={"device": device},
    )

    trainer.fit(trainloader, valloader, testloader)


if __name__ == "__main__":
    main()
