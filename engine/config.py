"""实验配置读取（`<exp_dir>/config.yaml`）。

主仓的 `test.py`、量化脚本等都需要"从一次训练运行里读回配置"；这里提供唯一实现，
避免每个入口各写一份 yaml 读取与兼容逻辑。
"""

import argparse
import os

import yaml


def load_experiment_config(exp_dir_or_path):
    """读取实验目录（或 config.yaml 路径），返回 argparse.Namespace。

    - 兼容旧版 config：没有 `tokenizer` 字段时按 `pixel` 处理；
    - 与训练时一致，`config.yaml` 里的键直接成为 Namespace 属性。
    """
    path = str(exp_dir_or_path)
    if os.path.isdir(path) or not path.endswith((".yaml", ".yml")):
        path = os.path.join(path, "config.yaml")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到配置文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    cfg = argparse.Namespace(**config)

    if not hasattr(cfg, "tokenizer"):
        cfg.tokenizer = "pixel"

    return cfg


def with_overrides(cfg, **overrides):
    """返回一个覆盖了指定字段的新 Namespace（不修改原对象）。"""
    values = vars(cfg).copy()
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return argparse.Namespace(**values)
