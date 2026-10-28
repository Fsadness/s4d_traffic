"""训练日志与运行元数据。

- `setup_logger`：写入 `<exp_dir>/train.log`，格式与历史 run 保持一致
  （`%(asctime)s | %(message)s`）。
- `write_meta`：写 `<exp_dir>/meta.json`，记录 seed、git 提交、完整参数与数据集规模，
  供后续批量实验汇总/追溯使用。
"""

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime

import torch


def setup_logger(log_file):
    """创建实验 logger；重复调用会重建 handler，避免同一进程内重复写。"""
    logger = logging.getLogger("experiment")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(file_handler)

    return logger


def _git(*args):
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_info():
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def write_meta(exp_dir, args, datasets=None, extra=None):
    """写 meta.json；不覆盖已有文件（续训时保留首次运行的元数据）。"""
    path = os.path.join(exp_dir, "meta.json")
    if os.path.exists(path):
        return path

    meta = {
        "run_id": os.path.basename(os.path.abspath(exp_dir)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "git": git_info(),
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": extra.get("device") if extra else None,
        },
        "config": {key: value for key, value in vars(args).items()},
        "datasets": datasets or {},
    }

    if extra:
        meta.update({k: v for k, v in extra.items() if k != "device"})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return path
