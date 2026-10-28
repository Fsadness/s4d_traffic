"""兼容入口：训练逻辑已统一到 `scripts/train.py`（Phase 2）。

保留该文件名是为了让历史命令、队列脚本与 muscle memory 继续可用。
行为差异（相对于最早的 train.py）：现在会写 `train.log`、支持早停与 tokenizer，
并支持 `--seed` / `--kernel_type` / `--max_samples` 等参数，详见 `scripts/train.py`。
"""

from scripts.train import main

if __name__ == "__main__":
    main()
