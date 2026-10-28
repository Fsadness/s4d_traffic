"""兼容入口：训练逻辑已统一到 `scripts/train.py`（Phase 2）。

这是当前最常用的训练文件名；参数完全保持不变，并新增
`--seed` / `--deterministic` / `--kernel_type` / `--max_samples`。
"""

from scripts.train import main

if __name__ == "__main__":
    main()
