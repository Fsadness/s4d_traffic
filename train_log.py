"""兼容入口：训练逻辑已统一到 `scripts/train.py`（Phase 2）。

`train_log.py` 原来只支持 pixel 输入；现在 tokenizer 默认仍是 pixel，
但可以选择 `--tokenizer patch/compressed_patch/packet/log_compressed_packet`。
"""

from scripts.train import main

if __name__ == "__main__":
    main()
