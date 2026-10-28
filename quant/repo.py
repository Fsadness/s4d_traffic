"""repo.py —— 定位仓库根目录并把它加入 sys.path。

合并进 s4d 仓库后，量化包位于 `<repo>/quant/`，因此默认根目录就是上两级；
独立仓库使用时仍可用环境变量 `S4D_REPO` 指定其他 s4d 副本。
"""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_root() -> str:
    """返回仓库根目录（优先环境变量 S4D_REPO）。"""
    return os.environ.get("S4D_REPO", PROJECT_ROOT)


def ensure_repo_path() -> str:
    """把仓库根目录插入 sys.path（幂等），返回该路径。"""
    root = repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
