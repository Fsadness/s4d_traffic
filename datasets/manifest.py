"""样本清单缓存。

`SafeImageFolder` 启动时会对每个文件做 `PIL.Image.verify()`，6 个数据集里
最大的一份有 3.7 万个训练样本，单次扫描约 1 分钟；做参数扫描时这笔开销会
在每个 run 上重复一次。

这里把"哪些文件是有效图片"的结果缓存到仓库的 `.cache/manifests/` 下，用
(相对路径, 文件大小, mtime) 的哈希做指纹：文件没变就直接复用，删了/改了
自动失效并重新校验。

缓存只影响启动速度，不改变样本集合的任何语义。
"""

import hashlib
import json
import os
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "manifests")
CACHE_VERSION = 1


def dataset_fingerprint(samples) -> str:
    """对 (path, class) 列表做指纹；任何文件增删改都会改变结果。"""
    digest = hashlib.sha1()
    for path, _ in samples:
        try:
            stat = os.stat(path)
        except OSError:
            digest.update(f"{path}\tMISSING\n".encode("utf-8"))
            continue
        digest.update(
            f"{path}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _cache_path(root: str, cache_dir: str) -> str:
    key = hashlib.sha1(os.path.abspath(root).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.json")


def load_cached_valid(root: str, fingerprint: str, cache_dir: str = None):
    """返回缓存的相对路径列表；未命中返回 None。"""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    path = _cache_path(root, cache_dir)
    if not os.path.isfile(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None

    if payload.get("version") != CACHE_VERSION:
        return None
    if payload.get("root") != os.path.abspath(root):
        return None
    if payload.get("fingerprint") != fingerprint:
        return None

    valid = payload.get("valid")
    return valid if isinstance(valid, list) else None


def save_cached_valid(root: str, fingerprint: str, valid_relpaths, cache_dir: str = None) -> None:
    """原子写入缓存；失败时静默放弃（缓存不是正确性的一部分）。"""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    payload = {
        "version": CACHE_VERSION,
        "root": os.path.abspath(root),
        "fingerprint": fingerprint,
        "valid": sorted(valid_relpaths),
    }

    path = _cache_path(root, cache_dir)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except OSError:
        if os.path.exists(tmp):
            os.remove(tmp)
