"""输入方案（InputSpec）：tokenizer 选择、参数与形状推导的唯一入口。

在此之前，`d_input` 的 if/elif 在 train_log_patch.py、test.py 里各写了一遍，
新增一种输入方案要改多处。这里把"方案名 → tokenizer 实例 → d_input / 序列长度"
收敛到一份，训练、评估、量化脚本都只调用本模块。

用法：
    spec = make_input_spec(args)      # 从 argparse / config Namespace 解析
    spec.d_input                      # 模型输入维度
    spec.seq_len(40, 40)              # 序列长度（已知图像尺寸时）
    spec.describe()                   # "patch(patch_size=4)"，用于日志与 run 名
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from datasets import tokenizer as _tokenizer


# 方案名 -> tokenizer 类。新增输入方案只需在这里注册 + 写一个 tokenizer 类。
TOKENIZERS = {
    "pixel": _tokenizer.PixelTokenizer,
    "patch": _tokenizer.PatchTokenizer,
    "compressed_patch": _tokenizer.CompressedPatchTokenizer,
    "packet": _tokenizer.PacketTokenizer,
    "log_compressed_packet": _tokenizer.LogCompressedPacketTokenizer,
}


@dataclass(frozen=True)
class InputSpec:
    """一种输入方案：方案名 + tokenizer 构造参数。"""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def build(self):
        """实例化 tokenizer（可调用对象，输入 (C,H,W) 张量）。"""
        if self.name not in TOKENIZERS:
            raise ValueError(
                f"Unsupported tokenizer type: {self.name}. "
                f"Available: {sorted(TOKENIZERS)}"
            )
        return TOKENIZERS[self.name](**self.params)

    @property
    def d_input(self) -> int:
        return int(self.build().d_input)

    def seq_len(self, height: int, width: int) -> int:
        return int(self.build().seq_len(height, width))

    def describe(self) -> str:
        if not self.params:
            return self.name
        args = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({args})"


def make_input_spec(args) -> InputSpec:
    """从 argparse.Namespace / config 解析输入方案。

    兼容旧版 config：没有 `tokenizer` 字段时按 `pixel` 处理。
    """
    name = getattr(args, "tokenizer", None) or "pixel"

    if name == "pixel":
        params = {}
    elif name == "patch":
        params = {"patch_size": getattr(args, "patch_size", 4)}
    elif name == "compressed_patch":
        params = {
            "patch_size": getattr(args, "patch_size", 4),
            "compressed_size": getattr(args, "compressed_size", 2),
        }
    elif name == "packet":
        params = {}
    elif name == "log_compressed_packet":
        params = {"segment": getattr(args, "log_compressed_segment", 64)}
    else:
        raise ValueError(
            f"Unsupported tokenizer type: {name}. "
            f"Available: {sorted(TOKENIZERS)}"
        )

    return InputSpec(name=name, params=params)


def infer_d_input(args) -> int:
    """模型输入维度。训练、评估共用，避免两边推导不一致。"""
    return make_input_spec(args).d_input
