"""config.py —— 量化超参数。

论文（Quantizing Small-Scale SSMs for Edge AI）用到的全部量化自由度：

    1. 位宽：Ā / B̄ / C / D / 状态 x / 其他激活 / 非 SSM 权重 可以各不相同（异构量化）
    2. 粒度：per-tensor 或 per-head（SSM 的 H 维）
    3. 对称性：symmetric / asymmetric
    4. 校准：static（部署前固定 scale）或 dynamic（每次前向重算）
    5. QAT 参数化：continuous（原始 S4D 连续参数）/ discrete（直接训练离散矩阵）/ frozen（冻结 Ā）
    6. **按序列位置的分段位宽**（`segment_bits`）：例如前 800 个 token 用 8-bit、
       后 800 个 token 用 4-bit；只影响运行时量（状态 x 与激活），不影响按参数分组的位宽。

默认为论文推荐的“非对称 + per-head + 静态”组合（论文 II 节）。
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import yaml


#############################
# 分段位宽
#############################
SEGMENT_KEYS = {"start", "end", "bits_act", "bits_state"}


def parse_segment_bits(spec: str) -> List[Dict[str, Any]]:
    """把命令行简写解析成 `segment_bits` 列表。

    语法：`start:end:bits`，逗号分隔多段；end 留空表示到序列结尾。
    bits 可以是：
        `8`        → 状态与激活都用 8-bit
        `a8`       → 只覆盖激活位宽
        `s8`       → 只覆盖状态位宽
        `a8+s8`    → 分别指定

    例：
        `0:800:8,800::4`       前 800 步 8-bit，其余 4-bit
        `0:400:a8,400::a4`     只改激活位宽
    """
    segments: List[Dict[str, Any]] = []

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        fields = part.split(":")
        if len(fields) != 3:
            raise ValueError(
                f"非法分段: {part!r}，应为 start:end:bits，例如 0:800:8"
            )

        start_s, end_s, bits_s = (item.strip() for item in fields)

        segment: Dict[str, Any] = {"start": int(start_s)}
        segment["end"] = int(end_s) if end_s else None

        for token in bits_s.split("+"):
            token = token.strip()
            if not token:
                continue
            if token.startswith("a"):
                segment["bits_act"] = int(token[1:])
            elif token.startswith("s"):
                segment["bits_state"] = int(token[1:])
            else:
                bits = int(token)
                segment["bits_act"] = bits
                segment["bits_state"] = bits

        segments.append(segment)

    return segments


#############################
# 量化配置
#############################
@dataclass
class QuantConfig:
    """一次量化实验的完整配置。

    位宽字段：
        bits_a      : 离散状态转移矩阵 Ā（论文指出最敏感，PTQ 下限 8-bit）
        bits_b      : 输入矩阵 B̄
        bits_c      : 输出矩阵 C
        bits_d      : 直通项 D
        bits_state  : 递推状态 x 的默认位宽（与 Ā 同样敏感）
        bits_act    : SSM 之外的激活的默认位宽（qGELU / Conv1D / GLU / 层输出）
        bits_other  : 非 SSM 权重（encoder / decoder / LayerNorm / Conv1D 权重）

    粒度与校准：
        symmetric   : True 时用对称量化（zero_point 固定为 0）
        per_head    : True 时沿 SSM 的 H 维逐头量化（论文称 per-head）
        percentile  : 权重量化时的百分位裁剪，论文用 99.999（去掉极端离群点）
        calibration : "static" | "dynamic"

    序列位置分段（可选）：
        segment_bits: [{"start": 0, "end": 800, "bits_act": 8, "bits_state": 8},
                       {"start": 800, "end": None, "bits_act": 4, "bits_state": 4}]
        - `start` 含、`end` 不含，`end=None` 表示到序列结尾；
        - 只覆盖运行时量（状态 / 激活），按参数分组的位宽仍由 bits_a…bits_other 决定；
        - 多段重叠时**后出现的覆盖先出现的**（便于“默认 + 例外”写法）；
        - 未覆盖的位置使用 bits_state / bits_act 默认值。

    QAT：
        param_mode  : "continuous" | "discrete" | "frozen"
        state_clip  : 状态实部/虚部的裁剪上界（论文取 50）
        grad_clip   : QAT 梯度逐元素裁剪上界（论文取 1000）
    """

    # ---- 位宽 ----
    bits_a: int = 8
    bits_b: int = 4
    bits_c: int = 4
    bits_d: int = 4
    bits_state: int = 8
    bits_act: int = 6
    bits_other: int = 4

    # ---- 粒度 / 对称性 / 校准 ----
    symmetric: bool = False
    per_head: bool = True
    percentile: float = 99.999
    calibration: str = "static"

    # ---- QAT ----
    param_mode: str = "continuous"
    state_clip: float = 50.0
    grad_clip: float = 1000.0

    # ---- 开关（消融用）----
    quantize_state: bool = True
    quantize_activations: bool = True
    quantize_weights: bool = True
    use_qgelu: bool = True

    # ---- 按序列位置的分段位宽（可为空）----
    segment_bits: List[Dict[str, Any]] = field(default_factory=list)

    # ---- 备注 ----
    tag: str = ""

    def __post_init__(self):
        assert self.calibration in ("static", "dynamic"), self.calibration
        assert self.param_mode in ("continuous", "discrete", "frozen"), self.param_mode
        for name in (
            "bits_a", "bits_b", "bits_c", "bits_d",
            "bits_state", "bits_act", "bits_other",
        ):
            bits = getattr(self, name)
            assert 2 <= bits <= 16, f"{name}={bits} 超出 [2, 16]"

        self._validate_segments()

    def _validate_segments(self):
        if not isinstance(self.segment_bits, list):
            raise TypeError("segment_bits 必须是 list")

        for index, segment in enumerate(self.segment_bits):
            if not isinstance(segment, dict):
                raise TypeError(f"segment_bits[{index}] 必须是 dict")

            unknown = set(segment) - SEGMENT_KEYS
            if unknown:
                raise ValueError(f"segment_bits[{index}] 含未知字段: {sorted(unknown)}")

            start = int(segment.get("start", 0))
            if start < 0:
                raise ValueError(f"segment_bits[{index}].start 必须 >= 0")

            end = segment.get("end")
            if end is not None and int(end) <= start:
                raise ValueError(
                    f"segment_bits[{index}] 的 end 必须大于 start: {start} >= {end}"
                )

            for key in ("bits_act", "bits_state"):
                if segment.get(key) is not None:
                    bits = int(segment[key])
                    if not 2 <= bits <= 16:
                        raise ValueError(
                            f"segment_bits[{index}].{key}={bits} 超出 [2, 16]"
                        )

    #########################
    # 分段解析
    #########################
    @property
    def has_segments(self) -> bool:
        return bool(self.segment_bits)

    def bits_at(self, step: int) -> Tuple[int, int]:
        """返回第 `step` 个 token 的 (激活位宽, 状态位宽)。"""
        act_bits, state_bits = self.bits_act, self.bits_state

        for segment in self.segment_bits:
            start = int(segment.get("start", 0))
            end = segment.get("end")
            if step < start or (end is not None and step >= int(end)):
                continue
            if segment.get("bits_act") is not None:
                act_bits = int(segment["bits_act"])
            if segment.get("bits_state") is not None:
                state_bits = int(segment["bits_state"])

        return act_bits, state_bits

    def segments_for_length(self, length: int) -> List[Tuple[int, int, int, int]]:
        """把分段展开成覆盖 [0, length) 的互不重叠区间。

        返回 [(start, end, bits_act, bits_state), ...]，相邻且位宽相同的区间会合并。
        """
        if length <= 0:
            return []

        if not self.segment_bits:
            return [(0, length, self.bits_act, self.bits_state)]

        boundaries = {0, int(length)}
        for segment in self.segment_bits:
            start = max(0, min(int(length), int(segment.get("start", 0))))
            end = segment.get("end")
            boundaries.add(start)
            boundaries.add(int(length) if end is None else max(0, min(int(length), int(end))))

        points = sorted(boundaries)
        pieces: List[Tuple[int, int, int, int]] = []

        for start, end in zip(points, points[1:]):
            if end <= start:
                continue
            act_bits, state_bits = self.bits_at(start)
            if pieces and pieces[-1][1] == start and pieces[-1][2:] == (act_bits, state_bits):
                prev = pieces[-1]
                pieces[-1] = (prev[0], end, act_bits, state_bits)
            else:
                pieces.append((start, end, act_bits, state_bits))

        return pieces

    #########################
    # 序列化
    #########################
    def to_dict(self) -> Dict[str, Any]:
        """转成可写入 checkpoint / json 的字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QuantConfig":
        """从字典恢复，忽略未知字段（向前兼容）。"""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str) -> "QuantConfig":
        """从 yaml 文件读取量化策略（可含 segment_bits）。"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def describe(self) -> str:
        """单行摘要，用于日志 / 结果表。"""
        text = (
            f"A{self.bits_a} B{self.bits_b} C{self.bits_c} D{self.bits_d} "
            f"x{self.bits_state} act{self.bits_act} W{self.bits_other} | "
            f"{'sym' if self.symmetric else 'asym'} "
            f"{'per-head' if self.per_head else 'per-tensor'} "
            f"{self.calibration} | {self.param_mode}"
        )
        if self.segment_bits:
            segments = ",".join(
                f"[{seg.get('start', 0)}:"
                f"{seg.get('end', '-')}]"
                f"(a{seg.get('bits_act', '-')}/s{seg.get('bits_state', '-')})"
                for seg in self.segment_bits
            )
            text += f" | seg {segments}"
        return text


#############################
# 常用预设
#############################
def uniform_config(bits: int = 8, **kwargs) -> QuantConfig:
    """所有权重/激活统一位宽（论文 Fig.2(c)(d) 的“同精度”对比）。"""
    return QuantConfig(
        bits_a=bits, bits_b=bits, bits_c=bits, bits_d=bits,
        bits_state=bits, bits_act=bits, bits_other=bits,
        **kwargs,
    )


def paper_config(param_mode: str = "discrete", **kwargs) -> QuantConfig:
    """论文 Table III 的异构方案：W4A6，Ā 与 x 用 8-bit。

    - SSM 之外权重 4-bit
    - B̄ / C / D 4-bit
    - 除状态外的激活 6-bit
    - Ā 与状态 x 8-bit
    """
    return QuantConfig(
        bits_a=8, bits_b=4, bits_c=4, bits_d=4,
        bits_state=8, bits_act=6, bits_other=4,
        param_mode=param_mode,
        **kwargs,
    )
