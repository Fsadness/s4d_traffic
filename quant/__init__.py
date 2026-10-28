"""S4D 量化工具包（论文：Quantizing Small-Scale State-Space Models for Edge AI, arXiv:2506.12480）。

已合并进主仓：
    - 数据与配置复用 `datasets.dataloader` / `engine.config`；
    - 评估指标复用 `engine.metrics`；
    - checkpoint 键名与 `models/networks/s4d_classification` 完全一致（可互相加载）。

量化策略由 `QuantConfig` 描述，支持两类“混合精度”：
    1. 按参数分组：Ā / B̄ / C / D / 状态 / 激活 / 非 SSM 权重各自位宽；
    2. 按序列位置分段：`segment_bits=[{start:0,end:800,bits_act:8,...}, ...]`。

目录结构：
    quant/
        fake_quant.py   均匀伪量化器（STE）、复数量化、qGELU、状态裁剪
        config.py       QuantConfig：位宽 / 粒度 / 对称性 / 校准 / 参数化 / 分段位宽
        modules.py      QuantLinear / QuantConv1d / QuantLayerNorm（权重量化 + 激活量化）
        recurrent.py    递推模式 S4D 核（量化 Ā、B̄、C、D、状态 x，支持分段位宽）
        model.py        QuantS4DClassification：与仓库 s4d_classification 键名对齐
        data.py         复用主仓数据管线，构建 DataLoader
        calibrate.py    静态校准（收集激活/状态范围）
        evaluate.py     精度、延迟、吞吐（复用 engine.metrics）
        qat.py          QAT 微调（STE + 梯度裁剪 + 三种参数化）
        memory.py       量化后权重/状态显存估算（考虑分段后的平均状态位宽）
        cli.py          脚本共用的量化参数与策略构造
        repo.py         定位仓库根目录
"""

from .config import QuantConfig, paper_config, parse_segment_bits, uniform_config
from .fake_quant import (
    UniformQuantizer,
    fake_quant_complex,
    clip_state,
    qgelu,
    QGELU,
)
from .model import QuantS4DClassification, build_quant_model

__all__ = [
    "QuantConfig",
    "uniform_config",
    "paper_config",
    "parse_segment_bits",
    "UniformQuantizer",
    "fake_quant_complex",
    "clip_state",
    "qgelu",
    "QGELU",
    "QuantS4DClassification",
    "build_quant_model",
]
