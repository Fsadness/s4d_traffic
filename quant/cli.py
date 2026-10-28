"""cli.py —— 量化脚本共用的命令行参数与配置构造。

目的：让 PTQ / QAT / 消融脚本共享同一套“策略声明 → QuantConfig”的逻辑，
新增一种量化策略（例如按序列位置分段）时，只需要在 yaml 或命令行里描述，
不需要改脚本。

支持两种声明方式（可混用，命令行优先）：
    1. `--quant-config configs/quant/segment_8_4.yaml`
    2. 命令行：`--preset paper --bits-a 8 --segment-bits "0:800:8,800::4"`
"""

import os

import yaml

from quant.config import (
    QuantConfig,
    paper_config,
    parse_segment_bits,
    uniform_config,
)

BITS_FIELDS = (
    "bits_a",
    "bits_b",
    "bits_c",
    "bits_d",
    "bits_state",
    "bits_act",
    "bits_other",
)


def add_quant_args(parser):
    """注册量化相关参数（默认值 None 表示"不覆盖 yaml/预设"）。"""
    group = parser.add_argument_group("quantization")

    group.add_argument("--quant-config", type=str, default=None,
                       help="量化策略 yaml（可包含 segment_bits）")
    group.add_argument("--preset", type=str, default=None, choices=["paper", "uniform"],
                       help="预设：paper=W4A6/Ā8/x8；uniform=统一位宽")

    group.add_argument("--bits", type=int, default=None, help="uniform 预设的统一位宽")
    for field in BITS_FIELDS:
        group.add_argument(f"--{field.replace('_', '-')}", type=int, default=None)

    group.add_argument("--segment-bits", type=str, default=None,
                       help='按序列位置的位宽分段，例如 "0:800:8,800::4"；'
                            '也支持 "0:400:a8,400::s4" 分别指定激活/状态')

    group.add_argument("--symmetric", dest="symmetric", action="store_true", default=None)
    group.add_argument("--asymmetric", dest="symmetric", action="store_false")
    group.add_argument("--per-tensor", dest="per_head", action="store_false", default=None)
    group.add_argument("--per-head", dest="per_head", action="store_true")
    group.add_argument("--dynamic", dest="calibration", action="store_const",
                       const="dynamic", default=None)
    group.add_argument("--static", dest="calibration", action="store_const", const="static")
    group.add_argument("--gelu", dest="use_qgelu", action="store_false", default=None,
                       help="保留原始 GELU（默认按论文换 qGELU）")
    group.add_argument("--qgelu", dest="use_qgelu", action="store_true")
    group.add_argument("--param-mode", type=str, default=None,
                       choices=["continuous", "discrete", "frozen"])

    return parser


def apply_overrides(cfg: QuantConfig, args) -> QuantConfig:
    """把命令行显式给出的值覆盖到配置上。"""
    for field in BITS_FIELDS:
        value = getattr(args, field, None)
        if value is not None:
            setattr(cfg, field, value)

    if getattr(args, "param_mode", None) is not None:
        cfg.param_mode = args.param_mode
    if getattr(args, "symmetric", None) is not None:
        cfg.symmetric = bool(args.symmetric)
    if getattr(args, "per_head", None) is not None:
        cfg.per_head = bool(args.per_head)
    if getattr(args, "calibration", None) is not None:
        cfg.calibration = args.calibration
    if getattr(args, "use_qgelu", None) is not None:
        cfg.use_qgelu = bool(args.use_qgelu)
    if getattr(args, "grad_clip", None) is not None:
        cfg.grad_clip = float(args.grad_clip)

    segment_spec = getattr(args, "segment_bits", None)
    if segment_spec:
        cfg.segment_bits = parse_segment_bits(segment_spec)

    # 覆盖后重新校验（位宽范围 / 分段合法性）
    cfg.__post_init__()
    return cfg


def build_config(args, default_preset="paper", default_param_mode="continuous"):
    """根据 yaml / 预设 / 命令行构造 QuantConfig，返回 (label, cfg)。"""
    if getattr(args, "quant_config", None):
        with open(args.quant_config, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = QuantConfig.from_dict(data)
        label = os.path.splitext(os.path.basename(args.quant_config))[0]
        # 策略文件未显式写 param_mode 时，用脚本的默认值（PTQ=continuous，QAT=discrete）
        if "param_mode" not in data:
            cfg.param_mode = default_param_mode
    elif getattr(args, "preset", None) == "uniform" or (
        getattr(args, "preset", None) is None and getattr(args, "bits", None) is not None
    ):
        bits = args.bits or 8
        cfg = uniform_config(bits, param_mode=default_param_mode)
        label = f"uniform-{bits}bit"
    else:
        cfg = paper_config(param_mode=default_param_mode)
        label = "paper-W4A6-A8x8"

    cfg = apply_overrides(cfg, args)

    if cfg.segment_bits and not label.endswith("+seg"):
        label += "+seg"

    return label, cfg
