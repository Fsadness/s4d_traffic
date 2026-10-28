#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_smoke.py —— 量化实现的冒烟测试。

分两层：
    单元测试：量化公式、per-head 粒度、STE、复数量化、qGELU、状态裁剪
    集成测试：加载真实 best.pth → 递推 FP 基线 → 静态校准 → PTQ 评估 → QAT 一步 → 显存估算

运行：
    # 全部
    python tests/test_smoke.py

    # 只跑单元测试
    python tests/test_smoke.py --unit-only

    # 换实验 / 减少样本
    python tests/test_smoke.py --ckpt-dir <实验目录> --max-samples 32 --skip-qat
"""

import argparse
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quant import QuantConfig, paper_config                     # noqa: E402
from quant.config import parse_segment_bits                     # noqa: E402
from quant.calibrate import calibrate_model                    # noqa: E402
from quant.data import load_experiment_config, make_loader     # noqa: E402
from quant.evaluate import evaluate, evaluate_quantization_off  # noqa: E402
from quant.fake_quant import (                                 # noqa: E402
    UniformQuantizer,
    clip_state,
    qgelu,
)
from quant.memory import estimate_memory                       # noqa: E402
from quant.model import build_quant_model                      # noqa: E402
from quant.qat import train_qat                                # noqa: E402


DEFAULT_CKPT_DIR = os.path.join(
    ROOT, "experiments", "new", "s4d_classification", "20261028_110942"
)


#############################
# 单元测试
#############################
def test_symmetric_quantizer():
    """对称量化：误差不超过半个 scale，且量化值落在均匀网格上。"""
    q = UniformQuantizer(bits=4, symmetric=True)
    x = torch.linspace(-1.0, 1.0, 1001)
    q.calibrate(x)
    y = q(x)

    qmax = 2 ** (4 - 1) - 1
    scale = 1.0 / qmax
    err = (y - x).abs().max().item()
    assert err <= scale / 2 + 1e-6, f"对称量化误差过大: {err} > {scale / 2}"
    assert torch.allclose(y, torch.round(y / scale) * scale, atol=1e-5)
    assert y.abs().max() <= 1.0 + 1e-6
    print(f"  [ok] 对称量化 4-bit: max_err={err:.4f}")


def test_asymmetric_quantizer():
    """非对称量化：应精确覆盖 [min, max] 两端点。"""
    q = UniformQuantizer(bits=4, symmetric=False)
    x = torch.linspace(0.0, 10.0, 1001)
    q.calibrate(x)
    y = q(x)

    span = 10.0
    step = span / (2 ** 4 - 1)
    err = (y - x).abs().max().item()
    assert err <= step / 2 + 1e-4, f"非对称量化误差过大: {err} > {step / 2}"
    assert abs(y.min().item() - 0.0) < step
    assert abs(y.max().item() - 10.0) < step
    print(f"  [ok] 非对称量化 4-bit: max_err={err:.4f}, range=[{y.min():.2f},{y.max():.2f}]")


def test_per_head_quantizer():
    """per-head：每个 head 有独立 scale，量程大的 head 误差也按比例放大。"""
    q = UniformQuantizer(bits=8, per_head=True, channel_axis=0, symmetric=False)
    x = torch.stack([
        torch.linspace(-1.0, 1.0, 100),
        torch.linspace(-10.0, 10.0, 100),
    ])
    q.calibrate(x)
    y = q(x)

    assert q.scale.numel() == 2, "per-head 应得到 2 个 scale"
    assert q.scale[1] > 5 * q.scale[0], "量程更大的 head 应有更大的 scale"
    for h in range(2):
        err = (y[h] - x[h]).abs().max().item()
        assert err <= q.scale[h].item() * 0.51, f"head {h} 误差过大"
    print(f"  [ok] per-head: scales={q.scale.tolist()}")


def test_ste_gradient():
    """STE：伪量化层应把梯度原样传回（dQ/dx = 1）。"""
    q = UniformQuantizer(bits=4, symmetric=True)
    q.calibrate(torch.tensor([-2.0, 2.0]))

    x = torch.tensor([0.3, -1.1, 1.9], requires_grad=True)
    y = q(x).sum()
    y.backward()

    assert torch.allclose(x.grad, torch.ones_like(x)), f"STE 梯度错误: {x.grad}"
    print("  [ok] STE 梯度直通")


def test_complex_quantizer():
    """复数量化：实部虚部共用同一 scale（论文要求），且输出仍是复数。"""
    torch.manual_seed(0)
    q = UniformQuantizer(bits=8, per_head=True, channel_axis=0)
    z = torch.complex(torch.randn(4, 8), torch.randn(4, 8))
    q.calibrate_complex(z)
    out = q.forward_complex(z)

    assert out.is_complex()
    assert q.scale.numel() == 4
    # 非对称量化保证结果落在输入范围内（实部/虚部共享同一量程）
    in_bound = max(z.real.abs().max().item(), z.imag.abs().max().item())
    assert out.real.abs().max().item() <= in_bound * 1.01
    assert out.imag.abs().max().item() <= in_bound * 1.01
    print(f"  [ok] 复数量化: scale={q.scale.tolist()[:2]}...")


def test_qgelu_and_clip():
    """qGELU 与状态裁剪的数值行为。"""
    x = torch.tensor([-3.0, -1.0, 0.0, 0.5, 2.0, 3.0])
    y = qgelu(x)
    # 分段行为：x<=-2 → 0；-2<x<-1 → x(x+2)；x>=-1 → x
    expect = torch.tensor([0.0, -1.0, 0.0, 0.5, 2.0, 3.0])
    assert torch.allclose(y, expect), f"qGELU 结果错误: {y}"

    z = torch.complex(torch.tensor([-100.0, 10.0]), torch.tensor([50.0, 100.0]))
    zc = clip_state(z, 50.0)
    assert zc.real.tolist() == [-50.0, 10.0]
    assert zc.imag.tolist() == [50.0, 50.0]
    print("  [ok] qGELU / 状态裁剪")


def test_segment_bits_config():
    """按序列位置的分段位宽：解析、查表、区间展开、序列化。"""
    spec = "0:800:8,800::4"
    segments = parse_segment_bits(spec)
    cfg = QuantConfig(segment_bits=segments)

    assert cfg.bits_at(0) == (8, 8)
    assert cfg.bits_at(799) == (8, 8)
    assert cfg.bits_at(800) == (4, 4)
    assert cfg.bits_at(1599) == (4, 4)

    pieces = cfg.segments_for_length(1600)
    assert pieces == [(0, 800, 8, 8), (800, 1600, 4, 4)], pieces

    # 只覆盖激活：状态位宽保持默认
    cfg_act = QuantConfig(bits_act=6, bits_state=8,
                          segment_bits=parse_segment_bits("0:400:a4"))
    assert cfg_act.bits_at(0) == (4, 8)
    assert cfg_act.bits_at(500) == (6, 8)

    # 序列化往返（checkpoint / json）
    restored = QuantConfig.from_dict(cfg.to_dict())
    assert restored.segment_bits == segments

    print(f"  [ok] 分段位宽: {pieces}")


#############################
# 集成测试
#############################
def test_integration(args):
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"\n[集成] 实验目录: {args.ckpt_dir}")
    cfg = load_experiment_config(args.ckpt_dir)
    ckpt = torch.load(os.path.join(args.ckpt_dir, "best.pth"), map_location="cpu")
    fp_state_dict = ckpt["model"] if "model" in ckpt else ckpt

    qcfg = paper_config(param_mode="continuous")

    # 1) 构建 + FP 递推基线
    model = build_quant_model(fp_state_dict, qcfg=QuantConfig(use_qgelu=False), verbose=False)
    model = model.to(device)

    test_loader = make_loader(
        cfg, split="test", data_dir=args.data_dir,
        batch_size=args.batch_size, num_workers=2,
        max_samples=args.max_samples, seed=0,
    )
    fp = evaluate_quantization_off(model, test_loader, device=device, max_batches=2)
    print(f"  [ok] FP32 递推基线: acc={fp['acc']:.2f}% (n={fp['n_samples']})")
    assert fp["acc"] > 80.0, "递推 FP 基线精度异常，先排查权重加载/递推实现"

    # 2) 换成量化模型并校准（校准用 valid，避免用测试集）
    model = build_quant_model(fp_state_dict, qcfg=qcfg, verbose=False).to(device)
    calib_loader = make_loader(
        cfg, split="valid", data_dir=args.data_dir,
        batch_size=args.batch_size, num_workers=2,
        max_samples=args.batch_size * 2, seed=1,
    )
    calibrate_model(model, calib_loader, device=device, max_batches=2, verbose=False)

    # 3) PTQ 评估
    ptq = evaluate(model, test_loader, device=device, max_batches=2)
    mem = estimate_memory(model, qcfg)
    print(f"  [ok] PTQ {qcfg.describe()}: acc={ptq['acc']:.2f}%")
    print(f"  [ok] 权重显存: {mem['weights']['fp32_mib']:.2f} → "
          f"{mem['weights']['quant_mib']:.2f} MiB (-{mem['weights']['savings_pct']:.1f}%)")
    assert mem["weights"]["savings_pct"] > 50.0, "异构量化的显存节省应超过 50%"

    # 4) QAT 一步（只验证梯度通路与伪量化训练不 NaN）
    if not args.skip_qat:
        train_loader = make_loader(
            cfg, split="valid", data_dir=args.data_dir,
            batch_size=args.batch_size, num_workers=2,
            max_samples=args.batch_size, shuffle=True, seed=2,
        )
        history = train_qat(
            model, train_loader, device=device,
            epochs=1, lr=1e-4, grad_clip=1000.0, max_batches_per_epoch=1,
        )
        loss = history[0]["train_loss"]
        assert math.isfinite(loss), "QAT 出现 NaN/Inf"
        print(f"  [ok] QAT 1 batch: loss={loss:.4f}")

        after = evaluate(model, test_loader, device=device, max_batches=1)
        print(f"  [ok] QAT 后复评: acc={after['acc']:.2f}%")

    print("\n[集成] 全部通过")


#############################
# 入口
#############################
def main():
    parser = argparse.ArgumentParser(description="S4D 量化冒烟测试")
    parser.add_argument("--ckpt-dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--skip-qat", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("单元测试")
    print("=" * 70)
    test_symmetric_quantizer()
    test_asymmetric_quantizer()
    test_per_head_quantizer()
    test_ste_gradient()
    test_complex_quantizer()
    test_qgelu_and_clip()
    test_segment_bits_config()

    if args.unit_only:
        print("\n单元测试全部通过")
        return

    print("\n" + "=" * 70)
    print("集成测试")
    print("=" * 70)
    test_integration(args)


if __name__ == "__main__":
    main()
