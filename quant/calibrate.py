"""calibrate.py —— 静态校准（PTQ 的“收集范围”阶段）。

论文用 99.999% 百分位校准 SSM 参数；对运行时张量（状态 x、激活）则采用
静态校准：用一小批数据收集每个 head 的 min/max，部署时固定 scale/zero_point。

流程：
    1. 关闭量化（enabled=False），但打开观察（observing=True）
    2. 前向若干 batch，激活/状态量化器累计 min/max
    3. freeze()：把统计量固化成 scale/zero_point
    4. 重新打开量化
"""

import torch

from .fake_quant import UniformQuantizer
from .modules import set_observing, set_quant_enabled


@torch.no_grad()
def calibrate_model(model, loader, device: str = "cuda", max_batches: int = 8,
                    verbose: bool = True) -> int:
    """对模型做静态校准，返回实际使用的 batch 数。

    注意：
        - 校准必须在 eval 模式下进行（关闭 dropout），否则统计量会被噪声污染；
        - 权重量化器在 build_quant_model 里已校准，这里只补激活/状态；
        - dynamic 模式下 freeze() 只是标记，不做任何事。
    """
    was_training = model.training
    model.eval()

    # 1) 关闭量化，打开观察
    model.set_quantization(False)
    model.set_observing(True)

    used = 0
    for inputs, _ in loader:
        model(inputs.to(device, non_blocking=True))
        used += 1
        if used >= max_batches:
            break

    # 2) 固化统计量
    model.set_observing(False)
    for q in model.quantizers():
        q.freeze()

    # 3) 恢复量化
    model.set_quantization(True)
    if was_training:
        model.train()

    if verbose:
        print(f"[calibrate] 使用 {used} 个 batch 完成静态校准")

    return used


def describe_calibration(model) -> dict:
    """统计每个量化器是否已校准（调试用）。"""
    info = {}
    for name, q in model.named_modules():
        if isinstance(q, UniformQuantizer):
            info[name or "root"] = {
                "bits": q.bits,
                "calibrated": bool(q.calibrated),
                "dynamic": bool(q.dynamic),
                "scale_shape": tuple(q.scale.shape),
            }
    return info
