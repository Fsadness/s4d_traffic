"""modules.py —— 带权重量化的基础模块。

仓库里的 S4D 模型由四类非 SSM 模块组成：
    backbone.encoder   : nn.Linear(d_input, d_model)
    backbone.norms.i   : nn.LayerNorm(d_model)
    s4_layers.i.output_linear.0 : nn.Conv1d(d_model, 2*d_model, 1)
    decoder            : nn.Linear(d_model, d_output)

为保持 **state_dict 键名与仓库完全一致**（便于直接 load 原始 best.pth），
这里不用“包装原模块”的做法，而是重新声明同名的 weight/bias，
并用相同的初始化方式；随后由 build_quant_model 从原 state_dict 覆盖。

论文设定：SSM 之外的权重统一 4-bit（bits_other）；偏置不量化（占比极小）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fake_quant import UniformQuantizer, make_quantizer


#############################
# 工具
#############################
def _init_linear_weight(weight: torch.Tensor, bias: torch.Tensor = None):
    """与 nn.Linear.reset_parameters() 一致的初始化（保证未加载权重时也可用）。"""
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        bound = 1.0 / math.sqrt(weight.shape[1]) if weight.dim() > 1 else 0.0
        nn.init.uniform_(bias, -bound, bound)


def _init_conv_weight(weight: torch.Tensor, bias: torch.Tensor = None):
    """与 nn.Conv1d.reset_parameters() 一致的初始化。"""
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in = weight.shape[1] * weight.shape[2]
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)


#############################
# 全连接
#############################
class QuantLinear(nn.Module):
    """权重量化（+ 可选输入激活量化）的全连接层。

    键名与 nn.Linear 相同：weight (out, in)、bias (out,)。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        cfg=None,
        act_quantizer: UniformQuantizer = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        _init_linear_weight(self.weight, self.bias)

        # 权重量化器：非 SSM 权重按 per-tensor 量化（论文的 per-head 只针对 SSM）
        self.weight_quantizer = make_quantizer(
            cfg.bits_other, cfg, per_head=False, name="linear.weight"
        )
        # 输入激活量化器（由模型构建时注入，可为 None）
        self.act_quantizer = act_quantizer

    def calibrate(self):
        """用当前权重做一次性校准（百分位裁剪）。"""
        self.weight_quantizer.calibrate(self.weight.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_quantizer is not None:
            x = self.act_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}"


#############################
# 一维卷积（只用于 kernel_size=1 的输出混合）
#############################
class QuantConv1d(nn.Module):
    """权重量化的 Conv1d（仓库里只用 kernel_size=1 的逐点卷积）。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        bias: bool = True,
        cfg=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        _init_conv_weight(self.weight, self.bias)

        self.weight_quantizer = make_quantizer(
            cfg.bits_other, cfg, per_head=False, name="conv1d.weight"
        )

    def calibrate(self):
        self.weight_quantizer.calibrate(self.weight.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_quantizer(self.weight)
        return F.conv1d(x, w, self.bias, stride=1, padding=0, dilation=1, groups=1)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"k={self.kernel_size}, bias={self.bias is not None}"
        )


#############################
# LayerNorm
#############################
class QuantLayerNorm(nn.Module):
    """权重量化的 LayerNorm（论文把非 SSM 参数统一压到 4-bit）。

    LayerNorm 的 weight/bias 仍保留为 float 参数，forward 时对 weight 做伪量化，
    因此 QAT 时梯度可以经 STE 回传到 weight。
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, cfg=None):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        self.weight_quantizer = make_quantizer(
            cfg.bits_other, cfg, per_head=False, name="layernorm.weight"
        )

    def calibrate(self):
        self.weight_quantizer.calibrate(self.weight.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_quantizer(self.weight)
        return F.layer_norm(x, self.normalized_shape, w, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"shape={self.normalized_shape}, eps={self.eps}"


#############################
# 辅助
#############################
def set_quant_enabled(module: nn.Module, enabled: bool):
    """统一开关所有量化器（校准阶段关闭，QAT/推理阶段打开）。"""
    for m in module.modules():
        if isinstance(m, UniformQuantizer):
            m.enabled = enabled


def set_observing(module: nn.Module, observing: bool, only_uncalibrated: bool = True):
    """统一设置观察开关。

    权重量化器在构建时已经 calibrated，默认不再观察；
    只让激活/状态量化器在数据流经时统计范围。
    """
    for m in module.modules():
        if not isinstance(m, UniformQuantizer):
            continue
        if only_uncalibrated and m.calibrated:
            m.observing = False
        else:
            m.observing = observing
