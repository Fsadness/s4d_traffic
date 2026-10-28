"""fake_quant.py —— 均匀伪量化（fake quantization）工具。

对应论文 II 节与第 IV 节：
    - 量化网格始终取“有符号整数”可表示的范围 [-(2^(b-1)), 2^(b-1)-1]
    - 范围校准默认用 99.999% 百分位裁剪离群点
    - 粒度支持 per-tensor 与 per-head（沿 SSM 的 H 维）
    - 复数张量（Ā / B̄ / C / 状态 x）实部虚部分离量化，但**共用同一套 scale/zero_point**
    - 反向传播用 STE（straight-through estimator）
    - 状态 x 在递推中每一步裁剪到 [-50, +50]（论文 IV 节，防止量化误差随 n 放大而发散）
    - qGELU(x) = x · min(max(0, x + 2), 1)

约定：
    enabled   = False 时 forward 直接返回原值（校准阶段用）
    observing = True  时顺便累计 min/max（静态校准用）
"""

import torch
import torch.nn as nn


#############################
# 基础函数
#############################
def qgelu(x: torch.Tensor) -> torch.Tensor:
    """量化友好的 GELU 近似（论文引用 Q-S5 的做法）。

    数学形式：qGELU(x) = x · min(max(0, x + 2), 1)
    值域与 GELU 接近，但只用到比较、加法和乘法，便于整数硬件实现。
    """
    return x * torch.clamp(torch.clamp(x + 2, min=0), max=1)


def clip_state(x: torch.Tensor, clip_value: float = 50.0) -> torch.Tensor:
    """对状态 x 的实部/虚部分别裁剪（论文 IV 节的“有界状态”技巧）。

    量化后的 Ā 可能落到单位圆外，导致 x 随时间步指数发散；
    每步把 x 限制在 [-clip_value, +clip_value] 是最简单且鲁棒的稳定化手段。
    """
    if x.is_complex():
        return torch.complex(
            x.real.clamp(-clip_value, clip_value),
            x.imag.clamp(-clip_value, clip_value),
        )
    return x.clamp(-clip_value, clip_value)


class QGELU(nn.Module):
    """qGELU 的模块封装，便于替换仓库里的 nn.GELU()。"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return qgelu(x)

    def extra_repr(self) -> str:
        return "x * min(max(0, x + 2), 1)"


#############################
# 均匀量化器
#############################
class UniformQuantizer(nn.Module):
    """均匀伪量化器：对称/非对称、per-tensor/per-head、静态/动态。

    量化公式（非对称，有符号网格）：
        qmin = -(2^(b-1)),  qmax = 2^(b-1) - 1
        scale = (max - min) / (qmax - qmin)
        zero_point = qmin - round(min / scale)
        Q(x) = clamp(round(x / scale) + zero_point, qmin, qmax)
        dequant = (Q(x) - zero_point) * scale

    对称时 zero_point = 0，scale = max(|x|) / qmax。

    前向输出伪量化值（float，模拟整数存储）；
    反向用 STE：dQ/dx = 1，因此 QAT 可以正常反传。
    """

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = False,
        per_head: bool = False,
        channel_axis: int = 0,
        percentile: float = 99.999,
        dynamic: bool = False,
        eps: float = 1e-8,
        name: str = "",
    ):
        super().__init__()
        assert 2 <= bits <= 16
        self.bits = bits
        self.symmetric = symmetric
        self.per_head = per_head
        self.channel_axis = channel_axis
        self.percentile = percentile
        self.dynamic = dynamic
        self.eps = eps
        self.name = name

        # 量化参数（buffer，会随 .to(device) 移动；persistent=True 便于保存进 checkpoint）
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer("zero_point", torch.zeros(()))
        # 校准统计量
        self.register_buffer("obs_min", None)
        self.register_buffer("obs_max", None)

        # 运行开关
        self.calibrated = False
        self.enabled = True
        self.observing = False
        self.bypass = False        # True 时永久跳过量化（消融实验用，如纯 FP 权重）

    #########################
    # 属性
    #########################
    @property
    def qmin(self) -> int:
        return -(2 ** (self.bits - 1))

    @property
    def qmax(self) -> int:
        return 2 ** (self.bits - 1) - 1

    def extra_repr(self) -> str:
        return (
            f"bits={self.bits}, sym={self.symmetric}, per_head={self.per_head}, "
            f"axis={self.channel_axis}, {self.calibration_mode()}"
        )

    def calibration_mode(self) -> str:
        return "dynamic" if self.dynamic else "static"

    #########################
    # 范围统计
    #########################
    def _channel_flat(self, x: torch.Tensor):
        """把 channel 维移到第 0 维并展平其余维度。

        返回:
            flat: (C, K)
            axis: 归一化后的 channel 轴
        """
        axis = self.channel_axis % x.dim()
        xm = x.movedim(axis, 0)
        flat = xm.reshape(xm.shape[0], -1)
        return flat, axis

    def _subsample(self, flat: torch.Tensor, limit: int = 1_000_000) -> torch.Tensor:
        """百分位计算前做等距抽样，避免超大张量触发 quantile 的尺寸限制。"""
        if flat.numel() <= limit:
            return flat
        step = max(1, flat.numel() // limit)
        return flat.reshape(-1)[::step].reshape(1, -1) if flat.shape[0] == 1 else flat[:, ::step]

    def _ranges(self, x: torch.Tensor, use_percentile: bool = False):
        """计算逐通道（或全局）的 min/max。

        返回 (min, max, axis)，其中 min/max 形状为 (C,) 或标量。
        """
        flat, axis = self._channel_flat(x) if (self.per_head and x.dim() > 1) else (x.reshape(1, -1), 0)

        if use_percentile and self.percentile < 100.0:
            low = (100.0 - self.percentile) / 200.0
            high = 1.0 - low
            flat_s = self._subsample(flat)
            mn = torch.quantile(flat_s, low, dim=1)
            mx = torch.quantile(flat_s, high, dim=1)
        else:
            mn = flat.amin(dim=1)
            mx = flat.amax(dim=1)

        # per-tensor：压成标量，便于广播
        if flat.shape[0] == 1 and not (self.per_head and x.dim() > 1):
            mn, mx = mn.reshape(()), mx.reshape(())
        return mn, mx, axis

    def _compute_params(self, mn: torch.Tensor, mx: torch.Tensor):
        """由 min/max 计算 scale 与 zero_point（有符号网格）。"""
        if self.symmetric:
            amax = torch.maximum(mn.abs(), mx.abs()).clamp(min=self.eps)
            scale = amax / float(self.qmax)
            zero_point = torch.zeros_like(scale)
        else:
            span = (mx - mn).clamp(min=self.eps)
            scale = span / float(self.qmax - self.qmin)
            zero_point = torch.round(self.qmin - mn / scale)
        return scale.clamp(min=self.eps).detach(), zero_point.detach()

    @torch.no_grad()
    def observe(self, x: torch.Tensor):
        """静态校准：累计 min/max（不做百分位，便于流式处理）。"""
        if self.dynamic:
            return
        mn, mx, _ = self._ranges(x, use_percentile=False)
        if self.obs_min is None:
            self.obs_min, self.obs_max = mn.clone(), mx.clone()
        else:
            self.obs_min = torch.minimum(self.obs_min, mn)
            self.obs_max = torch.maximum(self.obs_max, mx)

    @torch.no_grad()
    def freeze(self):
        """用累计到的范围固定 scale/zero_point（静态校准的收尾）。"""
        if self.dynamic:
            self.calibrated = True
            return
        if self.obs_min is None:
            return
        self.scale, self.zero_point = self._compute_params(self.obs_min, self.obs_max)
        self.calibrated = True

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor, use_percentile: bool = True):
        """一次性校准（权重/参数直接调用，论文用 99.999% 百分位）。"""
        mn, mx, _ = self._ranges(x, use_percentile=use_percentile)
        self.scale, self.zero_point = self._compute_params(mn, mx)
        self.calibrated = True

    #########################
    # 量化
    #########################
    def _broadcast(self, scale: torch.Tensor, zero_point: torch.Tensor, like: torch.Tensor):
        """把 (C,) 形状的量化参数 reshape 成可广播的形状。"""
        if scale.numel() == 1:
            return scale, zero_point
        axis = self.channel_axis % like.dim()
        shape = [1] * like.dim()
        shape[axis] = scale.numel()
        return scale.reshape(shape), zero_point.reshape(shape)

    def _quantize_with(self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
        """核心量化 + STE。"""
        scale, zero_point = self._broadcast(scale, zero_point, x)
        q = torch.clamp(
            torch.round(x / scale) + zero_point,
            float(self.qmin),
            float(self.qmax),
        )
        dq = (q - zero_point) * scale
        return x + (dq - x).detach()   # STE：前向用 dq，反向梯度直通

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """实数张量伪量化。"""
        if self.bypass:
            return x
        if self.observing:
            self.observe(x)
        if not self.enabled:
            return x

        if self.dynamic:
            mn, mx, _ = self._ranges(x, use_percentile=False)
            scale, zero_point = self._compute_params(mn, mx)
        else:
            if not self.calibrated:
                # 兜底：未校准就先按当前张量校准（部署时应避免走到这里）
                self.calibrate(x)
            scale, zero_point = self.scale, self.zero_point

        return self._quantize_with(x, scale, zero_point)

    #########################
    # 复数
    #########################
    def forward_complex(self, x: torch.Tensor) -> torch.Tensor:
        """复数张量伪量化：实部/虚部分别量化，但共用同一套量化参数。

        论文：Q(Ā) = Q(Re(Ā)) + j·Q(Im(Ā))，两部分使用相同 scale/zero_point，
        以降低未来硬件实现中复数乘法的复杂度。
        """
        parts = torch.cat([x.real, x.imag], dim=-1)

        if self.bypass:
            return x
        if self.observing:
            self.observe(parts)
        if not self.enabled:
            return x

        if self.dynamic:
            mn, mx, _ = self._ranges(parts, use_percentile=False)
            scale, zero_point = self._compute_params(mn, mx)
        else:
            if not self.calibrated:
                self.calibrate(parts)
            scale, zero_point = self.scale, self.zero_point

        return torch.complex(
            self._quantize_with(x.real, scale, zero_point),
            self._quantize_with(x.imag, scale, zero_point),
        )

    @torch.no_grad()
    def calibrate_complex(self, x: torch.Tensor, use_percentile: bool = True):
        """复数张量的一次性校准（实部虚部联合统计）。"""
        self.calibrate(torch.cat([x.real, x.imag], dim=-1), use_percentile=use_percentile)


def fake_quant_complex(x: torch.Tensor, quantizer: UniformQuantizer) -> torch.Tensor:
    """函数式接口，便于在核函数里直接调用。"""
    return quantizer.forward_complex(x)


#############################
# 便捷构造
#############################
def make_quantizer(bits: int, cfg, per_head: bool = True, axis: int = 0, name: str = "") -> UniformQuantizer:
    """按 QuantConfig 的公共字段创建量化器。

    cfg 需要提供 symmetric / percentile / calibration 三个属性。
    """
    return UniformQuantizer(
        bits=bits,
        symmetric=cfg.symmetric,
        per_head=per_head,
        channel_axis=axis,
        percentile=cfg.percentile,
        dynamic=(cfg.calibration == "dynamic"),
        name=name,
    )
