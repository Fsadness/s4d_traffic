"""recurrent.py —— 递推模式 S4D 核（量化版）。

对应论文 II 节与 Algorithm 1：
    x[n] = Ā · x[n-1] + B̄ · u[n]
    y[n] = Re(2 · Σ C · x[n]) + D · u[n]
    每步对 x 做 [-50, 50] 裁剪（可再量化到 8-bit）
    y 经 qGELU → Conv1D(1x1) → GLU

与仓库 models/kernels/s4d_recurrent_kernel.py 的区别：
    1. 离散参数（Ā、B̄、C、D）在每次前向时先做伪量化；
    2. 状态 x 在递推中先裁剪再量化（论文：状态是量化最敏感的运行时量）；
    3. 激活用 qGELU，并在 qGELU / Conv1D / GLU 之后做激活量化；
    4. 支持三种 QAT 参数化（continuous / discrete / frozen）。

state_dict 键名与仓库保持一致（continuous 模式）：
    kernel.log_dt / log_A_real / A_imag / C
discrete / frozen 模式下改为：
    kernel.A_bar_real / A_bar_imag / B_bar_real / B_bar_imag / C_real / C_imag
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from .fake_quant import clip_state, make_quantizer


#############################
# 量化递推核
#############################
class QuantS4DRecurrentKernel(nn.Module):
    """S4D 递推核：连续时间参数 → ZOH 离散化 → 伪量化 → 逐步递推。

    参数化（param_mode）：
        continuous : 训练 log_dt / log_A_real / A_imag / C（与仓库 S4D 一致），
                     Ā = Q(exp(A·Δt))、B̄ = Q((exp(A·Δt)-1)/A)
        discrete   : 直接训练 Ā / B̄ / C 的实部与虚部
        frozen     : 同 discrete，但 Ā 冻结（论文在低比特下表现最好）
    """

    def __init__(
        self,
        d_model: int,
        N: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float = None,
        cfg=None,
        param_mode: str = "continuous",
    ):
        super().__init__()
        self.H = d_model
        self.N = N
        self.N_half = N // 2
        self.cfg = cfg if cfg is not None else self._default_cfg()
        self.param_mode = param_mode

        # ---- 连续时间参数（先按仓库 S4D 的方式初始化，随后由 best.pth 覆盖）----
        self._init_continuous(dt_min, dt_max, lr)

        # ---- 量化器 ----
        # SSM 参数沿 H 维逐头量化（论文 per-head）；D 逐张量即可（每个头只有 1 个数）
        self.q_a = make_quantizer(self.cfg.bits_a, self.cfg, per_head=self.cfg.per_head, axis=0, name="A")
        self.q_b = make_quantizer(self.cfg.bits_b, self.cfg, per_head=self.cfg.per_head, axis=0, name="B")
        self.q_c = make_quantizer(self.cfg.bits_c, self.cfg, per_head=self.cfg.per_head, axis=0, name="C")
        self.q_d = make_quantizer(self.cfg.bits_d, self.cfg, per_head=False, axis=0, name="D")

        # 状态 x 形如 (B, H, N/2)，沿 H 轴（axis=1）逐头量化
        self.q_state = make_quantizer(
            self.cfg.bits_state, self.cfg, per_head=self.cfg.per_head, axis=1, name="state"
        )

        # 激活形如 (B, H, L)，同样沿 H 轴
        self.q_act_gelu = make_quantizer(
            self.cfg.bits_act, self.cfg, per_head=self.cfg.per_head, axis=1, name="act.gelu"
        )
        self.q_act_conv = make_quantizer(
            self.cfg.bits_act, self.cfg, per_head=self.cfg.per_head, axis=1, name="act.conv"
        )
        self.q_act_out = make_quantizer(
            self.cfg.bits_act, self.cfg, per_head=self.cfg.per_head, axis=1, name="act.out"
        )

        # ---- 按序列位置分段的额外量化器 ----
        # 未配置 segment_bits 时不创建任何模块，state_dict 键名与旧版完全一致。
        self._init_segment_quantizers()

    def _init_segment_quantizers(self):
        """为分段位宽创建额外的量化器（按位宽去重）。

        默认位宽沿用上面的 q_state / q_act_*，只有"与默认不同"的位宽才需要额外模块；
        这样默认方案（无分段）的 checkpoint 键名保持不变。
        """
        cfg = self.cfg
        self.q_state_extra = nn.ModuleDict()
        self.q_act_extra = nn.ModuleDict()

        if not getattr(cfg, "segment_bits", None):
            return

        state_bits = sorted(
            {
                int(segment["bits_state"])
                for segment in cfg.segment_bits
                if segment.get("bits_state") is not None
            }
            - {cfg.bits_state}
        )
        act_bits = sorted(
            {
                int(segment["bits_act"])
                for segment in cfg.segment_bits
                if segment.get("bits_act") is not None
            }
            - {cfg.bits_act}
        )

        for bits in state_bits:
            self.q_state_extra[str(bits)] = make_quantizer(
                bits, cfg, per_head=cfg.per_head, axis=1, name=f"state.b{bits}"
            )

        for bits in act_bits:
            for kind in ("gelu", "conv", "out"):
                self.q_act_extra[f"{kind}_{bits}"] = make_quantizer(
                    bits, cfg, per_head=cfg.per_head, axis=1,
                    name=f"act.{kind}.b{bits}",
                )

    def _state_quantizer(self, bits):
        if bits == self.cfg.bits_state:
            return self.q_state
        return self.q_state_extra[str(bits)]

    def _act_quantizer(self, kind, bits):
        if bits == self.cfg.bits_act:
            return {
                "gelu": self.q_act_gelu,
                "conv": self.q_act_conv,
                "out": self.q_act_out,
            }[kind]
        return self.q_act_extra[f"{kind}_{bits}"]

    def _quantize_activations(self, y: torch.Tensor, kind: str) -> torch.Tensor:
        """按 `segment_bits` 对激活沿序列维（最后一维）分段量化。"""
        cfg = self.cfg
        if not cfg.quantize_activations:
            return y

        if not getattr(cfg, "segment_bits", None):
            quantizer = self._act_quantizer(kind, cfg.bits_act)
            return quantizer(y) if self._active(quantizer) else y

        pieces = []
        for start, end, bits_act, _ in cfg.segments_for_length(y.shape[-1]):
            quantizer = self._act_quantizer(kind, bits_act)
            piece = y[..., start:end]
            if self._active(quantizer):
                piece = quantizer(piece)
            pieces.append(piece)

        if len(pieces) == 1:
            return pieces[0]
        return torch.cat(pieces, dim=-1)

    #########################
    # 初始化
    #########################
    @staticmethod
    def _default_cfg():
        """未显式传配置时，退回论文的异构默认值。"""
        from .config import paper_config

        return paper_config()

    def _init_continuous(self, dt_min: float, dt_max: float, lr: float):
        """与仓库 S4DBaseKernel / init_s4d_params 完全一致的初始化。"""
        H, N_half = self.H, self.N_half

        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        C = torch.randn(H, N_half, dtype=torch.cfloat)
        log_A_real = torch.log(0.5 * torch.ones(H, N_half))
        A_imag = math.pi * repeat(torch.arange(N_half), "n -> h n", h=H)

        # 与仓库 register_param 一致：SSM 参数零权重衰减、可指定更小学习率
        self.log_dt = self._make_param(log_dt, lr)
        self.log_A_real = self._make_param(log_A_real, lr)
        self.A_imag = self._make_param(A_imag, lr)
        # 仓库里 C 是普通 Parameter（无 _optim 钩子），这里保持一致
        self.C = nn.Parameter(torch.view_as_real(C))

    @staticmethod
    def _make_param(tensor: torch.Tensor, lr: float = None) -> nn.Parameter:
        """注册参数并挂上仓库优化器识别的 _optim 元信息。"""
        p = nn.Parameter(tensor)
        optim = {"weight_decay": 0.0}
        if lr is not None:
            optim["lr"] = lr
        p._optim = optim
        return p

    #########################
    # 参数化转换
    #########################
    @torch.no_grad()
    def to_discrete_mode(self, freeze_a: bool = False):
        """把连续参数物化为离散参数，并切换到 discrete / frozen 参数化。

        应在加载完 FP checkpoint 之后调用。
        """
        A_bar, B_bar, C = self._continuous_discrete()
        A_bar = A_bar.detach()
        B_bar = B_bar.detach()
        C = C.detach()

        # 删除连续参数（nn.Module.__delattr__ 会同步清理 _parameters）
        del self.log_dt
        del self.log_A_real
        del self.A_imag
        del self.C

        self.A_bar_real = nn.Parameter(A_bar.real.clone())
        self.A_bar_imag = nn.Parameter(A_bar.imag.clone())
        self.B_bar_real = nn.Parameter(B_bar.real.clone())
        self.B_bar_imag = nn.Parameter(B_bar.imag.clone())
        self.C_real = nn.Parameter(C.real.clone())
        self.C_imag = nn.Parameter(C.imag.clone())

        if freeze_a:
            # 论文的 Frozen Ā：只冻结状态转移矩阵，B̄ / C / D 仍训练
            self.A_bar_real.requires_grad_(False)
            self.A_bar_imag.requires_grad_(False)
            self.param_mode = "frozen"
        else:
            self.param_mode = "discrete"

    #########################
    # 离散化
    #########################
    def _continuous_discrete(self):
        """由连续参数计算 ZOH 离散参数（未量化）。"""
        dt = torch.exp(self.log_dt)                                   # (H,)
        C = torch.view_as_complex(self.C)                             # (H, N/2)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag            # (H, N/2)
        dtA = A * dt.unsqueeze(-1)
        A_bar = torch.exp(dtA)
        B_bar = (torch.exp(dtA) - 1.0) / A
        return A_bar, B_bar, C

    def _discrete_params(self):
        """按当前参数化返回未量化的 (Ā, B̄, C)。"""
        if self.param_mode == "continuous":
            return self._continuous_discrete()
        A_bar = torch.complex(self.A_bar_real, self.A_bar_imag)
        B_bar = torch.complex(self.B_bar_real, self.B_bar_imag)
        C = torch.complex(self.C_real, self.C_imag)
        return A_bar, B_bar, C

    def quantized_params(self):
        """返回量化后的 (Ā, B̄, C)——论文：Q(·) 总是作用在 Disc(·) 之后。"""
        A_bar, B_bar, C = self._discrete_params()
        return (
            self.q_a.forward_complex(A_bar),
            self.q_b.forward_complex(B_bar),
            self.q_c.forward_complex(C),
        )

    #########################
    # 前向
    #########################
    def forward(self, u, D, dropout, activation, output_linear, transposed=True):
        """完整序列前向。

        Args:
            u: (B, H, L)
            D: (H,) 直通项
            dropout: 仓库传入的 dropout 模块（eval 时无效）
            activation: qGELU 模块（构建量化模型时替换）
            output_linear: Sequential(QuantConv1d, nn.GLU)
            transposed: 是否为 (B, H, L) 布局
        """
        if not transposed:
            u = u.transpose(-1, -2)

        B_batch, H, L = u.shape

        # ---- 1) 参数：离散化 + 伪量化 ----
        A_bar, B_bar, C = self.quantized_params()
        D_q = self.q_d(D)

        # ---- 2) 递推 ----
        x = torch.zeros(
            B_batch, H, self.N_half, dtype=torch.cfloat, device=u.device
        )
        A_b = A_bar.unsqueeze(0)          # (1, H, N/2)
        B_b = B_bar.unsqueeze(0)
        C_b = C.unsqueeze(0)

        # 分段位宽：预先算出每个时间步的状态位宽（无分段时为 None，保持原来的快速路径）
        state_bits_seq = None
        if getattr(self.cfg, "segment_bits", None):
            state_bits_seq = [self.cfg.bits_at(t)[1] for t in range(L)]

        steps = []
        for t in range(L):
            u_t = u[:, :, t]                                # (B, H)
            x = A_b * x + B_b * u_t.unsqueeze(-1)           # 状态更新

            # 论文 IV 节：每步裁剪，防止量化误差随 n 放大导致发散
            x = clip_state(x, self.cfg.state_clip)

            # 状态量化（论文：量化 x 是低比特 SSM 的必要条件）
            # 快速路径：量化关闭且不在校准时，跳过（避免每步 torch.cat 的开销）
            if self.cfg.quantize_state:
                bits = state_bits_seq[t] if state_bits_seq is not None else self.cfg.bits_state
                quantizer = self._state_quantizer(bits)
                if self._active(quantizer):
                    x = quantizer.forward_complex(x)

            steps.append(2.0 * (C_b * x).sum(dim=-1).real)  # y_t = Re(2ΣCx)

        y = torch.stack(steps, dim=-1)                      # (B, H, L)

        # ---- 3) D 跳连 ----
        y = y + u * D_q.unsqueeze(-1)

        # ---- 4) qGELU + Dropout + 激活量化 ----
        # 与仓库 s4d_recurrent_kernel 一致：dropout(activation(y))
        y = dropout(activation(y))
        y = self._quantize_activations(y, "gelu")

        # ---- 5) Conv1D(1x1) + GLU ----
        y = self._quantized_output_linear(y, output_linear)

        if not transposed:
            y = y.transpose(-1, -2)
        return y, None

    #########################
    # 输出混合（Conv1D + GLU）
    #########################
    def _quantized_output_linear(self, y: torch.Tensor, output_linear: nn.Module):
        """Conv1d(1x1) 权重量化 + 输出激活量化 + GLU。"""
        conv = output_linear[0]
        w = conv.weight
        if hasattr(conv, "weight_quantizer"):
            w = conv.weight_quantizer(w)
        y = F.conv1d(y, w, conv.bias)

        y = self._quantize_activations(y, "conv")

        y = F.glu(y, dim=-2)

        y = self._quantize_activations(y, "out")
        return y

    @staticmethod
    def _active(quantizer) -> bool:
        """量化器是否需要执行（启用中，或校准观察中）。"""
        return quantizer.enabled or quantizer.observing

    #########################
    # 杂项
    #########################
    def reset_state(self, batch_size, device):
        """为流式推理预留的接口（与仓库命名一致）。"""
        self._state = torch.zeros(
            batch_size, self.H, self.N_half, dtype=torch.cfloat, device=device
        )

    def calibrate_params(self):
        """按当前权重一次性校准 SSM 参数的量化范围（论文的 PTQ 参数校准）。"""
        with torch.no_grad():
            A_bar, B_bar, C = self._discrete_params()
            self.q_a.calibrate_complex(A_bar)
            self.q_b.calibrate_complex(B_bar)
            self.q_c.calibrate_complex(C)

    def extra_repr(self) -> str:
        return f"H={self.H}, N={self.N}, mode={self.param_mode}"
