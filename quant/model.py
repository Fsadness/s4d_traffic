"""model.py —— 量化版 S4D 分类模型。

结构与仓库 models/networks/s4d_classification.py + models/backbones/s4d_backbone.py
完全对应，只是把每一层换成“递推 + 量化”的版本：

    x = encoder(u)                       # QuantLinear, 权重 4-bit
    x = x.transpose(-1, -2)              # (B, H, L)
    for layer, norm, dropout in ...:
        z = norm(x) if prenorm          # QuantLayerNorm, 权重 4-bit
        z, _ = layer(z)                 # QuantS4DLayer（递推 S4D + qGELU + Conv1D + GLU）
        z = dropout(z)
        x = z + x
        x = norm(x) if not prenorm
        x = q_block(x)                  # 块输出激活 6-bit
    x = x.transpose(-1, -2)
    x = x.mean(dim=1)                   # 平均池化
    x = q_head(x)                       # 池化后激活 6-bit
    logits = decoder(x)                 # QuantLinear, 权重 4-bit

state_dict 键名与仓库 FP 模型一致（continuous 参数化下可直接 load best.pth）：
    backbone.encoder.weight / bias
    backbone.s4_layers.{i}.D
    backbone.s4_layers.{i}.kernel.{log_dt,log_A_real,A_imag,C}
    backbone.s4_layers.{i}.output_linear.0.{weight,bias}
    backbone.norms.{i}.{weight,bias}
    decoder.{weight,bias}
"""

import torch
import torch.nn as nn

from .config import QuantConfig
from .fake_quant import QGELU, UniformQuantizer, make_quantizer
from .modules import (
    QuantConv1d,
    QuantLayerNorm,
    QuantLinear,
    set_observing,
    set_quant_enabled,
)
from .recurrent import QuantS4DRecurrentKernel


#############################
# 单个 S4D 层
#############################
class QuantS4DLayer(nn.Module):
    """递推 S4D 层：量化核 + D 跳连 + qGELU + Dropout + Conv1D/GLU。"""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        transposed: bool = True,
        lr: float = None,
        cfg: QuantConfig = None,
        param_mode: str = "continuous",
    ):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = d_model
        self.transposed = transposed

        # 直通项 D（论文归入 SSM 权重，量化到 bits_d）
        self.D = nn.Parameter(torch.randn(self.h))

        # 递推核（内部完成 Ā/B̄/C 的量化与状态量化）
        self.kernel = QuantS4DRecurrentKernel(
            self.h, N=self.n, lr=lr, cfg=cfg, param_mode=param_mode
        )

        # 论文的 qGELU 替换原始 GELU；做 FP 基线时可切回 nn.GELU
        self.activation = QGELU() if cfg.use_qgelu else nn.GELU()
        self.dropout = nn.Dropout1d(dropout) if dropout > 0.0 else nn.Identity()

        # 逐点混合：Conv1d(1x1) + GLU（权重 4-bit，输出激活 6-bit 在核内量化）
        self.output_linear = nn.Sequential(
            QuantConv1d(self.h, 2 * self.h, kernel_size=1, cfg=cfg),
            nn.GLU(dim=-2),
        )

    def forward(self, u):
        """输入/输出 (B, H, L)。"""
        return self.kernel(
            u, self.D, self.dropout, self.activation, self.output_linear, self.transposed
        )


#############################
# 骨干
#############################
class QuantS4DBackbone(nn.Module):
    """S4D 骨干：线性编码器 + N 个残差块 + 块间激活量化。"""

    def __init__(
        self,
        d_input: int,
        d_model: int = 256,
        n_layers: int = 4,
        dropout: float = 0.2,
        prenorm: bool = False,
        lr: float = 0.001,
        cfg: QuantConfig = None,
        param_mode: str = "continuous",
        d_state: int = 64,
    ):
        super().__init__()
        self.cfg = cfg
        self.prenorm = prenorm
        self.d_model = d_model

        # 线性编码器（d_input=1 对应灰度像素 / tokenizer 的最后一维）
        self.encoder = QuantLinear(d_input, d_model, bias=True, cfg=cfg)

        # 编码器输出激活量化（形状 (B, H, L)，沿 H 轴）
        self.q_in = make_quantizer(
            cfg.bits_act, cfg, per_head=cfg.per_head, axis=1, name="backbone.in"
        )

        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.q_blocks = nn.ModuleList()

        for _ in range(n_layers):
            self.s4_layers.append(
                QuantS4DLayer(
                    d_model,
                    d_state=d_state,
                    dropout=dropout,
                    transposed=True,
                    lr=min(0.001, lr),
                    cfg=cfg,
                    param_mode=param_mode,
                )
            )
            self.norms.append(QuantLayerNorm(d_model, cfg=cfg))
            self.dropouts.append(nn.Dropout1d(dropout))
            # 每个残差块输出后的激活量化（论文：除状态外激活 6-bit）
            self.q_blocks.append(
                make_quantizer(
                    cfg.bits_act, cfg, per_head=cfg.per_head, axis=1,
                    name=f"backbone.block{len(self.q_blocks)}",
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (B, L, d_input)，输出 (B, L, d_model)。"""
        x = self.encoder(x)                    # (B, L, d_model)
        x = x.transpose(-1, -2)                # (B, d_model, L)
        if self.cfg.quantize_activations and self._active(self.q_in):
            x = self.q_in(x)

        for layer, norm, dropout, q_block in zip(
            self.s4_layers, self.norms, self.dropouts, self.q_blocks
        ):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            z, _ = layer(z)                    # 量化递推 S4D

            if self.training:
                z = dropout(z)

            x = z + x                          # 残差

            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

            if self.cfg.quantize_activations and self._active(q_block):
                x = q_block(x)                 # 块输出激活量化

        return x.transpose(-1, -2)             # (B, L, d_model)

    @staticmethod
    def _active(quantizer) -> bool:
        """量化器是否需要执行（启用中，或校准观察中）。"""
        return quantizer.enabled or quantizer.observing


#############################
# 分类模型
#############################
class QuantS4DClassification(nn.Module):
    """量化版序列分类模型：骨干 + 均值池化 + 线性分类头。"""

    def __init__(
        self,
        d_input: int = 1,
        d_output: int = 10,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
        prenorm: bool = False,
        lr: float = 0.001,
        cfg: QuantConfig = None,
        param_mode: str = "continuous",
        d_state: int = 64,
    ):
        super().__init__()
        cfg = cfg if cfg is not None else QuantConfig()
        self.cfg = cfg
        self.prenorm = prenorm
        self.d_output = d_output
        self.d_model = d_model

        self.backbone = QuantS4DBackbone(
            d_input=d_input,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            prenorm=prenorm,
            lr=lr,
            cfg=cfg,
            param_mode=param_mode,
            d_state=d_state,
        )

        # 池化后激活量化（(B, H) 形状，沿 H 轴）
        self.q_head = make_quantizer(
            cfg.bits_act, cfg, per_head=cfg.per_head, axis=1, name="head"
        )

        # 线性分类头（权重 4-bit）
        self.decoder = QuantLinear(d_model, d_output, bias=True, cfg=cfg)

    #########################
    # 前向
    #########################
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, d_input) -> (B, d_output) logits。"""
        x = self.backbone(x)          # (B, L, d_model)
        x = x.mean(dim=1)             # 序列平均池化
        if self.cfg.quantize_activations and (self.q_head.enabled or self.q_head.observing):
            x = self.q_head(x)
        return self.decoder(x)

    #########################
    # 量化相关工具
    #########################
    def calibrate_weights(self):
        """校准所有“与数据无关”的量化参数（SSM 矩阵、Linear/Conv/Norm 权重）。"""
        for layer in self.backbone.s4_layers:
            layer.kernel.calibrate_params()
            layer.output_linear[0].calibrate()
        self.backbone.encoder.calibrate()
        for norm in self.backbone.norms:
            norm.calibrate()
        self.decoder.calibrate()

    def quantizers(self):
        """遍历所有量化器（便于统计/调试）。"""
        for m in self.modules():
            if isinstance(m, UniformQuantizer):
                yield m

    def mark_calibrated(self):
        """从 checkpoint 恢复后，把所有量化器标记为“已校准”。"""
        for q in self.quantizers():
            q.calibrated = True

    def set_quantization(self, enabled: bool):
        """推理/训练时开量化；校准阶段关闭。"""
        set_quant_enabled(self, enabled)

    def set_observing(self, observing: bool):
        """校准阶段只让未校准的量化器收集统计量。"""
        set_observing(self, observing)


#############################
# 从 FP checkpoint 构建
#############################
def infer_model_config(fp_state_dict: dict) -> dict:
    """从 FP state_dict 的键名/形状反推模型结构。"""
    d_model, d_input = fp_state_dict["backbone.encoder.weight"].shape
    d_output = fp_state_dict["decoder.weight"].shape[0]

    layer_ids = {
        int(k.split(".")[2])
        for k in fp_state_dict
        if k.startswith("backbone.s4_layers.")
    }
    d_state = fp_state_dict[
        "backbone.s4_layers.0.kernel.A_imag"
    ].shape[1] * 2

    return dict(
        d_input=d_input,
        d_output=d_output,
        d_model=d_model,
        n_layers=len(layer_ids),
        d_state=d_state,
    )


def build_quant_model(
    fp_state_dict: dict,
    model_cfg: dict = None,
    qcfg: QuantConfig = None,
    verbose: bool = True,
) -> QuantS4DClassification:
    """由 FP state_dict 构建量化模型。

    步骤：
        1. 按 FP 键名构建同构模型（continuous 参数化）
        2. 加载 FP 权重
        3. 若需要 discrete/frozen，把连续参数物化为离散参数
        4. 校准 SSM 参数与非 SSM 权重
    """
    qcfg = qcfg if qcfg is not None else QuantConfig()
    model_cfg = model_cfg or infer_model_config(fp_state_dict)

    model = QuantS4DClassification(
        **model_cfg,
        cfg=qcfg,
        param_mode="continuous",   # 先按连续参数加载，再按需转换
    )

    missing, unexpected = model.load_state_dict(fp_state_dict, strict=False)

    # 只允许“量化器 buffer”缺失；其余缺失/多余都说明键名不匹配
    real_missing = [k for k in missing if not _is_quant_buffer(k)]
    if real_missing or unexpected:
        raise RuntimeError(
            f"checkpoint 与模型结构不匹配：missing={real_missing[:5]} unexpected={list(unexpected)[:5]}"
        )

    # 参数化转换
    if qcfg.param_mode != "continuous":
        for layer in model.backbone.s4_layers:
            layer.kernel.to_discrete_mode(freeze_a=(qcfg.param_mode == "frozen"))

    # 校准权重相关的量化参数（不需要数据）
    model.calibrate_weights()

    # 消融实验：完全跳过非 SSM 权重量化
    if not qcfg.quantize_weights:
        for module in model.modules():
            if isinstance(module, (QuantLinear, QuantConv1d, QuantLayerNorm)):
                module.weight_quantizer.bypass = True

    if verbose:
        print(f"[quant] 模型: {model_cfg}")
        print(f"[quant] 方案: {qcfg.describe()}")

    return model


def _is_quant_buffer(key: str) -> bool:
    """判断 state_dict 的缺失键是否只是量化器 buffer（允许缺失）。"""
    return any(
        key.endswith(suffix)
        for suffix in ("scale", "zero_point", "obs_min", "obs_max")
    )


#############################
# checkpoint 存取
#############################
def save_quant_checkpoint(model: QuantS4DClassification, path: str, meta: dict = None):
    """保存量化模型（含量化参数与结构信息）。"""
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": {
                "d_input": model.backbone.encoder.in_features,
                "d_output": model.d_output,
                "d_model": model.d_model,
                "n_layers": len(model.backbone.s4_layers),
                "d_state": model.backbone.s4_layers[0].n,
                "dropout": 0.0,          # 推理时已无意义
                "prenorm": model.prenorm,
            },
            "quant_config": model.cfg.to_dict(),
            "meta": meta or {},
        },
        path,
    )


def load_quant_checkpoint(path: str, map_location="cpu") -> QuantS4DClassification:
    """加载量化模型 checkpoint。

    map_location 可以是 "cpu" / "cuda" / "cuda:0"，模型会在加载完成后整体搬到该设备
    （不能直接把参数量化 buffer 混在两个设备上）。
    """
    ckpt = torch.load(path, map_location="cpu")
    mcfg = ckpt["model_config"]
    qcfg = QuantConfig.from_dict(ckpt["quant_config"])

    model = QuantS4DClassification(**mcfg, cfg=qcfg, param_mode="continuous")
    if qcfg.param_mode != "continuous":
        for layer in model.backbone.s4_layers:
            layer.kernel.to_discrete_mode(freeze_a=(qcfg.param_mode == "frozen"))

    _load_state_dict_with_buffers(model, ckpt["model"])
    model.mark_calibrated()

    if map_location is not None and map_location != "cpu":
        model = model.to(map_location)
    return model


def _load_state_dict_with_buffers(model: nn.Module, state_dict: dict):
    """加载 state_dict，允许量化参数 buffer 的形状与初始值不同。"""
    own = model.state_dict()
    to_load = {}
    for key, value in state_dict.items():
        if key in own and own[key].shape != value.shape:
            # 量化参数（scale/zero_point）：直接替换 buffer
            module_path, _, attr = key.rpartition(".")
            module = model.get_submodule(module_path)
            setattr(module, attr, value)
        else:
            to_load[key] = value
    model.load_state_dict(to_load, strict=False)
