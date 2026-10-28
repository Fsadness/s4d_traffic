"""
S4D Backbone：把 (B, L, d_input) 映射到 (B, L, d_model)。

同一套参数支持两种核（conv / recurrent），并同时提供两种前向方式：

- `forward(x)`：整段序列前向，内部用 FFT 卷积核或整段递推核；
- `forward_step(x_t)` + `reset_state(B, device)`：逐步推理，供流式/早退使用
  （只有 `kernel_type='recurrent'` 时可用）。

历史实现里这两条路径分属 `s4d_backbone.py` 与 `s4d_recurrent_backbone.py`，
参数命名一致、代码 90% 重复；这里合并为一个类，state_dict 键名保持不变。
"""

import torch
import torch.nn as nn

from models.layers.s4d_layer import S4DLayer

# Dropout 在 PyTorch 1.11 有 bug（沿用历史判断）
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


class S4DBackbone(nn.Module):

    def __init__(
        self,
        d_input,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        lr=0.001,
        kernel_type='conv',
    ):
        super().__init__()

        self.prenorm = prenorm
        self.d_model = d_model
        self.kernel_type = kernel_type

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4DLayer(d_model, dropout=dropout, transposed=True, kernel_type=kernel_type, lr=min(0.001, lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

    # ------------------------------------------------------------------
    # 推理准备 / 流式状态
    # ------------------------------------------------------------------
    def prepare_for_inference(self):
        """缓存每层离散化后的 SSM 参数。"""
        for layer in self.s4_layers:
            layer.prepare_for_inference()

    def reset_state(self, batch_size, device=None):
        """重置所有循环层的状态（仅 recurrent 核有效）。"""
        for layer in self.s4_layers:
            if getattr(layer, "kernel_type", None) == "recurrent":
                layer.reset_state(batch_size, device)

    # ------------------------------------------------------------------
    # 整段前向
    # ------------------------------------------------------------------
    def forward(self, x):
        """Input x is shape (B, L, d_input)."""
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            if self.training:
                z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        return x

    # ------------------------------------------------------------------
    # 逐步前向
    # ------------------------------------------------------------------
    def forward_step(self, x_t):
        """Input x_t is shape (B, d_input); returns (B, d_model)."""
        x_t = self.encoder(x_t)  # (B, d_input) -> (B, d_model)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x_t
            if self.prenorm:
                z = norm(z)

            # S4D single step forward
            z = layer.forward_step(z)  # (B, d_model)

            # dropout
            if self.training:
                z = dropout(z.unsqueeze(-1)).squeeze(-1)

            # Residual connection
            x_t = z + x_t

            if not self.prenorm:
                x_t = norm(x_t)

        return x_t
