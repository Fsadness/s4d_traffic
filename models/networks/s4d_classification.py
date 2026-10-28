"""S4D 分类网络：整段前向 + 可选流式推理（早退）。

state_dict 键名与历史 checkpoint 完全一致：

    backbone.encoder.{weight,bias}
    backbone.s4_layers.{i}.{D,kernel.*,output_linear.0.*}
    backbone.norms.{i}.{weight,bias}
    decoder.{weight,bias}

因此 conv 训练的权重可以直接加载到 `streaming=True` 的模型里做递推推理。

历史实现拆成了 `s4d_classification.py` 与 `s4d_recurrent_classification.py` 两个类，
二者除流式状态机外完全相同；这里合并为一个类，`streaming` 控制走哪条前向。
"""

import torch
import torch.nn as nn

from models.backbones.s4d_backbone import S4DBackbone


class Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        lr=0.001,
        kernel_type='conv',
        streaming=False,
        early_stop_strategy=None,
        debug=True,
    ):
        super().__init__()

        if streaming and kernel_type != 'recurrent':
            raise ValueError(
                "streaming=True requires kernel_type='recurrent' "
                f"(got kernel_type={kernel_type!r})"
            )

        self.d_output = d_output
        self.d_model = d_model
        self.prenorm = prenorm
        self.streaming = streaming
        self.debug = debug

        # Backbone
        self.backbone = S4DBackbone(
            d_input=d_input,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            prenorm=prenorm,
            lr=lr,
            kernel_type=kernel_type,
        )

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

        # Early stop strategy（仅在 streaming=True 时使用）
        self.early_stop_strategy = early_stop_strategy

        # 流式推理状态
        self._currrent_step = 0
        self._total_steps = 0
        self._pool_sum = None       # (B, d_model)，已累积的特征和
        self._active_mask = None    # (B,) Bool，True 表示已停止
        self._stop_step = None      # (B,) Int，每个样本停止的步数
        self._final_logits = None   # (B, d_output)
        self._confidence_history = []

    # ------------------------------------------------------------------
    # 推理准备
    # ------------------------------------------------------------------
    def prepare_for_inference(self):
        """缓存每层离散化后的 SSM 参数。"""
        self.backbone.prepare_for_inference()

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self.prepare_for_inference()
        return self

    # ------------------------------------------------------------------
    # 流式推理
    # ------------------------------------------------------------------
    def reset_state(self, batch_size, device=None, total_steps=0):
        if not self.streaming:
            raise RuntimeError("reset_state() is only available when streaming=True")

        self.backbone.reset_state(batch_size, device)

        self._currrent_step = 0
        self._total_steps = total_steps
        self._pool_sum = torch.zeros(batch_size, self.d_model, device=device)
        self._active_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._stop_step = torch.full((batch_size,), -1, dtype=torch.int, device=device)
        self._final_logits = torch.zeros(batch_size, self.d_output, device=device)
        self._confidence_history = []

    def forward_step(self, x_t):
        """单步推理。

        input:  x_t: (B, d_input)
        output: logits (B, d_output),
                以及 debug=True 时的附加信息（prediction / confidence / should_stop / ...）
        """
        if not self.streaming:
            raise RuntimeError("forward_step() is only available when streaming=True")

        batch_size = x_t.shape[0]
        device = x_t.device
        info = None

        # Check reset state
        if self._active_mask is None or self._active_mask.shape[0] != batch_size:
            self.reset_state(batch_size, device)

        # Forward step and pooling
        torch.cuda.nvtx.range_push("pooling")
        self._pool_sum.add_(self.backbone.forward_step(x_t))  # (B, d_model)
        torch.cuda.nvtx.range_pop()

        # decoder
        torch.cuda.nvtx.range_push("decoder")
        logits = self.decoder(self._pool_sum / (self._currrent_step + 1))  # (B, d_output)
        torch.cuda.nvtx.range_pop()

        # Early stop strategy
        if self.early_stop_strategy is not None:
            should_stop, info = self.early_stop_strategy.should_stop(
                logits, self._currrent_step, self._total_steps
            )

            # Update active mask
            newly_stopped = should_stop & ~self._active_mask
            self._final_logits[~self._active_mask] = logits[~self._active_mask]
            self._stop_step[newly_stopped] = self._currrent_step
            self._active_mask = self._active_mask | should_stop
        else:
            should_stop = torch.zeros(batch_size, dtype=torch.bool, device=device)
            info = None

        self._currrent_step += 1

        all_stop = self._active_mask.all().item()

        # Update confidence history
        if self.debug:
            probs = torch.softmax(logits, dim=-1)
            confidence, prediction = probs.max(dim=-1)
            self._confidence_history.append(confidence.detach().cpu())
            return logits, all_stop, {
                'prediction': prediction,
                'confidence': confidence,
                'should_stop': should_stop,
                'current_step': self._currrent_step - 1,
                'info': info,
            }
        else:
            return logits, all_stop, None

    def _forward_stream(self, x, early_stop=True):
        """整段序列的流式前向：逐时间步调用 forward_step，可被策略提前结束。"""
        batch_size, length, _ = x.shape
        device = x.device

        self.reset_state(batch_size, device, total_steps=length)

        for x_t in x.unbind(dim=1):
            logits, all_stop, results = self.forward_step(x_t)

            if early_stop and all_stop:
                break

        # Final logits
        if not self._active_mask.all():
            self._final_logits[~self._active_mask] = logits[~self._active_mask]
            self._stop_step[~self._active_mask] = self._currrent_step - 1

        if self.debug:
            probs = torch.softmax(self._final_logits, dim=-1)
            confidence, prediction = probs.max(dim=-1)

            return logits, {
                'prediction': prediction,
                'confidence': confidence,
                'stop_step': self._stop_step,
                'confidence_history': self._confidence_history,
            }
        else:
            return logits, None

    def get_statistics(self):
        if self._stop_step is None:
            return {}

        stopped_mask = self._stop_step >= 0
        avg_stop_step = (
            self._stop_step[stopped_mask].float().mean().item()
            if stopped_mask.any() else -1
        )
        early_stop_ratio = stopped_mask.float().mean().item()

        return {
            'avg_stop_step': avg_stop_step,
            'early_stop_ratio': early_stop_ratio,
            'stop_step': self._stop_step,
            'confidence_history': self._confidence_history,
        }

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(self, x, early_stop=True):
        """(B, L, d_input) -> logits；streaming=True 时返回 (logits, info)。"""
        if self.streaming:
            return self._forward_stream(x, early_stop=early_stop)

        # 整段前向：平均池化 + 线性分类头
        x = self.backbone(x)          # (B, L, d_model)
        x = x.mean(dim=1)             # (B, d_model)
        x = self.decoder(x)           # (B, d_output)

        return x
