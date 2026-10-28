"""推理期早退策略。

当前阶段只提供**默认实现**：`ExitPolicy` 永不提前结束（始终返回全 False），
因此接上它之后模型行为与"完整递推"完全一致。后续做置信度阈值、margin、
patience 等策略时，只需新增一个子类，模型与评估入口都不需要改动。

接口约定（与 `models/networks/s4d_recurrent_classification.py` 对齐）：

- `reset(batch_size, total_steps, device)`：每段序列开始前重置内部状态；
- `should_stop(logits, step, total_steps)`：返回 `(B,)` 的 bool 掩码与可选 info；
  - 已经停止的样本必须持续返回 True（单调性）；
  - 只依赖 logits，因此**量化后的 logits 可以直接复用同一套策略**。

注意：这里的"早退"是**推理期**提前结束，与训练的早停（`--use_early_stop`）无关。
"""

import torch


class ExitPolicy:
    """默认策略：不早退。"""

    name = "none"

    def reset(self, batch_size, total_steps=None, device=None):
        return None

    def should_stop(self, logits, step, total_steps=None):
        should_stop = torch.zeros(
            logits.size(0), dtype=torch.bool, device=logits.device
        )
        return should_stop, None

    def describe(self):
        return self.name

    def state_dict(self):
        return {}


# 策略注册表：后续新增 ConfidenceExit 等实现时在这里登记。
POLICIES = {
    "none": ExitPolicy,
}


def build_exit_policy(name="none"):
    if name not in POLICIES:
        raise ValueError(
            f"Unknown exit policy: {name}. Available: {sorted(POLICIES)}"
        )
    return POLICIES[name]()
