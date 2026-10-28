"""护栏测试：默认 ExitPolicy 必须永不早退，流式模型必须跑满整段序列。

当前阶段 `policies.ExitPolicy` 是空实现（恒定返回 False），
这个测试保证"接入策略"不会悄悄改变模型行为。

运行：
    python tests/test_exit_policy.py
    python -m unittest tests.test_exit_policy -v
"""

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from policies import ExitPolicy, build_exit_policy  # noqa: E402


class ExitPolicyTest(unittest.TestCase):
    def test_default_policy_never_stops(self):
        policy = build_exit_policy("none")
        self.assertIsInstance(policy, ExitPolicy)
        policy.reset(batch_size=4, total_steps=100)

        # 即使 logits 极其"自信"，默认策略也不应提前结束
        logits = torch.randn(4, 6) * 100
        should_stop, info = policy.should_stop(logits, step=0, total_steps=100)

        self.assertEqual(should_stop.dtype, torch.bool)
        self.assertEqual(tuple(should_stop.shape), (4,))
        self.assertFalse(bool(should_stop.any()))
        self.assertIsNone(info)

    def test_streaming_model_runs_full_length(self):
        from models.networks.s4d_recurrent_classification import Model

        length = 24
        model = Model(
            d_input=1,
            d_output=6,
            d_model=32,
            n_layers=1,
            dropout=0.0,
            kernel_type="recurrent",
            early_stop_strategy=build_exit_policy("none"),
            debug=True,
        ).eval()
        model.prepare_for_inference()

        x = torch.zeros(2, length, 1)
        with torch.no_grad():
            logits, info = model(x)

        self.assertEqual(tuple(logits.shape), (2, 6))
        self.assertEqual(model._total_steps, length)
        # 未早退：所有样本都在最后一步才"停止"
        self.assertTrue(bool((info["stop_step"] == length - 1).all()))
        self.assertEqual(model.get_statistics()["avg_stop_step"], float(length - 1))

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            build_exit_policy("does-not-exist")


if __name__ == "__main__":
    unittest.main(verbosity=2)
