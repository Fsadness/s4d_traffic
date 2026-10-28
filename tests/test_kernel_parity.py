"""护栏测试：conv 与 recurrent 两种核必须"同键名、同数值"。

这是后续"conv 训练 → 递推推理 / 量化"路线的前提：
- 权重可以直接迁移（state_dict 键名完全一致）
- 迁移后输出数值等价（atol=1e-5）
- 并且必须通过 `engine.checkpoint.load_fp_checkpoint` 加载（它会重建离散参数缓存）

运行：
    python tests/test_kernel_parity.py
    python -m unittest tests.test_kernel_parity -v
"""

import os
import sys
import tempfile
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.checkpoint import load_fp_checkpoint  # noqa: E402
from models.networks.s4d_classification import Model  # noqa: E402


def build_model(kernel_type, seed=0):
    torch.manual_seed(seed)
    return Model(
        d_input=1,
        d_output=6,
        d_model=64,
        n_layers=2,
        dropout=0.0,
        kernel_type=kernel_type,
    ).eval()


class KernelParityTest(unittest.TestCase):
    def test_state_dict_keys_are_identical(self):
        conv = build_model("conv")
        rec = build_model("recurrent")

        conv_keys = set(conv.state_dict().keys())
        rec_keys = set(rec.state_dict().keys())

        self.assertEqual(conv_keys, rec_keys, "两种核的参数/缓冲键名必须一致")
        self.assertGreater(len(conv_keys), 0)

    def test_weights_transfer_and_outputs_match(self):
        conv = build_model("conv", seed=0)
        rec = build_model("recurrent", seed=1)  # 故意用不同初始化

        rec.load_state_dict(conv.state_dict(), strict=True)
        conv.prepare_for_inference()
        rec.prepare_for_inference()

        x = torch.randn(2, 50, 1)
        with torch.no_grad():
            y_conv = conv(x)
            y_rec = rec(x)

        self.assertEqual(tuple(y_conv.shape), tuple(y_rec.shape))
        self.assertTrue(
            torch.allclose(y_conv, y_rec, atol=1e-5, rtol=1e-4),
            f"max diff = {(y_conv - y_rec).abs().max().item():.3e}",
        )

    def test_load_fp_checkpoint_refreshes_stale_cache(self):
        """直接 load_state_dict 会留下旧缓存；load_fp_checkpoint 必须修掉这个问题。"""
        conv = build_model("conv", seed=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "best.pth")
            torch.save({"model": conv.state_dict()}, path)

            # 先 eval 再 load，且不刷新：缓存来自随机初始化
            stale = build_model("recurrent", seed=3)
            stale.load_state_dict(conv.state_dict(), strict=True)
            x = torch.randn(2, 32, 1)
            with torch.no_grad():
                y_conv = conv(x)
                y_stale = stale(x)

            self.assertGreater(
                (y_conv - y_stale).abs().max().item(),
                1e-4,
                "预期：不刷新缓存时输出明显不一致（这是被测试保护的那个坑）",
            )

            # 走统一加载入口：load -> eval -> prepare_for_inference
            fresh = build_model("recurrent", seed=4)
            fresh, _ = load_fp_checkpoint(fresh, path)
            with torch.no_grad():
                y_fresh = fresh(x)

            self.assertTrue(
                torch.allclose(y_conv, y_fresh, atol=1e-5, rtol=1e-4),
                f"max diff = {(y_conv - y_fresh).abs().max().item():.3e}",
            )
            self.assertIsNotNone(fresh.backbone.s4_layers[0].kernel.cached_A_bar)


if __name__ == "__main__":
    unittest.main(verbosity=2)
