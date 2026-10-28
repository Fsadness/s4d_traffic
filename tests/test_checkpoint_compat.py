"""护栏测试：历史实验的 `best.pth` 必须能被当前模型严格加载并完成前向。

覆盖两类 run：
- 旧 config（没有 tokenizer 字段）→ 按 pixel 处理
- 新 config（patch / compressed_patch 等）→ 由 `datasets.specs` 推导 d_input

运行：
    python tests/test_checkpoint_compat.py
    python tests/test_checkpoint_compat.py experiments/<...>/best.pth
    python -m unittest tests.test_checkpoint_compat -v
"""

import argparse
import glob
import os
import sys
import unittest

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.specs import infer_d_input  # noqa: E402
from engine.checkpoint import load_fp_checkpoint  # noqa: E402
from models.networks.s4d_classification import Model  # noqa: E402


def discover_checkpoints(limit=3):
    paths = sorted(glob.glob(os.path.join(ROOT, "experiments", "*", "*", "*", "best.pth")))
    return paths[:limit]


def load_case(checkpoint_path):
    exp_dir = os.path.dirname(checkpoint_path)
    with open(os.path.join(exp_dir, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = argparse.Namespace(**yaml.safe_load(f))

    if not hasattr(cfg, "tokenizer"):
        cfg.tokenizer = "pixel"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    d_output = int(checkpoint["model"]["decoder.weight"].shape[0])

    model = Model(
        d_input=infer_d_input(cfg),
        d_output=d_output,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        prenorm=cfg.prenorm,
        lr=cfg.lr,
    )
    return cfg, model, d_output


class CheckpointCompatibilityTest(unittest.TestCase):
    def test_historical_checkpoints_load_and_forward(self):
        checkpoints = discover_checkpoints()
        if not checkpoints:
            self.skipTest("experiments/ 下没有可用的 best.pth")

        for path in checkpoints:
            with self.subTest(checkpoint=os.path.relpath(path, ROOT)):
                cfg, model, d_output = load_case(path)

                # strict=True：键名或形状不一致会直接失败
                model, _ = load_fp_checkpoint(model, path, strict=True)

                d_input = infer_d_input(cfg)
                x = torch.zeros(2, 16, d_input)
                with torch.no_grad():
                    logits = model(x)

                self.assertEqual(tuple(logits.shape), (2, d_output))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertIsNotNone(
                    model.backbone.s4_layers[0].kernel.cached_A,
                    "加载后必须重建推理缓存",
                )


if __name__ == "__main__":
    explicit = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if explicit:
        # 允许直接指定 checkpoint
        for checkpoint_path in explicit:
            cfg, model, d_output = load_case(checkpoint_path)
            model, _ = load_fp_checkpoint(model, checkpoint_path, strict=True)
            with torch.no_grad():
                logits = model(torch.zeros(2, 16, infer_d_input(cfg)))
            print(f"OK  {checkpoint_path}  out={tuple(logits.shape)}")
        sys.exit(0)

    unittest.main(verbosity=2)
