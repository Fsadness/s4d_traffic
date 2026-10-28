# archive

这里存放**不再参与运行**的上游代码与历史入口，保留是为了可追溯与随时取回（git 历史仍在）。

| 文件 | 来源 | 说明 |
| --- | --- | --- |
| `example.py` | state-spaces/s4 | 序列 CIFAR / MNIST 的上游 demo，依赖 `models/layers/s4.py` |
| `example_pretrain.py` | 本仓库改自上游 | 上述 demo 的掩码重建变体 |
| `example_test.py` | 本仓库改自上游 | 上述 demo 的评估变体 |
| `train_full.py` | state-spaces/s4 | 上游 PyTorch-Lightning + Hydra 训练入口；本环境缺 hydra/lightning，无法运行 |
| `README_old.md` | state-spaces/s4 | 上游 README，已被根目录 `README.md` 取代 |
| `main.py` | uv 初始化占位 | 只打印 `Hello from s4d!`，无任何调用方 |

## 取回方式

```bash
git mv archive/example.py .            # 单个文件
# 或从历史中查看
git log --oneline -- archive/example.py
```

## 注意

- `archive/` 下的文件**不在任何入口的 import 链上**，不要把它们当作可运行脚本；
  如需运行，需自行修正其对 `models.layers.s4` / `models.layers.s4d` 的 import 路径。
- 本目录不收录 `src/`、`extensions/`、`models/layers/s4.py`、`models/layers/s4d.py`：
  按约定，其他目录保留原有组织逻辑，不做迁移。
