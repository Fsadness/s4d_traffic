# S4D 网络流量分类

基于 **S4D（Diagonal Structured State Space Model，对角结构化状态空间模型）** 的网络流量分类实现：把流量样本渲染成 40×40 的灰度图像，按不同 tokenizer 展开为序列，用 S4D 层（FFT 卷积核 / 循环核）建模，最后接线性分类头。

仓库同时包含一条**掩码重建预训练**支线，以及一份 state-spaces/s4 上游代码快照（`src/`、`extensions/`、`example*.py`、`models/layers/s4.py`），用于复用其中的 `DropoutNd` 等基础模块与对照参考。

---

## 目录

- [环境](#环境)
- [数据组织](#数据组织)
- [快速开始](#快速开始)
- [模型结构](#模型结构)
- [Tokenizer](#tokenizer)
- [训练机制与产物](#训练机制与产物)
- [评估](#评估)
- [代码结构](#代码结构)
- [已知限制](#已知限制)
- [上游代码与引用](#上游代码与引用)

---

## 环境

- WSL2 Ubuntu（代码与 `.venv` 均为 Linux 环境，需在 WSL 内运行）
- Python 3.10（见 `.python-version`；`pyproject.toml` 约束 `>=3.10,<3.11`）
- PyTorch 1.13.1 + cu116 / torchvision 0.14.1
- 依赖由 [uv](https://docs.astral.sh/uv/) 管理，虚拟环境位于 `.venv/`

```bash
cd /mnt/c/Users/Fsadness/Desktop/models/research/s4d

# 首次搭建（.venv 已存在时可跳过前两行）
uv venv --python 3.10
source .venv/bin/activate
uv pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116
uv pip install -r requirements.txt
```

说明：

- `requirements.txt` 由 `uv export --format requirements-txt` 生成，**不包含 torch**，PyTorch 需按上面的命令单独安装。
- `pykeops` 仅服务于上游 legacy 的 `models/layers/s4.py`（完整版 S4 的 Cauchy / Vandermonde 核），当前 S4D 主路径不依赖它。
- 启动时若打印 `CUDA extension for structured kernels ... not found`，它来自上游 `models/layers/s4.py` 的可选 CUDA 扩展探测，**不影响** 本仓库的 S4D 训练与评估，无需编译 `extensions/kernels`。

---

## 数据组织

数据以 `torchvision.datasets.ImageFolder` 的目录约定组织：

```text
<data_dir>/
└── <dataset>/
    ├── dataset_sampled/
    │   ├── train/<class>/*.png
    │   ├── valid/<class>/*.png
    │   └── test/<class>/*.png
    └── train/<class>/*.png          # 仅预训练模式使用
```

- `--data_dir` 是数据集根目录，脚本默认值 `/mnt/e/datasets/NetMamba`，请按实际位置覆盖。
- 图像在 `datasets/transforms.py` 中被强制转为单通道灰度、归一化到 `[-1, 1]`；尺寸由图片本身决定（`--img_size` 是保留参数，当前流程不参与缩放）。
- 除 `pixel` 之外的 tokenizer 依赖 40×40 输入，见下一节。
- 预训练模式（`--mode pretrain`）只读取 `<data_dir>/<dataset>/train`，不使用 `dataset_sampled`。

本机已验证的数据集规模（类别数 / train / valid / test，来自 `dataset_sampled`）：

| 数据集 | 类别数 | train | valid | test |
| --- | ---: | ---: | ---: | ---: |
| CICIoT2022 | 6 | 15597 | 1949 | 1951 |
| CrossPlatform-Android | 181 | 37244 | 4655 | 4656 |
| CrossPlatform-iOS | 124 | 33630 | 4203 | 4205 |
| ISCXVPN2016 | 7 | 11065 | 1383 | 1384 |
| ISCXTor2016 | 8 | 11655 | 1456 | 1458 |
| USTC-TFC2016 | 19 | 25902 | 3237 | 3239 |

> `datasets/dataloader.py` 中的 `SafeImageFolder` 会在启动时用 `PIL.Image.verify()` 过滤损坏文件，因此首次进入训练前会有一段时间的样本校验开销。

---

## 快速开始

```bash
export S4D_DATA=/home/fsadness/datasets/NetMamba   # 按实际数据根目录修改
source .venv/bin/activate
```

### 1. 训练（微调）

```bash
python train_log_patch.py -c finetune.yaml \
    --dataset CICIoT2022 \
    --tokenizer pixel \
    --data_dir "$S4D_DATA" \
    --save_dir ./experiments/new
```

`-c finetune.yaml` 读取配置，命令行参数优先级更高（见 [配置优先级](#配置优先级)）。产物写入
`<save_dir>/<model>/<YYYYmmdd_HHMMSS>/`。

### 2. 断点续训

```bash
python train_log_patch.py --resume experiments/new/s4d_classification/<timestamp>/latest.pth
```

`--resume` 会定位 checkpoint 所属的实验目录（若路径位于 `checkpoints/` 子目录，则取其上一级），自动读取其中的 `config.yaml`，并恢复模型、优化器、调度器、最佳精度、early-stop 计数器与起始 epoch。

### 3. 评估

```bash
python test.py <save_dir>/<model>/<timestamp>
```

结果写入 `<exp_dir>/evaluation/`。加 `--kernel recurrent` 会用循环核重算一遍，写入 `evaluation_recurrent/`：

```bash
python test.py <save_dir>/<model>/<timestamp> --kernel recurrent
```

### 4. 预训练（掩码重建）

```bash
python train_log_patch.py -c pretrain.yaml --mode pretrain \
    --data_dir "$S4D_DATA" \
    --save_dir ./experiments/pretrain
```

预训练用 50% 随机掩码，只在被掩码位置计算 MSE 损失，产物为 `best.pth` / `latest.pth`（不含分类头）。

### 5. 用预训练权重初始化分类模型

```bash
python train_log_patch.py -c finetune.yaml \
    --dataset CICIoT2022 --tokenizer pixel \
    --data_dir "$S4D_DATA" \
    --pretrained_path experiments/pretrain/s4d_reconstruction/<timestamp>/best.pth \
    --save_dir ./experiments/new
```

加载时会丢弃 `head` / `decoder` / `reconstruction_head` 相关权重。

---

## 模型结构

```text
输入 (B, L, d_input)
  └─ Linear(d_input → d_model)                          # S4DBackbone.encoder
     └─ × n_layers:                                     # 残差 + LayerNorm
          S4DLayer(d_model)
            ├─ kernel: S4DConvKernel 或 S4DRecurrentKernel
            ├─ D 项（跳连）
            ├─ GELU + DropoutNd
            └─ Conv1d(h → 2h, k=1) + GLU                # 逐点特征混合
     └─ mask 分支（预训练）/ mean-pool（分类）
        └─ Linear(d_model → d_output)
```

**S4D 卷积核**（训练与默认推理路径）：

- 参数：`log_dt`、`log_A_real`、`A_imag`、`C`
- `A = -exp(log_A_real) + j·A_imag`，`dt = exp(log_dt)`
- 由 Vandermonde 乘法得到卷积核 `K`，再用 `rfft` 做 FFT 卷积
- `S4D-Lin` 初始化：`A_real = -0.5`，`A_imag = πn`

**S4D 循环核**（流式/逐步推理）：ZOH 离散化 `Ā = exp(A·dt)`、`B̄ = (exp(A·dt) − 1)/A`，按 `x_t = Ā·x_{t−1} + B̄·u_t` 递推，输出 `y_t = 2·Re{Σ C·x_t}`。两种核在数学上等价，仅实现与耗时不同。

模型变体（`models/networks/`，训练脚本按 `--model` 动态导入同名模块的 `Model` 类）：

| 模块 | 用途 | 输出 |
| --- | --- | --- |
| `s4d_classification` | 主分类模型（卷积核，可切换 `kernel_type='recurrent'`） | `(B, d_output)` logits |
| `s4d_reconstruction` | 掩码重建预训练 | `(B, L, d_input)` 重建序列 |
| `s4d_recurrent_classification` | 流式/早退分类实验支线（逐步 `forward_step`、平均池化、可插早停策略） | `(logits, info_dict)` |

默认规模 `d_model=128, n_layers=4`，在 6 类任务上约 20 万参数。

---

## Tokenizer

`datasets/tokenizer.py` 定义序列化方式，`datasets/transforms.py` 按 `--tokenizer` 组装。以 40×40 输入为例：

| `--tokenizer` | 序列长度 L | `d_input` | 说明 |
| --- | ---: | ---: | --- |
| `pixel` | 1600 | 1 | 逐像素展平，默认方式 |
| `patch` | 100（p=4） | p² = 16 | 按 `patch_size` 不重叠切块 |
| `compressed_patch` | 100（p=4） | c² = 9 | 切块后 `adaptive_avg_pool1d` 压到 `compressed_size²` |
| `packet` | 320 | 5 | 按 5 个包 × 320 字节重排（`packet_count=5, packet_size=320` 固定） |
| `log_compressed_packet` | 146（segment=64） | 5 | 分段落池化压缩，段长按序号递减 |

约束：

- `patch` 需让 `patch_size` 整除图像边长，否则边缘像素会被丢弃；
- `compressed_patch` 建议 `compressed_size ≤ patch_size`；
- `packet` / `log_compressed_packet` 要求图像像素数恰为 `5 × 320 = 1600`，即 40×40；
- 训练脚本根据 tokenizer 自动推导 `d_input`（`pixel→1`、`patch→p²`、`compressed_patch→c²`、packet 系→5）。

---

## 训练机制与产物

### 训练脚本

| 脚本 | 说明 |
| --- | --- |
| `train_log_patch.py` | **当前主入口**：配置加载、结构化日志、tokenizer、早停、最佳/最新 checkpoint |
| `train_log.py` | 上一代脚本：有日志与早停，但没有 tokenizer 选项（固定 pixel） |
| `train.py` | 最早版本：无 `train.log`、无早停 |

三者共享同一套 `datasets/`、`models/`、`optim/` 代码；新实验请使用 `train_log_patch.py`。

### 配置优先级

1. `--resume` 指定的实验目录中的 `config.yaml`（续训时自动加载）；
2. `-c <file>` 指定的配置文件；
3. 命令行参数。

配置**只会覆盖仍处于 parser 默认值的参数**，因此命令行显式给出的值始终优先。例：

```bash
python train_log_patch.py -c finetune.yaml --dataset CICIoT2022 --tokenizer patch --patch_size 4
```

### 优化器与调度

- `AdamW`，默认 `lr=1e-3`、`weight_decay=0.01`；
- S4D 的 SSM 参数（`log_dt` / `log_A_real` / `A_imag` / `C`）单独成组：`weight_decay=0`，`lr=min(1e-3, --lr)`；
- 学习率：`CosineAnnealingLR(T_max=epochs)`，按 epoch 步进；
- 损失：分类用 `CrossEntropyLoss`，预训练用 `MSELoss`（仅掩码位置）。

### 输出目录

```text
<save_dir>/<model>/<YYYYmmdd_HHMMSS>/
├── config.yaml            # 本次运行的完整参数快照
├── train.log              # 结构化训练日志（时间戳 | 指标）
├── best.pth               # 验证集最优（预训练为训练损失最优）
├── latest.pth             # 每个 epoch 覆盖
└── checkpoints/
    └── epoch_XXX.pth      # 每 --save_freq 个 epoch 的备份
```

checkpoint 字典字段：`model`、`optimizer`、`scheduler`、`acc`、`epoch`，`train_log_patch.py` 额外保存 `early_stop_counter`；预训练阶段的 checkpoint 只保存 `model` 与 `epoch`。

### 早停

`--use_early_stop` 开启，`--early_stop_patience`（默认 10）、`--early_stop_delta`（默认 0.001）控制。指标为验证集准确率；连续 `patience` 个 epoch 未提升 `delta` 即停止，并保留 `best.pth`。

### 参考耗时

GTX 1650（4 GB）上，CICIoT2022 + `pixel`：单 epoch（15597 训练样本、batch 64）约 1.5 分钟，验证/测试各数秒；进程启动时另有约 1 分钟的样本校验开销（见 `SafeImageFolder`）。

---

## 评估

```bash
python test.py <exp_dir> [--batch_size N] [--num_workers N] [--device cuda|cpu] [--kernel conv|recurrent]
```

`test.py` 只加载测试集（跳过 train/valid，避免重复校验），输出：

```text
<exp_dir>/evaluation/            # 或 evaluation_recurrent/
├── metrics.json                 # 总体 + 逐类指标、延迟、吞吐、平均置信度/margin
├── classification_report.csv    # 逐类 P/R/F1 + macro/weighted 平均
└── confusion_matrix.png         # 行归一化混淆矩阵（类别多时自动降低标注密度）
```

指标包含：Accuracy、Macro/Weighted Precision/Recall/F1、平均交叉熵、平均置信度、置信度 margin、单样本平均延迟（ms/sample）与吞吐（samples/s）。延迟为单次前向的实测值，首次运行含预热影响。

---

## 代码结构

```text
.
├── train_log_patch.py       # 训练主入口
├── train_log.py             # 训练入口（旧）
├── train.py                 # 训练入口（最早）
├── test.py                  # 评估入口
├── finetune.yaml            # 微调配置（ISCXVPN2016，含早停）
├── finetune_no_earlystop.yaml  # 微调配置（关闭早停）
├── pretrain.yaml            # 预训练配置
├── datasets/
│   ├── dataloader.py        # ImageFolder 封装 + SafeImageFolder + DataLoader 构建
│   ├── transforms.py        # 灰度/归一化 + tokenizer 组装
│   └── tokenizer.py         # 五种序列化方式
├── models/
│   ├── kernels/             # S4D 参数初始化、离散化、卷积核/循环核
│   ├── layers/              # S4DLayer；s4.py / s4d.py 为上游参考实现
│   ├── backbones/           # S4DBackbone / S4DRecurrentBackbone
│   └── networks/            # 分类、重建、流式分类三个 Model
├── optim/optimizer.py       # AdamW + SSM 参数分组 + CosineAnnealing
├── engine/trainer.py        # 面向对象的 Trainer 封装（结构相近，未被入口使用）
├── src/                     # 上游 state-spaces/s4 代码快照（运行时只用 src.models.nn）
├── extensions/kernels/      # 上游可选 CUDA 扩展（当前 S4D 路径不需要）
├── example.py               # 上游 demo：序列 CIFAR / MNIST
├── example_pretrain.py      # 上游 demo 的掩码重建变体
├── example_test.py          # 上游 demo 的评估变体
├── train_full.py            # 上游 PyTorch-Lightning + Hydra 训练入口（本仓库未使用）
├── pyproject.toml / uv.lock / requirements.txt
└── experiments/             # 训练产物（未纳入版本控制）
```

> `engine/trainer.py` 是训练循环的另一种面向对象封装，结构与主脚本相近但未被任何入口调用，且预训练损失的计算方式与主脚本不同；实际使用请以 `train_log_patch.py` 为准。

---

## 已知限制

- **单卡、小显存**：本仓库在 4 GB 显卡上验证；显存不足时优先降 `--batch_size` 或 `--d_model`。
- **训练只走卷积核**：`train_log_patch.py` 未暴露 `kernel_type`，训练固定使用 `S4DConvKernel`；循环核用于 `test.py --kernel recurrent` 与流式推理模型。
- **`img_size` 不参与缩放**：序列长度由图片实际尺寸与 tokenizer 决定，`--img_size` 目前仅为兼容保留。
- **预训练数据缺位**：`pretrain.yaml` 指向 `pretrain_dataset`，需自行准备 `<data_dir>/pretrain_dataset/train/<class>/*.png`。
- **循环核为逐时间步 Python 循环**：长序列（如 L=1600）推理明显慢于 FFT 卷积核，仅适合流式/早退场景。
- **`SafeImageFolder` 启动校验**：样本越多启动越慢；数据确认无损时可自行绕过。

---

## 上游代码与引用

`src/`、`extensions/`、`example*.py`、`train_full.py`、`models/layers/s4.py`、`models/layers/s4d.py` 来自
[state-spaces/s4](https://github.com/state-spaces/s4)（S4 / S4D / SaShiMi / S4ND 官方实现），保留用于参考与复用。
本仓库的 S4D 主路径（`models/kernels/`、`models/layers/s4d_layer.py`）是在其基础上拆分、重写得到的。

如果使用本仓库或上游代码，请引用：

```bibtex
@inproceedings{gu2022efficiently,
  title={Efficiently Modeling Long Sequences with Structured State Spaces},
  author={Gu, Albert and Goel, Karan and R\'e, Christopher},
  booktitle={The International Conference on Learning Representations ({ICLR})},
  year={2022}
}

@inproceedings{gu2022s4d,
  title={On the Parameterization and Initialization of Diagonal State Space Models},
  author={Gu, Albert and Gupta, Ankit and Goel, Karan and R\'e, Christopher},
  booktitle={Advances in Neural Information Processing Systems ({NeurIPS})},
  year={2022}
}
```
