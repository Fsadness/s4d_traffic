# s4d 代码库分析与整理报告

- 仓库：`C:\Users\Fsadness\Desktop\models\research\s4d`（WSL 路径 `/mnt/c/Users/Fsadness/Desktop/models/research/s4d`）
- 分析日期：2026-10-28（初版）；**整理进展更新：2026-10-28（第二轮，已删文件已复核）**
- 运行环境：WSL2 Ubuntu + uv 虚拟环境 `.venv`（Python 3.10.20，torch 1.13.1+cu116），GPU：GTX 1650 4 GB
- 分析方式：通读全部自研代码 + 依赖导入/前向实测 + 用现有 checkpoint 跑通评估 + 1 epoch 训练冒烟 + 数据集体检
- 当前状态：初版分析后用户已清理第一批临时文件（**≈ 298 MiB**），仓库从 **≈ 798 MiB 降到 ≈ 500 MiB**；剩余可清理项约 **390 MiB**（主要是 159 个 epoch 备份），清理完约 **110 MiB**。整理过程中**未改动任何代码**。

---

## 一、仓库定位与整体结构

这是一个**基于 S4D 的网络流量分类研究仓库**：把流量样本渲染成 40×40 灰度图，用不同 tokenizer 展开成序列，经 S4D（对角状态空间模型）骨干分类。它的形成方式分三层：

1. **上游快照**：`src/`、`extensions/`、`example*.py`、`train_full.py`、`models/layers/s4.py`、`models/layers/s4d.py`、`README_old.md` 来自 [state-spaces/s4](https://github.com/state-spaces/s4)，基本原样保留；
2. **自研主体**：`datasets/`、`models/kernels/`、`models/layers/s4d_layer.py`、`models/backbones/`、`models/networks/`、`optim/`、`train*.py`、`test.py` 是在上游基础上拆出/重写的主线代码；
3. **实验产物**：`experiments/`（23 个 run，未纳入版本控制）与 `checkpoint/`（内容已清空，仅剩空目录）。

git 状态：`master` 分支，6 个提交，182 个已跟踪文件；当前工作区 **10 个已修改、5 个未跟踪**（`git status` 共 15 项）。初版分析时有 12 个未跟踪项，其中 8 个临时文件已随本次整理删除，剩下的是正式代码（`engine/`、`models/backbones/s4d_recurrent_backbone.py`、`models/networks/s4d_recurrent_classification.py`）、待决的 `test_inference.py` 与本报告。

体积账本（不含 `.venv` 3.9 GiB 与 `.git`，**为清理后的当前值**）：

| 位置 | 大小 | 性质 |
| --- | ---: | --- |
| `experiments/` | 498.3 MiB | 训练产物（其中 `checkpoints/epoch_*.pth` 仍占 388.4 MiB） |
| `src/` | 1.2 MiB | 上游代码快照 |
| 其余代码/配置/文档 | ≈ 0.4 MiB | 持久内容 |
| `checkpoint/`（空目录） | 0 B | 内容已删，目录待移除 |
| **合计** | **≈ 500 MiB** | 初版为 798 MiB |

---

## 二、代码功能分析

### 2.1 数据管线（`datasets/`）

| 文件 | 职责 |
| --- | --- |
| `dataloader.py` | `build_dataloader(args)` 按 `mode` 构建 DataLoader；`SafeImageFolder` 继承 `ImageFolder`，用 `PIL.Image.verify()` 逐个校验样本并剔除损坏文件 |
| `transforms.py` | `build_transform(args)`：`Grayscale(1)` → `ToTensor` → `Normalize((0.5,),(0.5,))` → 按 `--tokenizer` 追加对应 tokenizer |
| `tokenizer.py` | 5 种序列化方式：`PixelTokenizer` / `PatchTokenizer` / `CompressedPatchTokenizer` / `PacketTokenizer` / `LogCompressedPacketTokenizer` |

数据目录约定（`ImageFolder` 风格）：

```text
<data_dir>/<dataset>/dataset_sampled/{train,valid,test}/<class>/*.png   # 微调
<data_dir>/<dataset>/train/<class>/*.png                                # 预训练
```

样本经变换后是 `(L, d_input)`，默认 collate 后为 `(B, L, d_input)`。实测 40×40 输入下各 tokenizer 的输出：

| tokenizer | 输出形状 | 含义 |
| --- | --- | --- |
| `pixel` | `(1600, 1)` | 逐像素 |
| `patch` (p=4) | `(100, 16)` | 100 个 4×4 块 |
| `compressed_patch` (p=4,c=3) | `(100, 9)` | 块内压到 9 维 |
| `packet` | `(320, 5)` | 5 包 × 320 字节，转置成 320 步 × 5 通道 |
| `log_compressed_packet` (seg=64) | `(146, 5)` | 分 5 段、段长递减（64/32/22/16/12） |

### 2.2 模型（`models/`）

**核（`models/kernels/`）** —— 本次提交新拆出的核心：

- `base.py`：`S4DBaseKernel` 持有 `log_dt / log_A_real / A_imag / C` 四个参数；
- `init.py`：S4D-Lin 初始化（`A_real=-0.5`、`A_imag=πn`）；
- `discretize.py`：`materialize_continuous_params` 还原 `A、dt、C`；`discretize` 做 ZOH；
- `s4d_conv_kernel.py`：`K = 2·Re{Σ C·(e^{dtA}−1)/A · e^{dtA·t}}`，再 `rfft/irfft` 做 FFT 卷积，复杂度 O(L log L)；
- `s4d_recurrent_kernel.py`：逐步递推 `x_t = Āx_{t−1} + B̄u_t`，`y_t = 2Re{Σ C x_t}`，Python 循环 O(L)；
- `utils.py`：`register_param`（挂 `_optim` 钩子，供优化器分组）、`register_cache`（非持久 buffer）。

**层（`models/layers/s4d_layer.py`）**：`S4DLayer` 按 `kernel_type` 在 conv / recurrent 核之间分派，含 `D` 跳连、GELU、`DropoutNd`、`Conv1d(h→2h)+GLU` 输出混合；`prepare_for_inference()` 预离散化参数以避免推理时重复计算；循环核额外提供 `reset_state` / `forward_step`。

**骨干与网络**：

- `backbones/s4d_backbone.py`：`Linear(d_input→d_model)` → `n_layers × [S4DLayer + Dropout + 残差 + LayerNorm]`；
- `backbones/s4d_recurrent_backbone.py`：同上但逐步前向，维护逐层循环状态；
- `networks/s4d_classification.py`：骨干 + 序列均值池化 + 线性分类头；`train(False)` 时自动 `prepare_for_inference()`；
- `networks/s4d_reconstruction.py`：骨干 + `Linear(d_model→d_input)` 逐 token 重建；
- `networks/s4d_recurrent_classification.py`：流式分类，维护池化和、活跃掩码、停止步数，预留 `early_stop_strategy` 接口，返回 `(logits, info)`。

实测参数量：默认配置（`d_model=128, n_layers=4`，6 类）为 **200,710**，与 `best.pth` 一致。

### 2.3 训练脚本的演进（4 个入口）

| 脚本 | 状态 | 差异 |
| --- | --- | --- |
| `train.py` | 可用（旧） | 无 `train.log`、无早停；`d_input` 硬编码为 1 |
| `train_log.py` | 可用（过渡） | 加 logging 与早停；仍固定 pixel tokenizer |
| `train_log_patch.py` | **当前主入口** | 增加 `--tokenizer/--patch_size/--compressed_size/--log_compressed_segment` 与 `d_input` 推导；`checkpoint` 多存 `early_stop_counter` |
| `train_full.py` | **不可运行** | 上游 PL + Hydra 入口，`hydra` / `pytorch_lightning` 未安装（实测 ImportError），已脱离主线 |

共同机制：

- 优化器 `AdamW`（`lr=1e-3, wd=0.01`），SSM 参数单独分组（`wd=0`，`lr=min(1e-3, --lr)`）；
- `CosineAnnealingLR(T_max=epochs)`；
- 训练/验证/测试三集循环，验证集最优存 `best.pth`，每 epoch 覆盖 `latest.pth`，每 `--save_freq` 存 `checkpoints/epoch_XXX.pth`；
- 早停：验证准确率、`patience` + `delta`；
- 配置优先级：`--resume` 目录的 `config.yaml` > `-c` 指定配置 > 命令行；且**配置只覆盖仍为默认值的参数**。

### 2.4 评估（`test.py` / `test_inference.py`）

`test.py <exp_dir>` 只加载 test 集，重建模型（`--kernel conv|recurrent`）、加载 `best.pth`，输出 `evaluation/` 或 `evaluation_recurrent/` 下的 `metrics.json`、`classification_report.csv`、`confusion_matrix.png`；指标含 Accuracy、Macro/Weighted P/R/F1、平均 CE、平均置信度、置信度 margin、单样本延迟、吞吐，并对大类别数做了绘图内存保护。

`test_inference.py` 是另一个分支：固定用 `s4d_recurrent_classification`（流式、逐步）评估，产物写 `inference/`，与 `test.py` 有大量重复代码，是循环核/早退研究的一次性脚本。

### 2.5 上游遗留与未接入代码

- `src/`（128 个文件）：运行时**只用到 `src.models.nn` 的 `DropoutNd`**（被 `models/layers/s4d.py`、`s4d_layer.py` 引用；原先的 `s4d_layer_new.py` 已删），其余为上游参考；
- `extensions/kernels/`：仅被 `models/layers/s4.py` 的可选 CUDA 核引用，当前 S4D 路径不需要（因此启动时的 “CUDA extension … not found” 警告与主线无关）；
- `models/layers/s4.py`（1960 行）、`s4d.py`：上游完整/极简实现，被 `train*.py`、`backbones/*`、`example*.py` import，但主路径**从未实例化**（`S4Block`、`S4D` 是纯未使用 import）；
- `example.py` / `example_pretrain.py` / `example_test.py`：上游 CIFAR/MNIST demo；
- `main.py`：`uv init` 占位（打印 “Hello from s4d!”）；
- `engine/trainer.py`：训练循环的另一种面向对象封装，结构与主脚本相近但**没有任何入口调用**，且 checkpoint 少了 `scheduler` 字段、预训练损失与主脚本不同。

### 2.6 实测验证记录

| 验证项 | 命令/方式 | 结果 |
| --- | --- | --- |
| 依赖与导入 | 逐个 `import` 12 个主线模块 | 全部 OK；`s4d_layer_new`（已删）、`train_full` 失败（见第四节） |
| 前向形状 | conv/recurrent/重建三个 Model | `(2,6)` / `(2,6)` / `(2,1600,1)` |
| tokenizer | 40×40 输入 | 与 2.1 表格一致 |
| 评估链路 | `test.py` 跑历史 checkpoint（CPU） | 通过，Accuracy 98.36%（该 run 记录 98.31%，属正常数值差异） |
| 训练链路 | `train_log_patch.py -c finetune.yaml`，CICIoT2022/pixel，1 epoch | 通过：epoch 循环 115.9 s（另有约 1 分钟启动校验），Train Acc 42.72%、Val 54.28%、Test 53.31%，正常落盘 |
| 数据集 | 遍历 `dataset_sampled` | 6 个数据集结构完整（规模见 README） |

---

## 三、清理进展与剩余清单

判定依据：① 是否被任何代码/配置引用；② 是否可由脚本再生成；③ 是否在 git 中；④ 时间戳与所在目录语义。

### 3.1 已完成清理（2026-10-28，用户操作，已逐项复核）

| 路径 | 原大小 | 判定依据 |
| --- | ---: | --- |
| `android_info.csv` | 175.7 MiB | 1600 行 × 4607 列浮点 dump；全仓库代码/配置/文档**零引用**；生成脚本不在仓库 |
| `android_infomean.csv` | 39 KiB | 同一批次的 1600 行单列均值中间量，零引用 |
| `experiments/packet.zip` | 88.2 MiB | `experiments/packet/` 的完整打包（69 项），目录本身仍在，纯重复 |
| `experiments/patch/log_*.txt`（6 个） | 22.7 MiB | 单次实验的原始 tqdm 输出（`log_patch_p2/p5/p8`、`log_cpatch_c1/c2/c3`），与各 run 的 `train.log` 重复 |
| `experiments/new/queue_cic_no_es.log` | 7.2 MiB | 训练排队守护脚本的 stdout，含大量 GBK 乱码，仅过程记录 |
| `checkpoint/` 内 4 个 `.pth` | 3.2 MiB | `ckpt.pth / ckpt1 / ckpt_test / ckpt_test1`，早期 `train.py` 默认 `save_dir` 的临时权重，已被 `experiments/` 取代 |
| `models/layers/s4d_layer_new.py` | 5.5 KiB | import 不存在的 `models.kernels.kernel_fns`，**导入即 ImportError**（已实测），是未完成的薄壳重构 |
| `dev.py` | 2.0 KiB | 一次性对拍脚本：给 `s4d_classification.Model` 传不存在的 `training=` 参数会 **TypeError**（已实测），且硬编码历史 run 路径 |
| `dev2.py` | 399 B | 一次性脚本：把 `best.pth` 的 `backbone.*` 拆成 `backbone.pth` |
| `backbone.pth` | 792 KiB | `dev2.py` 的产物，仅该脚本使用 |
| `OPTIMIZATION_PLAN.md` | 7.7 KiB | 早期优化计划，Phase 1/2/3 均已落地，文档过期 |
| `run_android_experiments.sh` | 1 KiB | `--save_dir ./experiment/finetune`（单数）指向不存在的目录，需修正后才可用 |

**小计 ≈ 298 MiB 已释放**：`experiments/` 616.5 → 498.3 MiB，仓库 798 → 500 MiB。复核结果：上述路径均已不存在；`experiments/packet/`、`experiments/new/` 等正式产物完好；`test_inference.py`、`2025old/`、epoch 备份均未动。

> 唯一的残留：`checkpoint/` 目录被保留了下来，但已为空（0 B），可直接 `rmdir`。

### 3.2 剩余可清理项（≈ 390 MiB）

| 路径 | 大小 | 建议 |
| --- | ---: | --- |
| `experiments/*/checkpoints/epoch_*.pth`（159 个） | 388.4 MiB | 周期性备份；`best.pth` + `latest.pth` 已能覆盖复现与续训。保留需要续训的 run 的 `latest.pth`，其余 `checkpoints/` 清空 |
| `experiments/pretrain/s4d_reconstruction/2025old/s4_traffic_pretrain.pth` | 796 KiB | 孤儿权重：无 `config.yaml`、无 `train.log`，无法追溯来源，建议归档 |
| `experiments/patch/.../20260602_160857/evaluation_old`、`evaluation_conv_old`、`evaluation_s4d`、`evaluation_recurrent` | ≈ 0.4 MiB | 同一 run 的历史评估副本，命名已被 `evaluation/` + `--kernel` 取代 |
| `__pycache__/`（10 处） | 0.19 MiB | 可再生成 |
| `checkpoint/`（空目录） | 0 B | 直接删除 |
| `test_inference.py` | 16 KiB | 流式评估分支，与 `test.py` 高度重复、依赖 experimental 模型；若继续早退研究可正式纳入版本控制，否则归档 |

**小计 ≈ 390 MiB**；全部处理后期望仓库 ≈ **110 MiB**。

### 3.3 C 级：必须保留（持久内容）

- 代码：`datasets/`、`models/`（`s4d_layer_new.py` 已删）、`optim/`、`engine/`、`train*.py`、`test.py`；
- 配置：`finetune.yaml`、`finetune_no_earlystop.yaml`、`pretrain.yaml`；
- 依赖与环境：`pyproject.toml`、`uv.lock`、`requirements.txt`、`.python-version`、`.gitignore`；
- 文档：`README.md` 与本报告；
- 环境：`.venv/`（3.9 GiB，可重建但重装 torch 成本高，不要删）；
- 建议保留的产物：23 个 run 的 `best.pth` + `config.yaml` + `train.log` + `latest.pth`（约 110 MiB，是唯一能证明指标的结果）。

### 3.4 D 级：已跟踪的“历史包袱”，本次不动

`src/`（仅 `src.models.nn` 被用到）、`extensions/`、`example*.py`、`train_full.py`、`models/layers/s4.py`、`models/layers/s4d.py`、`train.py`、`train_log.py`、`main.py`。

它们都在 git 历史里、且被新 README 明确标注了用途（上游参考/旧入口）。删除会改动 import 链（例如 `DropoutNd` 就在 `src/models/nn`），**本次不建议动**；若将来要裁剪 `src/`，需同步把 `DropoutNd` 内联到 `models/`。

---

## 四、代码层面的观察（仅记录，未修改）

1. **`models/layers/s4d_layer_new.py` 无法导入**（该文件已随本次整理删除，此处保留记录供追溯）：依赖不存在的 `models.kernels.kernel_fns`（实测 ImportError）——它是一次未落地的“去 Module 化”重构。
2. **`dev.py` 参数不兼容**（已删）：`ModelClass1(..., training=False)` 中 `s4d_classification.Model.__init__` 没有 `training` 形参（实测 TypeError）。
3. **`train_full.py` 缺依赖**：`hydra`、`pytorch_lightning`、`wandb` 均未安装，也不在 `pyproject.toml` 中。
4. **`test.py` 的 profiler 片段不严谨**：`with torch.profiler.profile(...)` 复用了前面循环遗留的 `inputs`，且未包在 `no_grad` 里，只影响打印的算子表，不影响指标。
5. **`s4d_layer.py:88-102` 有死代码**：`forward` 在 `return self.kernel(...)` 之后还有一整段旧实现，永远不会执行。
6. **未使用 import**：`train.py` / `train_log.py` / `train_log_patch.py` 的 `S4Block as S4`、`S4D`、`torchvision`、`ImageFolder`；`backbones/*` 的 `S4`、`S4D`。副作用是 import `models.layers.s4` 时会触发 pykeops/ CUDA 扩展探测并打印警告。
7. **`--img_size` 无效**：`train_log_patch.py` 只把它写进 config，不参与任何 resize；序列长度完全由图片实际尺寸与 tokenizer 决定。
8. **训练侧无法选循环核**：`train_log_patch.py` 构造 Model 时不传 `kernel_type`，训练固定 conv；循环核只能用于 `test.py --kernel recurrent` 和流式模型。
9. **`engine/trainer.py` 与主脚本重复**，且 `save_checkpoint` 不写 `scheduler`，与 `train_log_patch.py` 的 checkpoint 结构不一致。
10. **`data_dir` 两套路径混用**：`finetune.yaml` / `pretrain.yaml` 用 `/mnt/e/datasets/NetMamba`，2026-10 的实验用 `/home/fsadness/datasets/NetMamba`；两处数据当前都存在，README 已改为提示显式传入。
11. **预训练是“因果”重建**：S4D 的卷积核是单边（因果）核，掩码重建只能利用前文，不能像 BERT 那样双向；作为预训练目标可行，但语义上更像自回归式去噪。
12. **`run_android_experiments.sh` 路径笔误**（已删）：`--save_dir ./experiment/finetune` 与仓库实际的 `experiments` 不一致，且脚本内 `cd` 到固定绝对路径。
13. **旧 README 与配置矛盾**：旧 `README.md` 写 Python 3.11，而 `.python-version` 与 `pyproject.toml` 都是 3.10 —— 新 README 已更正。

---

## 五、剩余整理步骤

> 初版的“移出大文件”“删除第一批临时文件”两步已完成；以下仅针对 3.2 的剩余项，逐步执行、每步可回退。

### 第 1 步：确认当前基线

```bash
cd /mnt/c/Users/Fsadness/Desktop/models/research/s4d
git status          # 当前 15 项：10 个已修改 + 5 个未跟踪
```

未跟踪的**正式代码**建议先纳入版本控制，避免后续误删：`engine/`、`models/backbones/s4d_recurrent_backbone.py`、`models/networks/s4d_recurrent_classification.py`（若决定保留 `test_inference.py`，也一并 `git add`）。

### 第 2 步：清掉 0 字节残留与缓存

```bash
rmdir checkpoint                       # 空目录，内容已在上一轮删除
find . -path ./.venv -prune -o -name __pycache__ -type d -print -exec rm -rf {} +
```

### 第 3 步：epoch 备份瘦身（收益最大，≈388 MiB）

```bash
# 先看每个 run 的备份体量，确认哪些还需要续训
find experiments -type d -name checkpoints -exec du -sh {} \;
ls experiments/*/*/*/latest.pth

# 确认后删除全部 epoch 备份（best.pth / latest.pth / train.log 不受影响）
find experiments -type f -path '*/checkpoints/*.pth' -delete
```

### 第 4 步：孤儿权重与历史评估副本

```bash
rm -rf experiments/pretrain/s4d_reconstruction/2025old
# 仅该 run 的重复评估副本；evaluation/ 会保留
rm -rf experiments/patch/s4d_classification/20260602_160857/evaluation_{old,conv_old,s4d,recurrent}
```

### 第 5 步：决定 `test_inference.py` 去留

```bash
# 继续做流式/早退研究 → 纳入版本控制
git add test_inference.py
# 否则归档
mkdir -p ~/s4d_archive && mv test_inference.py ~/s4d_archive/
```

### 第 6 步：校验

```bash
# 1) 仓库应只剩代码与精简后的产物
git status
du -sh .

# 2) 导入/前向冒烟
python -c "from models.networks.s4d_classification import Model; \
import torch; m=Model(1,6,64,2,0.0); print(m(torch.randn(2,100,1)).shape)"

# 3) 评估链路（任一保留的 run）
python test.py experiments/new/s4d_classification/20261028_110942 --device cpu
```

预期：仓库从当前 ≈ 500 MiB 降到 **≈ 110 MiB**（23 个 run 的 `best.pth` + `latest.pth` + `config.yaml` + `train.log`）。

---

## 六、风险提示

- **不要删 `.venv/`**（3.9 GiB）：重建要重装 cu116 版 torch，且 `pykeops` 编译耗时。
- **不要删 `experiments/*/best.pth` + `config.yaml` + `train.log`**：它们是唯一能复现报告指标的依据（`experiments/` 本身未进 git）。
- **epoch 备份删除前先确认续训需求**：`latest.pth` 在 run 根目录，不在 `checkpoints/` 里，删 `checkpoints/*.pth` 不影响续训与复现。
- **已删除的文件 git 找不回**：上一轮删掉的 `dev.py`、`dev2.py`、`s4d_layer_new.py`、`run_android_experiments.sh` 等本来就是未跟踪文件，不要尝试 `git checkout` 恢复（会失败）；如需找回只能从备份/回收站。
- **不要让 WSL 满载**：在 4 GB 显存上同时跑两个 `cuda` 任务时，WSL 曾整体卡死（`wsl -e` 命令 2 分钟无响应），单任务则正常。清理/重训请串行执行。
- **删除前先留退路**：剩下的大头是 `experiments/` 的产物，建议先 `tar` 一份到仓库外，再执行第 3 步。

---

## 七、附：文档变更记录

| 文件 | 动作 | 说明 |
| --- | --- | --- |
| `README.md` | 第一轮**重写** | 只保留持久内容：环境、数据组织、训练/评估/预训练用法、模型与 tokenizer、产物结构、已知限制、引用 |
| `ANALYSIS_REPORT.md` | 第一轮新增，第二轮**更新** | 本轮按实际清理结果复核：新增 3.1“已完成清理”，改写体积账本与 git 状态，第五节改为“剩余整理步骤”，并标注已删文件的相关观察 |

两轮整理过程中**未改动任何代码与配置**；第一轮的删除动作全部由用户执行，本轮仅复核并更新文档。
