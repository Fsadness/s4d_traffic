# s4d 代码整合方案（去冗余 · 共用数据链路 · 支撑四轴实验）

- 目标：把**同类功能收拢到单一实现**，数据/配置/指标只留一份，让后续实验变成"加配置/加策略"而不是"复制脚本"
- 约束：checkpoint 键名与数据目录约定**不能变**（已有 23 个 run 的 `best.pth` 要能直接加载）
- 结论：主线代码其实只有 ~2.5k 行，仓库里却有 **~36k 行不参与运行**；真正需要合并的是 **3 个训练脚本、2 个评估脚本、3 个网络类、2 个 backbone，以及量化仓里重复的 data/metrics**

**进度**：Phase 0 ✅（`a684de5`）｜Phase 1 ✅（`124419c`、`ecd6419`）｜Phase 2–4 待做，详见[第九节](#九phase-01-执行记录与下一步)

---

## 一、现状盘点：重复在哪

### 1.1 入口重复（最直接的成本）

| 功能 | 文件 | 行数 | 重复度 | 实际差异 |
| --- | --- | ---: | --- | --- |
| 训练 | `train.py` | 455 | 与下者 ~80% 相同 | 无日志、无早停、`d_input` 硬编码 |
| 训练 | `train_log.py` | 550 | 与下者 **96% 相同**（+20/−1 行） | 有日志/早停，但无 tokenizer |
| 训练 | `train_log_patch.py` | 569 | 主线 | 多了 4 个 tokenizer 参数与 `d_input` 推导 |
| 训练（未接入） | `engine/trainer.py` | 276 | 训练循环的另写一份 | 无入口调用、checkpoint 少 `scheduler`、预训练 loss 算法不同 |
| 评估 | `test.py` | 474 | 与下者 **~90% 相同**（17 个 hunk） | conv 模型 + `evaluation/` |
| 评估 | `test_inference.py` | 462 | 同上 | recurrent 流式模型 + `inference/` + profiler |

三个训练脚本合计 1574 行，约 1500 行是同一份训练循环；两个评估脚本合计 936 行，约 850 行重复。

### 1.2 模型重复

| 对比 | 新增行 | 说明 |
| --- | ---: | --- |
| `s4d_classification.py` → `s4d_recurrent_classification.py` | +148 | 只多了流式状态机 + early-exit 桩，其余（encoder/decoder/init/loss 接口）全同 |
| `s4d_backbone.py` → `s4d_recurrent_backbone.py` | +48 | 只是把 `forward` 改成 `forward_step` 循环 |
| `s4d_conv_kernel.py` → `s4d_recurrent_kernel.py` | +75 | **这部分该保留**：两种算法本就不同，共用 `S4DBaseKernel` 已足够 |

### 1.3 数据 / 指标重复

| 能力 | 重复实现 | 问题 |
| --- | --- | --- |
| tokenizer → `d_input` 推导 | `train_log_patch.py`、`test.py`、`s4d_quantization/quant/model.py` | 三处各写一遍 `if tokenizer == ...`，新增输入方案必漏改 |
| 数据集构建 | `datasets/dataloader.py::build_dataloader`、`test.py::load_test_only`、`quant/data.py::make_dataset` | 三份 `ImageFolder` + 目录拼接逻辑 |
| 指标/混淆矩阵 | `test.py`、`test_inference.py`、`quant/evaluate.py` | 三份 accuracy/P/R/F1 + 延迟统计 + 画图 |
| 配置读取/覆盖 | `train*.py` × 3、`quant/data.py::load_experiment_config` | 同一套"config 覆盖默认值"逻辑复制 4 次 |
| 模型结构 | `models/networks/*`、`quant/model.py`、`quant/recurrent.py` | 量化版把整网重写一遍（当前靠键名一致维持兼容） |

### 1.4 死代码（不运行、但仍参与 import 与维护）

| 路径 | 行数 | 是否被主线使用 |
| --- | ---: | --- |
| `src/`（128 个文件，上游快照） | **31,108** | 只用 `src.models.nn.DropoutNd`（25 行） |
| `models/layers/s4.py`（完整 S4） | 1,960 | 否，仅被 `import` 后闲置（还带来 pykeops/CUDA 扩展探测与警告） |
| `extensions/`（CUDA 核） | 1,214 | 否，仅被 `s4.py` 引用 |
| `example*.py` | 1,091 | 否，上游 CIFAR/MNIST demo |
| `train_full.py` | 832 | 否，缺 hydra/lightning，跑不起来 |
| `models/layers/s4d.py` | 107 | 仅 `example*.py` 用 |

**合计约 36.3k 行**，是主线代码（~2.5k 行）的 14 倍。

---

## 二、目标结构

原则：**按生命周期分层**（数据 → 模型 → 训练/评估 → 策略 → 入口），依赖只能向下，禁止 `datasets` ← `models` 这类反向 import。

```text
s4d/
├── datasets/                     # 只负责"目录 → 张量"
│   ├── specs.py                  # ★新增：InputSpec + TOKENIZERS 注册表 + infer_d_input()
│   ├── tokenizers.py             # 由 tokenizer.py 改名，5 个 tokenizer（纯函数，无 argparse）
│   ├── transforms.py             # build_transform(spec)
│   ├── dataloader.py             # build_dataset/build_dataloader(spec, splits, max_samples, seed)
│   └── manifest.py               # ★新增：样本清单缓存（避免每次 1 分钟 verify）
├── models/                       # 只负责"张量 → logits"
│   ├── kernels/                  # base / init / discretize / conv / recurrent / registry
│   ├── layers/s4d_layer.py       # 单一 S4DLayer（forward + forward_step）
│   ├── backbones/s4d_backbone.py # ★合并：一个 S4DBackbone，同时提供 forward / forward_step
│   ├── networks/
│   │   ├── classification.py     # ★合并：分类（含 streaming / early-exit 开关）
│   │   └── reconstruction.py     # 预训练重建（保留）
│   └── layers/dropout.py         # ★新增：DropoutNd 内联（替代 src.models.nn）
├── policies/                     # ★新增：推理期策略，与模型解耦
│   ├── base.py                   # ExitPolicy 协议
│   └── confidence.py             # 阈值/margin/patience/min_steps/stride + 阈值标定
├── engine/                       # 只负责"跑起来"
│   ├── trainer.py                # ★合并：唯一训练循环
│   ├── evaluator.py              # ★合并：唯一评估循环（conv / stream 两种模式）
│   ├── checkpoint.py             # ★新增：save/load/resume + prepare_for_inference 顺序
│   ├── metrics.py                # ★新增：指标/混淆矩阵/延迟，三处共用
│   └── logging.py                # train.log + meta.json
├── quant/                        # ★合并自 s4d_quantization（10 个模块）
├── configs/                      # ★新增：数据/输入/模型/训练/推理/量化 + 实验矩阵
│   ├── base.yaml
│   ├── input/{pixel,patch_p4,cpatch_c3,packet,logcp_seg64,mixed_rate}.yaml
│   ├── model/{conv,recurrent}.yaml
│   ├── infer/{full,conf_tau090,conf_margin}.yaml
│   ├── quant/{fp32,w8a8,paper_w4a6}.yaml
│   └── experiments/{c1_baseline,c2_transfer,c3_earlyexit,c4_ptq,c5_combined}.yaml
├── scripts/                      # 薄入口（≤50 行，只做参数解析与调用）
│   ├── train.py / eval.py / tune_exit.py
│   ├── quant/{ptq_eval,qat_train,diag_components}.py
│   └── sweep.py / collect.py     # 实验矩阵展开与汇总
├── tests/
│   ├── test_kernel_parity.py     # conv ↔ recurrent 数值等价 + 键名一致
│   ├── test_checkpoint_compat.py # 旧 best.pth 严格加载
│   ├── test_inputs.py            # 各 tokenizer 形状契约
│   └── test_quant_smoke.py
├── docs/                         # 本方案、量化说明、实验记录规范
├── experiments/                  # 产物：<cluster>/<run_id>/
└── archive/upstream/             # 上游快照（src/extensions/example/s4.py…）按需保留
```

根目录的 `train.py` / `test.py` 保留为 **10 行 shim**（`from scripts.train import main; main()`），
这样你现有的命令、队列脚本、肌肉记忆都不用改。

---

## 三、合并细则（从 → 到）

### 3.1 训练：3 → 1

`train_log_patch.py` 的逻辑整体搬进 `engine/trainer.py::Trainer`，差异全部变成参数：

| 旧差异 | 新形态 |
| --- | --- |
| `train.py` 无日志 | `engine/logging.py` 始终写 `train.log`（无副作用） |
| `train_log.py` 无 tokenizer | tokenizer 统一由 `InputSpec` 决定 |
| `train_log_patch.py` 的 4 个 CLI 参数 | `--input pixel|patch|cpatch|packet|logcp` + `--set input.patch_size=4` |
| `engine/trainer.py` 未被调用 | 成为唯一实现 |

`Trainer` 只暴露 4 个方法：`fit()` / `train_epoch()` / `evaluate()` / `save()`；
早停、AMP、梯度累积都放这里，避免再复制第 4 份训练循环。

### 3.2 评估：2 → 1

`engine/evaluator.py` 支持 `--mode {conv, stream}`：

- `conv`：全序列前向（现在的 `test.py`）
- `stream`：`forward_step` 逐步 + 早退策略（现在的 `test_inference.py`）

两者共用 `engine/metrics.py`（指标、混淆矩阵、延迟/吞吐、置信度/margin），
流式模式额外输出 `steps_hist`、`avg_stop_step`、`early_exit_ratio`、`acc@avg_steps`。
`test_inference.py` 删除，`test.py` 变 shim。

### 3.3 模型：3 networks + 2 backbones → 2 + 1

```python
# models/backbones/s4d_backbone.py
class S4DBackbone(nn.Module):
    def forward(self, x): ...          # (B,L,D) -> (B,L,H)
    def forward_step(self, x_t): ...   # (B,D) -> (B,H)，内部逐层 step
    def reset_state(self, batch_size, device): ...
    def prepare_for_inference(self): ...
```

分类网络合并流式状态机（当前 148 行）：

```python
# models/networks/classification.py
class Model(nn.Module):
    def __init__(..., kernel_type='conv', streaming=False, exit_policy=None): ...
    def forward(self, x): ...                    # 训练/整段推理
    def forward_step(self, x_t): ...             # 只在 streaming=True 时可用
```

**关键约束**：`state_dict` 键名保持不变（`backbone.*`、`decoder.*`、`s4_layers.i.kernel.*`），
用 `tests/test_checkpoint_compat.py` 锁死。

### 3.4 数据：三份 → 一份

```python
# datasets/specs.py
@dataclass(frozen=True)
class InputSpec:
    name: str
    params: dict
    def build(self) -> Callable[[Tensor], Tensor]: ...
    def d_input(self) -> int: ...
    def seq_len(self, h: int, w: int) -> int: ...
    def describe(self) -> str: ...       # 用于 run_id 与 meta.json

TOKENIZERS = {"pixel": ..., "patch": ..., "cpatch": ..., "packet": ..., "logcp": ...}
def make_input_spec(cfg) -> InputSpec: ...
def infer_d_input(cfg) -> int: return make_input_spec(cfg).d_input()
```

```python
# datasets/dataloader.py
def build_dataset(spec, split, data_dir, max_samples=None, seed=0): ...
def build_dataloader(spec, splits=("train","valid","test"), ...) -> dict[str, DataLoader]
```

于是：`test.py` 不再写 `load_test_only`，量化仓的 `quant/data.py` 删掉自己那份实现，
`train/test/quant/sweep` 全部走同一个入口。
`manifest.py` 把 `SafeImageFolder` 的 `PIL.verify()` 结果缓存成 `split.manifest.json`
（文件数 + size + mtime 做指纹），**批量实验时每次启动能省约 1 分钟**。

### 3.5 配置：4 份 → 1 份

```python
# configs/schema.py
@dataclass
class RunSpec:
    data: DataCfg
    input: InputSpec
    model: ModelCfg
    train: TrainCfg
    infer: InferCfg
    quant: QuantCfg | None = None
    def run_id(self) -> str: ...     # cici-patch4-conv-full-fp32-s0
```

- 单一 argparse（`--config` + `--set k=v`），删除 3 份重复的 `parser.add_argument`
- 保留现有 `config.yaml` 的字段名，旧实验目录仍能直接 `--resume`
- 每次运行写 `meta.json`：`run_id / git_commit / dirty / argv / resolved_config / dataset_sizes / seed`

### 3.6 量化仓：合并 + 只共用三件事

按 `s4d_quantization/MERGE_GUIDE.md` 合并到 `quant/`、`scripts/quant/`、`tests/`、`docs/quantization.md`，并额外做三处去重：

1. **数据**：`quant/data.py` 改为调用 `datasets.dataloader.build_dataset`（保留 `max_samples` 子集参数）；
2. **指标**：`quant/evaluate.py` 改为调用 `engine/metrics.py`（量化版只多一个显存估算）；
3. **配置**：`load_experiment_config` 与主仓共用 `configs/schema.py` 的读取函数。

**不合并**的部分：`quant/model.py` 与 `quant/recurrent.py` 保持独立前向——
量化语义（Q(·) 的位置、状态裁剪、qGELU）与 FP 不同，硬合并会把 FP 路径也拖进量化分支。
用 `tests/test_key_parity.py` 保证两者 `state_dict` 键名与 FP 完全一致即可。

### 3.7 死代码处置（按最终约定调整）

| 处置 | 内容 | 状态 |
| --- | --- | --- |
| 归档 | `example.py`、`example_pretrain.py`、`example_test.py`、`train_full.py`、`README_old.md`、`main.py` → `archive/` | ✅ 已完成（`a684de5`） |
| 保留原位 | `src/`（31k 行）、`extensions/`、`models/layers/s4.py`、`models/layers/s4d.py` | 按"其他文件夹保留原组织"的约定，**不迁移**；`DropoutNd` 暂不内联 |
| 暂缓 | `engine/trainer.py`（未接入）、`test_inference.py`、`train_log.py` | 保留到 Phase 2/3 统一收敛时再处理 |

> 因此启动时仍会出现 `CUDA extension for structured kernels ... not found` 与 KeOps 提示：
> 它们来自仍未迁移的 `models/layers/s4.py`，不影响 S4D 主路径。

---

## 四、为四轴实验预留的接缝

你的计划是 4 个正交维度，结构上应各占一个"缝"，加实验时不改主干。对照关系：

| 你的计划 | 实验簇 | 依赖的接缝 | 需要新增的东西 |
| --- | --- | --- | --- |
| ① 输入结构 / 压缩方案 | `c1_inputs` | `InputSpec` + registry | 1 个 tokenizer + N 个 yaml |
| ② 递推推理能力（conv 权重迁移） | `c2_transfer` | kernel registry + `load_fp_checkpoint` | 0（已有，仅补 parity 测试） |
| ②⁺ 提前结束推理 + 阈值标定 | `c3_earlyexit` | `policies/` + `evaluator --mode stream` | 1 个 policy + 1 个 tune 脚本 |
| ③ 模型量化（PTQ/QAT） | `c4_quant` | `quant/` 合并 | 0（迁移已有代码） |
| ③⁺ 变速率压缩（按节点/段） | `c4b_varrate` | `schedule` + `node_overrides` | 1 个 schedule tokenizer + 扩展 QuantConfig |
| ④ 综合实验 | `c5_combined` | `configs/experiments` + `sweep` | 组合 yaml |

各轴的具体落点与"加新变体要改哪里"：

| 轴 | 落点 | 加新变体要改哪里 |
| --- | --- | --- |
| ① 输入结构 / 压缩 | `datasets/specs.py` + `configs/input/*.yaml` | 只加 1 个 tokenizer 类 + 1 个 yaml |
| ② 卷积 ↔ 递推 | `models/kernels/registry.py` + `--kernel-type` | 不改（两种核已同键名） |
| ②⁺ 提前结束 | `policies/` + `engine/evaluator.py --mode stream` | 只加 1 个 policy 类 + 1 个 yaml |
| ③ 量化 | `quant/` + `configs/quant/*.yaml` | 只加 1 个 QuantConfig 预设 |
| ③⁺ 变速率压缩 | `InputSpec.params.schedule` + `quant` 的 per-node 位宽 | 二者共用 `node_group` 命名 |
| ④ 综合实验 | `configs/experiments/*.yaml` + `scripts/sweep.py` | 只加组合 yaml |

### 4.1 输入轴：把"压缩"变成数据而不是代码

现在 `LogCompressedPacketTokenizer` 把段长规则写死在 `__call__` 里。改成"调度表"：

```python
# datasets/tokenizers.py
class SegmentScheduleTokenizer:
    def __init__(self, packet_count=5, packet_size=320,
                 schedule=((64, 1.0), (64, 0.5), (64, 0.35), (64, 0.25), (64, 0.2))):
        ...
    def __call__(self, x): ...   # 按 schedule 逐段 pool，返回 (L', 5)

# datasets/specs.py
InputSpec(name="mixed_rate", params={"schedule": [(64, 1.0), (64, 0.5), ...]})
```

`schedule` 同时被两处消费：
- tokenizer 决定每段保留多少 token；
- 量化配置决定每段（node_group）用几位宽 → 正好接 ③⁺。

### 4.2 核轴：权重迁移已经可用，但要修一个坑（实测）

我刚验证过：

```text
conv keys: 22  rec keys: 22   (完全一致)
conv -> recurrent load_state_dict(strict=True): OK
直接换核输出差异 max|diff| = 0.18        ← 错误：缓存未刷新
调用 prepare_for_inference() 后 = 1.19e-7 ← 正确：数值等价
```

原因：`Model.eval()` 会自动 `prepare_for_inference()`，但**在 `load_state_dict` 之前**调用过一次时，
缓存里存的还是初始化权重；加载后必须再刷新。`test.py` 恰好顺序正确（load → eval），
自己写迁移脚本时极易踩。整合时把它封进唯一入口：

```python
# engine/checkpoint.py
def load_fp_checkpoint(model, path, device="cpu", strict=True):
    model.load_state_dict(torch.load(path, map_location=device)["model"], strict=strict)
    model.eval()
    if hasattr(model, "prepare_for_inference"):
        model.prepare_for_inference()      # 必须在 load 之后
    return model
```

并加 `tests/test_kernel_parity.py`：同一权重、同一输入，conv 与 recurrent 输出 `allclose(atol=1e-5)`。

### 4.3 早退轴：现在只有桩，需要真策略

现状：`s4d_recurrent_classification.early_stop_strategy` 默认 `None`，
代码里只有 `early_stop_strategy.should_stop(logits, step, total_steps)` 的调用与 `# TODO`；
`test_inference.py` 却把 `--early_stop` 的**字符串**直接传进去 → 一旦使用就会 `AttributeError`。

建议的协议（与量化无关，只看 logits，因此量化后可直接复用）：

```python
# policies/base.py
class ExitPolicy:
    def reset(self, batch_size: int, total_steps: int, device): ...
    def should_stop(self, logits, step: int) -> tuple[Tensor, dict]: ...
        # 返回 (B,) bool + info；已停止的样本必须保持 True（单调性）

# policies/confidence.py
class ConfidenceExit(ExitPolicy):
    # 参数：tau（置信度阈值）、margin（top1-top2）、min_steps、patience、stride
    # 规则：conf >= tau 且 margin >= margin_min 连续 patience 次 → 停
    # 附加：per_class_tau（按类别标定，缓解长尾类过早退出）
```

配套：
- `scripts/tune_exit.py`：在 valid 上扫 `tau × margin × min_steps`，画 `acc vs avg_steps` 曲线，输出 `infer.yaml`；
- `engine/evaluator.py --mode stream` 记录每个样本的 `stop_step`，算 `speedup = 全序列延迟 / 平均退出延迟`；
- **注意** `stride`：现在是"每步都过 decoder"，每步都有 `H×C` 的计算；
  实际部署可每 k 步评估一次置信度（k=4/8/16），这是早退能否真正加速的关键参数。

### 4.4 量化轴：合并后只加配置

```yaml
# configs/quant/paper_w4a6.yaml
bits_w: 4
bits_a: 6
bits_a_bar: 8
bits_state: 8
use_qgelu: true
param_mode: continuous
per_head: true
node_overrides:            # ③⁺：按输入节点/段改位宽
  segment_0: {bits_state: 8}
  segment_3: {bits_state: 6}
```

### 4.5 综合实验：只写组合

```yaml
# configs/experiments/c5_combined.yaml
defaults: {data: cici, model: conv}
grid:
  input: [pixel, patch4, mixed_rate]
  quant: [fp32, w8a8, paper_w4a6]
  infer: [full, conf_tau090, conf_margin]
seeds: [0, 1, 2]
```

`scripts/sweep.py` 展开成 27×3 个 run，写 `experiments/<cluster>/<run_id>/`，
`scripts/collect.py` 汇总成一张表（run_id × 轴 × acc × 延迟 × 平均步数）用于画图。

---

## 五、迁移路线（每阶段可独立运行、可回滚）

### Phase 0 ✅：归档根目录未使用的上游代码（已完成，提交 `a684de5`）

按"只动根目录、其他文件夹保留原组织"的约定执行：

```bash
mkdir archive
git mv example.py example_pretrain.py example_test.py train_full.py README_old.md main.py archive/
# 另加 archive/README.md 说明来源与取回方式
```

没有按最初方案迁移 `src/`、`extensions/`、`models/layers/s4.py`、`models/layers/s4d.py`——
因此也没有内联 `DropoutNd`、没有摘除未使用的 `S4Block/S4D` import（它们仍可正常 import），
启动时那条 pykeops/CUDA 警告依旧存在，属于已知且可接受的状态。

验收：14 个主线模块 import 通过；仓库根目录只剩 6 个脚本（train/test/train_log/train_log_patch/test_inference）。

### Phase 1 ✅：数据与指标共用（已完成，提交 `124419c`、`ecd6419`）

| 改动 | 内容 |
| --- | --- |
| `datasets/specs.py` | `InputSpec` + `TOKENIZERS` 注册表 + `make_input_spec` / `infer_d_input` |
| `datasets/tokenizer.py` | 每个 tokenizer 增加 `d_input` 与 `seq_len()`，形状信息不再散落在调用方 |
| `datasets/transforms.py` | 改为 `make_input_spec(args).build()`，不再 if/elif |
| `datasets/manifest.py` | 把 `PIL.verify()` 结果缓存到 `.cache/manifests/`（指纹 = 路径+大小+mtime） |
| `datasets/dataloader.py` | 唯一入口 `build_dataset` / `build_loaders(splits, max_samples, seed)`；`build_dataloader` 保留为旧接口 |
| `engine/metrics.py` | `compute_metrics` / `normalized_confusion` / `write_classification_report` / `plot_confusion_matrix` |
| `test.py` | 删掉自带的 `load_test_only`、sklearn/matplotlib 代码块，改用共用模块；新增 `--max_samples` |
| `train_log_patch.py` | 改用 `build_loaders` + `infer_d_input`；新增 `--max_samples` |

验收结果（均为极小样本/1 epoch）：

| 检查 | 结果 |
| --- | --- |
| 5 种 tokenizer 的 `d_input` / `seq_len` | 与旧逻辑逐项一致（1/16/9/5/5，1600/100/100/320/146） |
| 数据集缓存开关 | 样本列表逐条一致；1951 样本：无缓存 2.55s → 命中缓存 0.04s |
| `build_dataloader` 旧接口 | 返回 `(15597, 1949, 1951, 6)`，与旧版一致 |
| 训练（patch, 96 样本, 1 epoch） | 通过，`encoder.weight = (128, 16)` |
| 训练（pixel, 64 样本, 1 epoch） | 通过，`Data shape = (1600, 1)` |
| 评估（全量 1951，CPU） | **98.360% / macro-F1 98.047% / loss 0.0600 / conf 0.9926 / margin 0.9856**，与改造前逐位一致 |
| 评估（`--max_samples 64`） | 通过（顺带修了"子集缺类"时混淆矩阵/CSV 维度错位的问题） |
| 评估（`--kernel recurrent`，16 样本） | 通过，产物写入 `evaluation_recurrent/` |

未做（按约定）：不合并 `quant/`；不迁移 `src/`、`extensions/`、`models/layers/*`；不动 `test_inference.py`、`train_log.py`。

### Phase 2：训练收敛（1–2 天，待做）

- `engine/trainer.py` 吸收 `train_log_patch.py`；`engine/checkpoint.py` 统一存取
- `scripts/train.py` 新入口；根 `train.py` 变 shim
- 验收：1 epoch 冒烟 + `--resume` 续训 + 旧 `config.yaml` 仍可读

### Phase 3：模型收敛（2–3 天）

- backbone 合并 `forward_step`；classification 合并流式/早退；删 `test_inference.py`
- 验收：`test_checkpoint_compat`（旧 best.pth 严格加载）+ `test_kernel_parity`（1e-7）

### Phase 4：量化合并 + 四轴接线（2–3 天）

- 合并 `quant/`，改 data/metrics/config 三处调用
- 加 `policies/`、`scripts/tune_exit.py`、`configs/experiments/*`、`sweep/collect`
- 验收：PTQ/QAT 冒烟通过；一个 3×3 小网格能自动跑完并汇总

---

## 六、预期收益

| 指标 | Phase 1 前 | 现在（Phase 1 后） | 目标（Phase 4 后） |
| --- | ---: | ---: | ---: |
| 训练实现 | 3 脚本 + 1 未接入 Trainer | 3 脚本（数据层已共用） | 1 Trainer + 2 shim |
| 评估实现 | 2 脚本 | 2 脚本（`test.py` 已瘦身） | 1 Evaluator + 1 shim |
| `d_input` 推导 | 3 份 | **1 份**（`InputSpec`） | 1 份 |
| 数据集构建 | 3 份 | **2 份**（主仓 1 份；quant 待合并） | 1 份 |
| 指标/画图 | 3 份 | **2 份**（主仓 1 份；quant 待合并） | 1 份 |
| 参与运行的行数 | ~2.5k | ~2.6k（新增 specs/manifest/metrics） | ~2.4k |
| 不参与运行的行数 | ~36k | ~34.5k（根目录 1.5k 已归档） | 0 |
| 新增输入方案 | 改 3 处 + 1 个 tokenizer | **改 1 处 + 1 个 yaml** | 同左 |
| 新增早退策略 | 改模型类 | 改模型类 | 加 1 个 policy + 1 个 yaml |

---

## 七、边界：哪些不该合并

- **两种核保持两个类**：conv（FFT）与 recurrent（逐步递推）算法不同，只共用 `S4DBaseKernel` 与键名；
- **FP 与量化保持两套前向**：量化语义（`Q(Disc(·))`、状态裁剪、qGELU）会污染 FP 路径；共用数据/指标/配置即可；
- **训练与评估不合并**：评估要支持"加载任意 checkpoint + 多种推理模式"，塞进 Trainer 会再次耦合；
- **`datasets/` 的目录约定不动**：`dataset_sampled/{train,valid,test}` 是你已有 6 个数据集的既有结构，改了收益小、迁移成本高。

---

## 八、如果只做一件事

优先做 **Phase 0 + Phase 1**（✅ 已完成）：
归档根目录的上游代码、把 `InputSpec`/`build_dataloader`/`metrics` 抽成单份。
这两步不碰模型与训练逻辑，风险最低，但已经能让"加一种输入方案"从改 3 处变成改 1 处，
并把你的 4 轴实验从"复制脚本"变成"写配置"。

---

## 九、Phase 0–1 执行记录与下一步

### 9.1 提交记录（仅本地 commit，未 push）

| 提交 | 内容 | 变更 |
| --- | --- | --- |
| `a684de5` | 归档根目录未使用的上游代码 | 6 个文件 → `archive/`，+`archive/README.md` |
| `124419c` | 数据层：`InputSpec` / 清单缓存 / 统一加载器 | 7 文件，+392 / −92 |
| `ecd6419` | 评估层：抽出 `engine/metrics.py` | 2 文件，+234 / −213 |
| `5c04eb6` | 文档：README 与新目录/新参数；本方案记录 Phase 0–1 | 2 文件，+545 / −7 |
| `2782969` | 修正 `--max_samples` 的帮助文案（随机子集，非前 N 个） | 2 文件，+3 / −3 |

基线：`4bce5b2`（Phase 0 之前），需要整体回滚时用 `git reset --hard 4bce5b2`。

### 9.2 新增的公开接口（后续实验直接用）

```python
from datasets.specs import make_input_spec, infer_d_input
from datasets.dataloader import build_loaders, build_dataset, SplitData
from engine.metrics import compute_metrics, normalized_confusion, write_classification_report

loaders = build_loaders(cfg, splits=("train", "valid"), max_samples=256, seed=0)
spec = make_input_spec(cfg)          # spec.d_input / spec.seq_len(40, 40) / spec.describe()
```

- `train_log_patch.py` / `test.py` 都新增了 `--max_samples`，可直接跑"极小样本冒烟"。
- 数据集校验结果缓存在 `.cache/manifests/`（已加入 `.gitignore`），更换/修改数据会自动失效。

### 9.3 下一步建议（按收益/风险排序）

1. **先把当前状态提交为实验基线**（已完成，HEAD=`ecd6419`）。
   现在可以开始你的簇 1（输入结构 / 压缩方案）实验，因为加方案只需写一个 tokenizer + 一个 yaml。

2. **补两个"护栏"测试（半天，强烈建议）**，它们直接保护你计划里的第 2、3 步：
   - `tests/test_kernel_parity.py`：同一权重下 conv 与 recurrent 输出 `allclose(atol=1e-5)`，
     并断言两者 `state_dict` 键名一致。这是"conv 训练、递推推理/量化"迁移路线的前提。
   - `tests/test_checkpoint_compat.py`：用任意一个历史 `best.pth` 严格加载新模型。
   现在这套结构下写测试成本很低，等 Phase 3 改模型时再补就来不及发现回归。

3. **抽 `engine/checkpoint.py`（半天，可与 2 合并）**：
   ```python
   def load_fp_checkpoint(model, path, device="cpu", strict=True):
       model.load_state_dict(torch.load(path, map_location=device)["model"], strict=strict)
       model.eval()
       if hasattr(model, "prepare_for_inference"):
           model.prepare_for_inference()     # 必须在 load 之后重建离散参数缓存
       return model
   ```
   已经实测过：不刷新缓存时 conv→recurrent 的 logits 差 0.18，刷新后 1.19e-7。
   你的第 2 步与量化迁移都会用到它，先封好能省掉一类难查的 bug。

4. **Phase 2（训练收敛）**：`train_log_patch.py` → `engine/trainer.py`，加 `--kernel-type`、
   `--seed`、每 run 写 `meta.json`（git commit / argv / resolved config）。
   `--seed` 是后面扫阈值与量化的前提——现在训练完全没有设种子。

5. **Phase 3 前半（评估统一）**：`test.py` 加 `--mode stream`，吸收 `test_inference.py`；
   同时把早退策略抽到 `policies/`（当前 `early_stop_strategy` 只是桩，传字符串会崩）。
   这是你第 2+ 步的入口。

6. **Phase 4（量化合并）**：按 `MERGE_GUIDE.md` 并入 `quant/`，只改 data/metrics/config 三处调用。
   合并前不需要动主仓任何前向代码。

> 建议顺序：**2 → 3 → 4（簇 1 实验并行跑）→ 5 → 6**。
> 其中 2、3 是纯新增文件，不改变现有行为，可以立刻做；4 之后你的训练才是可复现的。
