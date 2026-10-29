# 量化实验（Edge AI）

复刻论文 **[Quantizing Small-Scale State-Space Models for Edge AI](https://arxiv.org/abs/2506.12480)**
（Zhao et al., 2025），作用对象是主仓 `experiments/<...>/best.pth` 里的 S4D 流量分类模型。

> 本目录已合并进 s4d 仓库，不再是独立工程：
> 数据/配置复用 `datasets/` 与 `engine/config.py`，指标复用 `engine/metrics.py`，
> checkpoint 键名与 `models/networks/s4d_classification.py` 完全一致。

```text
quant/                       # 量化库
├── fake_quant.py            # 均匀伪量化 + STE、复数量化、qGELU、状态裁剪
├── config.py                # QuantConfig：位宽/粒度/校准/参数化 + 分段位宽(segment_bits)
├── modules.py               # QuantLinear / QuantConv1d / QuantLayerNorm
├── recurrent.py             # 递推 S4D 核：量化 Ā/B̄/C/D、状态 x（支持分段位宽）
├── model.py                 # QuantS4DClassification（键名与主仓对齐）
├── data.py                  # 复用 datasets.dataloader 的数据加载
├── calibrate.py             # 静态校准
├── evaluate.py              # 精度/延迟/吞吐（复用 engine.metrics）
├── qat.py                   # QAT（STE + 梯度裁剪 + 三种参数化）
├── memory.py                # 权重/状态显存估算（分段时按平均状态位宽）
├── cli.py                   # 脚本共用的量化参数与策略构造
└── repo.py                  # 定位仓库根目录

scripts/quant/               # 入口
├── ptq_eval.py              # PTQ 评估（单策略 / 位宽扫描）
├── qat_train.py             # QAT 微调
└── diag_components.py       # 组件消融

configs/quant/               # 策略文件（可被 PTQ / QAT 共用）
├── paper_w4a6.yaml
├── uniform8.yaml
└── segment_8_4.yaml         # 前 800 token 8-bit，其余 4-bit

tests/test_quant_smoke.py    # 单元 + 集成冒烟
```

---

## 1. 环境

沿用主仓的 uv 虚拟环境（torch 1.13.1+cu116）：

```bash
cd /mnt/c/Users/Fsadness/Desktop/models/research/s4d
source .venv/bin/activate
```

---

## 2. 快速开始

### 2.1 冒烟测试

```bash
# 单元测试（秒级）
python tests/test_quant_smoke.py --unit-only

# 集成：加载 best.pth → FP 递推 → 校准 → PTQ → QAT 一步
python tests/test_quant_smoke.py --max-samples 16
```

### 2.2 PTQ

```bash
# 论文异构方案（W4A6，Ā 与 x 8-bit）
python scripts/quant/ptq_eval.py \
    --ckpt-dir experiments/new/s4d_classification/20261028_110942 \
    --preset paper --max-samples 256

# 统一位宽扫描
python scripts/quant/ptq_eval.py --ckpt-dir <exp> --sweep 16,8,6,4

# 组件敏感度（只量化 Ā / 状态）
python scripts/quant/ptq_eval.py --ckpt-dir <exp> --bits-a 8 --bits-state 8

# 策略文件（推荐：策略可复用、可记录）
python scripts/quant/ptq_eval.py --ckpt-dir <exp> \
    --quant-config configs/quant/segment_8_4.yaml
```

输出：终端表格 + 可选 `--out result.json`（含完整 `quant_config`、精度、延迟、显存）。

### 2.3 QAT

```bash
python scripts/quant/qat_train.py --ckpt-dir <exp> \
    --quant-config configs/quant/segment_8_4.yaml \
    --tag seg84 --epochs 2 --param-mode discrete
```

产物：`<exp>/quant_qat_<tag>/quant_best.pth` 与 `qat_report.json`
（含 `quant_config`、PTQ/QAT 精度、显存估算、训练历史）。

小比例数据微调（按类别分层抽样，每类至少 1 个）：

```bash
python scripts/quant/qat_train.py --ckpt-dir <exp> \
    --quant-config configs/quant/paper_w4a6.yaml --param-mode discrete \
    --train-fraction 0.05 --epochs 10 --batch-size 32 --eval-batch-size 128 \
    --num-workers 0 --max-eval-samples 0 --eval-every-epoch \
    --tag frac5
```

`--eval-every-epoch` 会在每个 epoch 后用 valid 集评估并写入 `history`；
最终指标仍在全量 test 集上计算。参考结果见
[docs/qat_data_fraction.md](qat_data_fraction.md)。

### 2.4 组件消融

```bash
python scripts/quant/diag_components.py --ckpt-dir <exp> --max-samples 32
```

---

## 3. 量化策略（两类混合精度）

### 3.1 按参数分组（论文默认）

`QuantConfig` 的位宽字段：

| 字段 | 含义 |
| --- | --- |
| `bits_a` | 离散状态转移矩阵 Ā（最敏感，PTQ 下限 8-bit） |
| `bits_b` / `bits_c` | 输入矩阵 B̄ / 输出矩阵 C |
| `bits_d` | 直通项 D |
| `bits_state` | 递推状态 x |
| `bits_act` | SSM 之外的激活（qGELU / Conv1D / GLU / 层输出 / 池化） |
| `bits_other` | 非 SSM 权重（encoder / decoder / LayerNorm / Conv1D） |

粒度 `per_head`、对称性 `symmetric`、校准 `static|dynamic`、QAT 参数化
`continuous|discrete|frozen` 同样在 `QuantConfig` 中。

### 3.2 按序列位置分段（本轮新增）

`segment_bits` 允许不同 token 位置使用不同的**运行时**位宽，例如"前 800 个 token
用 int8、后 800 个用 int4"：

```yaml
# configs/quant/segment_8_4.yaml（节选）
bits_state: 8          # 未被分段覆盖时的默认值
bits_act: 6
segment_bits:
  - {start: 0,   end: 800,  bits_state: 8, bits_act: 8}
  - {start: 800, end: null, bits_state: 4, bits_act: 4}
```

语义：

- `start` 含、`end` 不含，`end: null` 表示到序列结尾；
- 只覆盖**运行时量**（状态 x 与激活）；按参数分组的位宽（Ā/B̄/C/D/权重）仍由
  `bits_a…bits_other` 决定，因为它们与位置无关；
- 多段重叠时**后出现的覆盖先出现的**（方便写"默认 + 例外"）；
- 未覆盖的位置回落到 `bits_state` / `bits_act`；
- 显存估算按"每段长度加权平均"计算 state/sample。

命令行简写（等价于上面的 yaml）：

```bash
--segment-bits "0:800:8,800::4"        # 状态与激活都改
--segment-bits "0:400:a8,400::a4"      # 只改激活位宽
--segment-bits "0:400:s8"              # 只改状态位宽
```

实现上：默认位宽沿用原来的 `q_state` / `q_act_*` 量化器；只有"与默认不同"的位宽
才会额外创建量化器（`q_state_extra` / `q_act_extra`），因此**不配置分段时
state_dict 键名与旧版完全一致**。校准时这些额外量化器只在各自负责的区间/时间步
观察范围，因此 PTQ 与 QAT 都无需特殊处理。

### 3.3 策略优先级

`--quant-config`（yaml）< 命令行显式参数。布尔项用成对开关避免"默认值覆盖 yaml"：

```bash
--symmetric / --asymmetric     --per-head / --per-tensor
--dynamic / --static           --qgelu / --gelu
```

---

## 4. 实验流程（每个策略都从 FP 独立出发）

两类混合精度、PTQ/QAT 都遵循同一流程：

```text
FP best.pth
   │  build_quant_model（键名对齐，可直接 load）
   ├─ PTQ：权重校准 → 静态校准（valid 小批）→ 评估 → result.json
   └─ QAT：PTQ 起点 → 递推模式 STE 微调 → 评估 → quant_best.pth + qat_report.json
```

- 每个策略都是**独立一次** PTQ / QAT，互不共享量化参数；
- 产物用 `--tag`（QAT）或 `--out`（PTQ）区分，便于横向对比；
- `qat_report.json` / `result.json` 都写入完整 `quant_config`，可复现。

---

## 5. 已验证（CPU 小样本冒烟）

| 检查 | 结果 |
| --- | --- |
| 单元测试（含分段位宽解析/区间展开/序列化） | 全部通过 |
| PTQ `--preset paper`（8 样本） | FP 递推基线 100%；PTQ 25%（qGELU 未做 QAT，符合此前结论） |
| PTQ `--quant-config segment_8_4.yaml` | 正常跑通；`describe` 显示 `seg [0:800](a8/s8),[800:None](a4/s4)`；state/sample 131072B → 24576B（平均 6 bit） |
| QAT（1 batch，segment 策略） | 16 s 完成，loss 有限；`quant_best.pth` + `qat_report.json` 正常落盘 |
| checkpoint 往返 | `load_quant_checkpoint` 恢复 `param_mode=discrete` 与 `segment_bits`，前向形状正确 |

> 完整实验（全量测试集、多 epoch QAT）建议在更大显存机器上跑；
> 本仓 4 GB 显卡适合做策略可行性验证。

---

## 6. 已知限制

- **只量化递推模式**：卷积（FFT）路径未量化，论文的部署形态也是递推；
- **分段位宽作用于状态与激活**，不影响权重（权重没有"位置"维度）；
- **QAT 代价高**：L=1600 的逐步反传很慢，冒烟时用 `--max-train-batches` / `--batch-size 2~4`；
- **显存估算是常驻量**（权重 + 单样本状态），不含激活峰值；
- **qGELU 会显著掉点**（PTQ 阶段），需要 QAT 才能恢复，与论文一致；做"纯量化"对照请加 `--gelu`。

---

## 7. 与论文的对应关系

| 论文内容 | 实现 |
| --- | --- |
| II 节：均匀量化、有符号网格 | `UniformQuantizer` |
| II 节：99.999% 百分位校准 | `percentile` |
| II 节：per-tensor / per-head | `per_head` + `channel_axis` |
| II 节：对称 / 非对称 | `symmetric` |
| II 节：静态 / 动态校准 | `calibration` |
| II 节：复数量化共用 scale | `forward_complex` |
| II 节：qGELU | `qgelu` / `QGELU` |
| IV 节：状态裁剪 ±50 | `clip_state` |
| V 节：STE + 梯度裁剪 1000 | `qat.py` |
| V 节：三种参数化 | `param_mode` |
| V 节：异构位宽 W4A6 + Ā8/x8 | `paper_config()` / `configs/quant/paper_w4a6.yaml` |
| Table III：显存节省 | `memory.py` |
| Fig.2：单组件消融 | `scripts/quant/diag_components.py` |
| （扩展）按序列位置的混合精度 | `QuantConfig.segment_bits` |
