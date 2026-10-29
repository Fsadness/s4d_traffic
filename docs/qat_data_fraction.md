# 实验：小比例分层数据做 QAT 能否恢复精度

- 日期：2026-10-29
- 问题：QAT 微调时，用 **10% / 5% 按类别分层抽样** 的原始训练集，能否在全量测试集上恢复到接近 FP 的精度？
- 结论：**能**。在 patch 模型（L=100）上，5% 分层数据 + 10 epochs 已恢复到 **98.10%**（FP 99.85%），
  10% 数据恢复到 **98.46%**；两者只差 0.36 个百分点。

---

## 1. 实验设置

| 项 | 值 |
| --- | --- |
| 实验目录 | `experiments/patch/s4d_classification/20260602_182157` |
| 数据集 | CICIoT2022（6 类） |
| 输入 | `patch`，`patch_size=4` → L=100，`d_input=16` |
| 模型 | `s4d_classification`，d_model=128，4 层，约 0.2M 参数 |
| FP 基线（全量测试集 1951） | **99.85%** |
| 量化方案 | 论文 W4A6：A8 B4 C4 D4 / x8 / act6 / W4，per-head、非对称、静态校准、qGELU |
| 参数化 | `discrete`（QAT 直接训练离散 Ā/B̄/C） |
| PTQ 起点 | **87.08%**（同一配置，全量测试集） |
| QAT | Adam，lr=1e-4，grad clip 1000，batch 32，**10 epochs** |
| 校准 | valid 集 4 × 128 样本 |
| 监控 | 每个 epoch 在 valid 上评估；最终指标只在 test 上算 |

分层抽样（`quant/data.py::make_stratified_loader`）：每类取 `round(n_c × fraction)` 个（至少 1 个），
固定 seed=0，保证两档比较的是同一套抽样规则。

| 比例 | 样本数 | 每类样本数（0–5） | 每 epoch batch | 训练耗时 |
| --- | ---: | --- | ---: | ---: |
| 5% | 780 | 89 / 105 / 240 / 94 / 10 / 242 | 25 | 296 s |
| 10% | 1560 | 179 / 210 / 480 / 187 / 20 / 484 | 49 | 499 s |

---

## 2. 结果（全量测试集）

| 比例 | PTQ | QAT 后 | 相对 FP 的差距 | 恢复的差距占比 | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP（参考） | — | **99.85%** | — | — | 99.81% |
| 5% | 87.08% | **98.10%** | −1.75 pt | 86.3% | 97.71% |
| 10% | 87.08% | **98.46%** | −1.39 pt | 89.1% | 98.16% |

> “恢复的差距占比” = (QAT − PTQ) / (FP − PTQ)，FP−PTQ = 12.77 pt。

逐 epoch 的 valid 准确率：

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | 96.46 | 96.72 | 96.31 | 96.67 | 97.08 | 97.38 | 97.38 | 97.23 | 97.33 | 96.87 |
| 10% | 96.82 | 96.97 | 97.49 | 97.84 | 97.84 | 97.95 | 97.74 | 97.84 | 97.90 | 98.05 |

两档都在 **第 2–4 个 epoch 就接近饱和**，后面基本在 97–98% 区间波动；
10% 的曲线整体比 5% 高约 0.3–0.5 pt，最终测试集差距 0.36 pt。

---

## 3. 解读

1. **5% 已经“够用”**：只用了 780 个样本（原训练集的 5%，约 1/20），10 epochs 就把 87% 的 PTQ 模型拉回 98.1%，
   离 FP 仅 1.75 pt。
2. **10% 的边际收益很小**：样本翻倍、训练时间 +70%，只多 0.36 pt。对“快速验证量化策略”来说，
   5% 是更划算的预算。
3. **恢复主要发生在前期**：两档 epoch 1 的 valid 已到 96.5%+，说明 QAT 主要是在修正 qGELU 与低比特带来的
   分布偏移，而不是重新学习任务。
4. **注意模型差异**：本实验用的是 patch（L=100）模型，PTQ 只掉到 87%；pixel 模型（L=1600）的 PTQ 只有
   **15.89%**，退化剧烈得多。pixel 上 5% 数据跑 30 step 只能到 57.3%，说明“序列越长、量化误差累积越严重，
   需要的 QAT 步数越多”。因此本结论应表述为：
   - 对 **L=100 这类短序列**，5% 分层数据 + 10 epochs 足够；
   - 对 **L=1600 的 pixel 模型**，5% 数据是否够用尚未验证（按 30 s/step 估算，5% × 10 epochs 约 8 小时），
     不能直接外推。

---

## 4. 复现命令

```bash
cd /mnt/c/Users/Fsadness/Desktop/models/research/s4d

RUN=experiments/patch/s4d_classification/20260602_182157

# 5%
python scripts/quant/qat_train.py --ckpt-dir $RUN --device cuda \
  --quant-config configs/quant/paper_w4a6.yaml --param-mode discrete \
  --train-fraction 0.05 --epochs 10 --batch-size 32 --eval-batch-size 128 \
  --num-workers 0 --max-eval-samples 0 --eval-every-epoch --calib-batches 4 \
  --seed 0 --tag frac5 --out-dir $RUN/quant_qat_frac5

# 10%
python scripts/quant/qat_train.py --ckpt-dir $RUN --device cuda \
  --quant-config configs/quant/paper_w4a6.yaml --param-mode discrete \
  --train-fraction 0.10 --epochs 10 --batch-size 32 --eval-batch-size 128 \
  --num-workers 0 --max-eval-samples 0 --eval-every-epoch --calib-batches 4 \
  --seed 0 --tag frac10 --out-dir $RUN/quant_qat_frac10
```

产物：`quant_qat_frac5/`、`quant_qat_frac10/` 下的 `quant_best.pth` 与 `qat_report.json`
（`qat_report.json` 含 `train_subset`、逐 epoch `val_acc`、PTQ/QAT 精度与显存估算）。

---

## 5. 附注：旧 config 的数据集大小写修复

patch 实验的 `config.yaml` 里写的是 `dataset: CICIot2022`，而实际目录是 `CICIoT2022`，
导致旧实验无法直接评估。本轮已把 `experiments/**/config.yaml` 中 10 处
`dataset: CICIot2022` 统一修正为 `CICIoT2022`（备份在 `/tmp/s4d_config_backup_casefix_*.tgz`）。
