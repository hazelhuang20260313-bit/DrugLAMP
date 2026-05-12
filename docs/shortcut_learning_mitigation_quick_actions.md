# 缓解模型“片段捷径学习”的快速执行方案

> 目标：让 RORγ/RORC QSAR 模型在学习过程中更关注可迁移的 SAR 结构规律，而不是只要遇到某类官能团、芳香片段或 Morgan bit 就直接判为正样本。本文只整理**当前数据和脚本基础上可以尽快执行**的措施。

---

## 0. 问题定义

当前 baseline 和深度模型可能存在 shortcut learning：

```text
模型发现某些 Morgan bit / 官能团 / 骨架片段在 y=1 中高频出现，
于是学习到：出现该片段 → positive，
而不是学习：该片段在什么 scaffold、取代基环境、理化性质和 SAR 背景下才真正 active。
```

这类问题在以下情况中特别常见：

```text
1. active 来自某些已知 inhibitor 系列
2. inactive 来自结构空间明显不同的分子
3. 0 类并不是真正实验 inactive，而是 unknown / background compounds
4. train 与 predict 之间存在 near-neighbor analog
5. 某些单个 Morgan bit 或 descriptor 对标签区分能力极强
```

---

## 1. 立即执行优先级总览

| 优先级 | 措施 | 目的 | 预计改动 |
|---|---|---|---|
| 最高 | hard negatives | 让模型看到“含相似片段但 inactive”的反例 | 新增小脚本 |
| 最高 | distant subset evaluation | 判断模型是否只会近邻插值 | 小改 `analyze_predictions.py` |
| 最高 | Y-randomization | 排除隐藏泄漏 | 小改 baseline 脚本 |
| 高 | 高偏置 bit / descriptor audit | 找出模型依赖的捷径片段 | 新增小脚本 |
| 高 | bit / fragment dropout | 降低模型对少数片段的依赖 | 小改训练代码 |
| 高 | scaffold / cluster-balanced sampling | 防止大化学系列主导训练 | 小改数据采样 |
| 中 | hard-negative weighted loss | 强化 active-like inactive 的学习 | 小改训练循环 |
| 中 | attribution sanity check | 检查模型到底看了什么结构 | 新增解释脚本 |

---

## 2. 措施一：构建 hard-negative 数据集

### 2.1 目标

如果模型一看到某个片段就判 positive，最有效的修正方法是加入：

```text
含相似片段 / 相似骨架 / 相似 fingerprint 但 y=0 的分子
```

让模型必须学会区分：

```text
active-like active vs active-like inactive
```

而不是：

```text
active-like molecules vs random inactive molecules
```

### 2.2 新增脚本

```text
make_hard_negative_dataset.py
```

### 2.3 推荐输入

```text
input.csv
```

需要至少包含：

```text
Sample 或 canonical_smiles
y
```

### 2.4 核心代码

```python
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd


def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    mol = max(frags, key=lambda m: m.GetNumAtoms(), default=mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def fp(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def nearest_similarity_to_set(query_smiles, ref_smiles):
    ref_fps = [fp(s) for s in ref_smiles]
    ref_fps = [x for x in ref_fps if x is not None]
    scores = []
    for s in query_smiles:
        q = fp(s)
        if q is None or not ref_fps:
            scores.append(np.nan)
        else:
            scores.append(float(max(DataStructs.BulkTanimotoSimilarity(q, ref_fps))))
    return scores


def make_hard_negative_dataset(df, smiles_col="canonical_smiles", label_col="y", threshold=0.4):
    df = df.copy()
    if smiles_col not in df.columns:
        raise ValueError(f"Missing smiles column: {smiles_col}")
    df["canonical_smiles"] = df[smiles_col].apply(standardize_smiles)
    df = df[df["canonical_smiles"].notna()].copy()

    active = df[df[label_col] == 1].copy()
    inactive = df[df[label_col] == 0].copy()

    inactive["nearest_active_tanimoto"] = nearest_similarity_to_set(
        inactive["canonical_smiles"].tolist(),
        active["canonical_smiles"].tolist(),
    )

    hard_inactive = inactive[inactive["nearest_active_tanimoto"] >= threshold].copy()
    hard_inactive["hard_negative_flag"] = True
    active["hard_negative_flag"] = False

    out = pd.concat([active, hard_inactive], ignore_index=True)
    out = out.sample(frac=1, random_state=42).reset_index(drop=True)
    return out
```

### 2.5 推荐阈值

```text
宽松 hard negative: nearest_active_tanimoto >= 0.4
严格 hard negative: nearest_active_tanimoto >= 0.5
更严格 hard negative: nearest_active_tanimoto >= 0.6
```

### 2.6 推荐命令

```bash
python make_hard_negative_dataset.py \
  --input_csv D:/hyl_Data/Luo/input.csv \
  --smiles_col Sample \
  --label_col y \
  --threshold 0.4 \
  --output_csv D:/hyl_Data/Luo/hard_negative_input_t0p4.csv
```

### 2.7 后续操作

对 hard-negative 数据重新 split，再训练：

```bash
python drug_lamp_ror_gamma.py \
  --mode split \
  --input_csv D:/hyl_Data/Luo/hard_negative_input_t0p4.csv \
  --smiles_col canonical_smiles \
  --label_col y
```

然后重新跑 baseline：

```bash
python train_morgan_baselines.py \
  --train_csv D:/hyl_Data/Luo/train.csv \
  --test_csv D:/hyl_Data/Luo/predict.csv \
  --label_col y
```

### 2.8 结果解释

```text
如果 hard-negative 后 baseline 明显下降：
说明原始 inactive 太容易区分，模型之前确实存在片段捷径或系列识别。

如果 hard-negative 后 baseline 仍接近 1：
说明标签和局部结构片段仍高度绑定，需要 endpoint/source/assay curation。
```

---

## 3. 措施二：hard-negative weighted loss

### 3.1 目标

在深度模型训练时，提高 hard negatives 的损失权重，让模型更重视这些“容易被误判为 positive 的 inactive”。

### 3.2 适用脚本

```text
drug_lamp_ror_gamma.py
```

### 3.3 数据要求

训练 CSV 中包含：

```text
hard_negative_flag
```

### 3.4 修改 Dataset

在 `RORCDataset.__getitem__()` 中加入：

```python
sample_weight = 1.0
if "hard_negative_flag" in row and bool(row["hard_negative_flag"]):
    sample_weight = 2.0
```

返回：

```python
"sample_weight": torch.tensor(sample_weight, dtype=torch.float)
```

### 3.5 修改 collate_fn

```python
sample_weights = torch.stack([b["sample_weight"] for b in valid_batch])
```

返回值中加入 `sample_weights`。

### 3.6 修改 train_loop

不要直接使用默认 reduction 的 BCE。改为：

```python
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
```

训练时：

```python
logits = model(batch_prot, drug_inputs, pyg_batch)
loss_vec = criterion(logits, labels)
loss = (loss_vec * sample_weights.to(device)).mean()
```

### 3.7 推荐权重

```text
普通样本 weight = 1.0
hard negative weight = 2.0 或 3.0
```

---

## 4. 措施三：distant subset evaluation

### 4.1 目标

判断模型是否只在近邻 analog 上表现好。

### 4.2 适用脚本

```text
analyze_predictions.py
```

### 4.3 新增参数

```python
parser.add_argument("--distant_threshold", type=float, default=0.6)
parser.add_argument("--very_distant_threshold", type=float, default=0.5)
```

### 4.4 修改 `build_strata()`

```python
if "nearest_train_tanimoto" in df.columns:
    ad = pd.to_numeric(df["nearest_train_tanimoto"], errors="coerce")
    strata.extend([
        (f"Distant train < {args.distant_threshold}", df[ad < args.distant_threshold]),
        (f"Very distant train < {args.very_distant_threshold}", df[ad < args.very_distant_threshold]),
    ])
```

### 4.5 推荐报告层级

```text
All holdout
Non-near train < 0.8
Distant train < 0.6
Very distant train < 0.5
Outside AD < 0.4
```

### 4.6 解释标准

```text
All holdout 接近完美，但 distant <0.6 明显下降：
模型主要依赖近邻插值。

distant <0.6 仍接近完美：
可能存在标签构造偏差、source bias 或强片段规则。
```

---

## 5. 措施四：Y-randomization

### 5.1 目标

确认模型没有隐藏泄漏。如果标签打乱后仍然高分，说明存在严重问题。

### 5.2 适用脚本

```text
train_morgan_baselines.py
```

### 5.3 新增参数

```python
parser.add_argument("--y_randomization", action="store_true")
parser.add_argument("--y_randomization_repeats", type=int, default=50)
parser.add_argument("--y_randomization_output", default="D:/hyl_Data/Luo/y_randomization_report.json")
```

### 5.4 核心代码

```python
from sklearn.base import clone


def run_y_randomization(model, x_train, y_train, x_test, y_test, repeats=50, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(repeats):
        y_shuffle = rng.permutation(y_train)
        m = clone(model)
        m.fit(x_train, y_shuffle)
        prob = m.predict_proba(x_test)[:, 1]
        rows.append({
            "repeat": i,
            "AUROC": float(roc_auc_score(y_test, prob)),
            "AUPRC": float(average_precision_score(y_test, prob)),
        })
    return rows
```

### 5.5 解释标准

```text
AUROC ≈ 0.5：基本排除直接泄漏
AUROC > 0.6：检查 y、split、AD、leakage 等列是否混入特征
```

---

## 6. 措施五：高偏置 Morgan bit / descriptor audit

### 6.1 目标

找出模型可能依赖的捷径特征。

### 6.2 新增脚本

```text
descriptor_bias_audit.py
```

### 6.3 核心代码

```python
from sklearn.metrics import roc_auc_score
import pandas as pd


def single_feature_auc(df, label_col="y", feature_cols=None):
    y = df[label_col].astype(int).to_numpy()
    rows = []
    for c in feature_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        mask = x.notna()
        if mask.sum() < 10 or x[mask].nunique() < 2:
            continue
        auc = roc_auc_score(y[mask], x[mask])
        auc_abs = max(auc, 1 - auc)
        rows.append({
            "feature": c,
            "single_feature_auc": float(auc_abs),
            "active_mean": float(x[df[label_col] == 1].mean()),
            "inactive_mean": float(x[df[label_col] == 0].mean()),
            "active_presence": float((x[df[label_col] == 1] > 0).mean()),
            "inactive_presence": float((x[df[label_col] == 0] > 0).mean()),
        })
    return pd.DataFrame(rows).sort_values("single_feature_auc", ascending=False)
```

### 6.4 推荐 feature 列

```python
feature_cols = [c for c in df.columns if c.startswith("morgan_")]
```

也可以加入 RDKit descriptor，但必须排除：

```text
y
split
split_group
nearest_train_tanimoto
valid_smiles
leakage_warning
known_label
activity_label
```

### 6.5 输出

```text
results/descriptor_bias_report.csv
```

### 6.6 解释标准

```text
single_feature_auc > 0.85：强偏置特征
single_feature_auc > 0.90：高度可疑 shortcut feature
```

---

## 7. 措施六：bit / fragment dropout

### 7.1 目标

训练时随机屏蔽部分 fingerprint bit，防止模型过度依赖少数高频片段。

### 7.2 适用模型

最适合：

```text
Morgan-MLP
任何直接输入 fingerprint 的神经网络
```

对于 RF/XGB/SVM 不适合直接使用训练时 dropout，但可以用于神经网络或增强数据。

### 7.3 普通 bit dropout

```python
def apply_bit_dropout(x, dropout_rate=0.1, training=True):
    if not training or dropout_rate <= 0:
        return x
    mask = torch.rand_like(x) > dropout_rate
    return x * mask.float()
```

### 7.4 高偏置 bit dropout

如果已有 `descriptor_bias_report.csv`，可以对高偏置 bit 使用更高 dropout：

```python
def apply_biased_bit_dropout(x, high_bias_indices, base_rate=0.05, high_bias_rate=0.3):
    mask = torch.ones_like(x)
    base_mask = (torch.rand_like(x) > base_rate).float()
    mask *= base_mask
    if high_bias_indices:
        hb = torch.tensor(high_bias_indices, device=x.device, dtype=torch.long)
        hb_mask = (torch.rand((x.size(0), len(hb)), device=x.device) > high_bias_rate).float()
        mask[:, hb] = hb_mask
    return x * mask
```

### 7.5 推荐参数

```text
普通 bit dropout: 0.05–0.10
高偏置 bit dropout: 0.20–0.30
```

### 7.6 注意事项

```text
不要在最终预测时 dropout，除非做 MC dropout uncertainty。
bit dropout 是 regularization，不是替代数据 curation。
```

---

## 8. 措施七：scaffold / cluster-balanced sampling

### 8.1 目标

防止某些大 scaffold 或大 analog series 主导训练。

### 8.2 可执行方案 A：限制每个 scaffold 最大样本数

```python
def cap_group_size(df, group_col="scaffold", max_per_group=20, seed=42):
    return (
        df.groupby(group_col, group_keys=False)
          .apply(lambda g: g.sample(n=min(len(g), max_per_group), random_state=seed))
          .reset_index(drop=True)
    )
```

### 8.3 可执行方案 B：按 scaffold 反比加权

```python
cluster_size = train_df.groupby("scaffold")["scaffold"].transform("count")
sample_weight = 1.0 / cluster_size
```

用于 XGBoost / RF：

```python
model.fit(x_train, y_train, sample_weight=sample_weight)
```

用于深度模型：

```python
loss = (loss_vec * sample_weight_tensor).mean()
```

### 8.4 解释标准

```text
balanced sampling 后性能明显下降：
原始模型高度依赖大化学系列。

balanced sampling 后性能稳定：
说明规律可能跨多个 scaffold 存在。
```

---

## 9. 措施八：诊断性 feature ablation

### 9.1 目标

判断模型是否依赖少数高偏置 Morgan bit 或 descriptor。

### 9.2 方法

1. 从 `descriptor_bias_report.csv` 中取：

```text
single_feature_auc > 0.85
```

2. 从 feature_cols 中移除这些特征。
3. 重新训练 baseline。

### 9.3 代码示例

```python
bias_df = pd.read_csv("results/descriptor_bias_report.csv")
shortcut_features = set(bias_df[bias_df["single_feature_auc"] > 0.85]["feature"])
feature_cols_filtered = [c for c in feature_cols if c not in shortcut_features]
```

### 9.4 注意

这只是诊断，不是最终建模推荐。因为某些强特征可能是真实 SAR 的一部分。最终更推荐 hard negatives 和 balanced split，而不是永久删除强特征。

---

## 10. 措施九：预测结果加入 shortcut warning

### 10.1 目标

即使模型输出高概率，也要提示该预测是否可能由高偏置片段驱动。

### 10.2 需要输入

```text
descriptor_bias_report.csv
prediction CSV
morgan bit features
```

### 10.3 推荐输出字段

```text
shortcut_feature_count
shortcut_features_present
shortcut_warning
```

### 10.4 逻辑

```python
shortcut_bits = set(bias_df[bias_df["single_feature_auc"] > 0.85]["feature"])

for each molecule:
    present = [bit for bit in shortcut_bits if molecule[bit] > 0]
    shortcut_feature_count = len(present)
    shortcut_warning = shortcut_feature_count >= 1 and predicted_probability >= 0.8
```

### 10.5 解释

```text
高概率 + shortcut_warning=True：
需要降级为“需人工检查 / 需 docking / 需 hard-negative 验证”的候选。
```

---

## 11. 最小可执行工作流

### Step 1：descriptor / Morgan bit bias audit

```bash
python descriptor_bias_audit.py \
  --input_csv D:/hyl_Data/Luo/input.csv \
  --label_col y \
  --feature_prefix morgan_ \
  --output_csv D:/hyl_Data/Luo/descriptor_bias_report.csv
```

---

### Step 2：构建 hard-negative 数据集

```bash
python make_hard_negative_dataset.py \
  --input_csv D:/hyl_Data/Luo/input.csv \
  --smiles_col Sample \
  --label_col y \
  --threshold 0.4 \
  --output_csv D:/hyl_Data/Luo/hard_negative_input_t0p4.csv
```

---

### Step 3：重新 split

```bash
python drug_lamp_ror_gamma.py \
  --mode split \
  --input_csv D:/hyl_Data/Luo/hard_negative_input_t0p4.csv \
  --smiles_col canonical_smiles \
  --label_col y \
  --train_csv D:/hyl_Data/Luo/hard_negative_train.csv \
  --predict_csv D:/hyl_Data/Luo/hard_negative_predict.csv
```

---

### Step 4：重新训练 baseline

```bash
python train_morgan_baselines.py \
  --train_csv D:/hyl_Data/Luo/hard_negative_train.csv \
  --test_csv D:/hyl_Data/Luo/hard_negative_predict.csv \
  --smiles_col canonical_smiles \
  --label_col y \
  --output_predictions D:/hyl_Data/Luo/hard_negative_baseline_predictions.csv \
  --output_metrics D:/hyl_Data/Luo/hard_negative_baseline_metrics.json
```

---

### Step 5：统一评估 baseline

```bash
python analyze_predictions.py \
  --input_csv D:/hyl_Data/Luo/hard_negative_baseline_predictions.csv \
  --label_col y \
  --score_col morgan_xgb_probability \
  --output_json D:/hyl_Data/Luo/hard_negative_xgb_eval.json \
  --output_markdown D:/hyl_Data/Luo/hard_negative_xgb_eval.md \
  --bootstrap 1000
```

---

### Step 6：进行 Y-randomization

```bash
python train_morgan_baselines.py \
  --train_csv D:/hyl_Data/Luo/hard_negative_train.csv \
  --test_csv D:/hyl_Data/Luo/hard_negative_predict.csv \
  --label_col y \
  --y_randomization \
  --y_randomization_repeats 50 \
  --y_randomization_output D:/hyl_Data/Luo/hard_negative_y_randomization.json
```

---

## 12. 结果解释模板

### 情况 A：hard-negative 后性能下降

```text
说明原始模型确实依赖 easy negatives 或片段捷径。
这是好现象，表示新评估更接近真实 QSAR。
```

### 情况 B：hard-negative 后性能仍接近完美

```text
说明标签与结构片段绑定仍然很强。
需要进一步做 endpoint、assay、source 和 true external validation。
```

### 情况 C：Y-randomization 后仍高分

```text
说明存在泄漏或 split 问题，必须先排查 feature_cols。
```

### 情况 D：distant subset 性能下降

```text
说明模型主要适用于近邻 analog interpolation，不能过度声称 novel scaffold 泛化。
```

### 情况 E：shortcut_warning 高的 top hits 很多

```text
说明 top hits 可能主要由高偏置片段驱动，需要优先进行人工 SAR 检查、docking、PAINS/ADMET 和实验验证。
```

---

## 13. 最推荐的短期组合

当前项目最建议优先执行以下组合：

```text
1. descriptor_bias_audit.py
2. make_hard_negative_dataset.py
3. hard-negative split + baseline retraining
4. analyze_predictions.py 增加 distant subset
5. train_morgan_baselines.py 增加 Y-randomization
6. 深度模型训练中加入 hard-negative weighted BCE
```

这套组合可以最快回答三个关键问题：

```text
1. 当前高分是否由少数片段驱动？
2. 模型能否区分 active-like inactive？
3. 模型在远离训练集的分子上是否仍有效？
```

---

## 14. 核心原则

不要试图通过删除所有强特征来“人为降低 baseline”。正确方向是：

```text
让模型看到足够多的反例：
含相同片段但 inactive；
相似结构但不同活性；
同 scaffold 内的 activity cliffs；
远离训练集但标签可靠的分子。
```

这样模型才会从：

```text
某片段出现 → positive
```

逐渐转向：

```text
某结构组合在特定 scaffold、取代基环境、理化性质和 SAR 背景下更可能 active
```
