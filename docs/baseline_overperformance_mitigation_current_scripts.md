# 基于现有脚本缓解 baseline 表现过高的可执行措施

> 目的：在不重新收集外部活性数据的前提下，利用当前已有的 `input.csv / train.csv / predict.csv`、`train_morgan_baselines.py`、`analyze_predictions.py`、`external_independence_audit.py` 和 `drug_lamp_ror_gamma.py`，尽快排查并缓解 Morgan RF / SVM / XGBoost baseline 接近完美的问题。

---

## 0. 当前问题判断

当前 baseline 表现极高，可能不是单纯代码错误，而是当前数据结构使任务过于容易：

```text
1. 标签 y 与部分 Morgan bit / descriptor 强相关
2. train.csv 和 predict.csv 都来自同一个 input.csv
3. active / inactive 可能来自不同化学系列或不同来源
4. predict set 中仍存在较多 near-neighbor analog
5. 0 类可能并非严格实验确认 inactive
6. 若训练脚本误用全部数值列，可能把 y 或 split 后辅助列混入 X
```

本文件只整理**目前脚本中可以较快执行或小幅修改即可执行**的措施。

---

## 1. 措施一：强制确认没有标签列或辅助列进入特征

### 1.1 目标

排除最直接的数据泄漏。

### 1.2 适用脚本

```text
train_morgan_baselines.py
任何后续新增 baseline / ML 脚本
```

### 1.3 当前建议

在 `train_morgan_baselines.py` 中，当前 `featurize()` 会优先读取 `morgan_0` 到 `morgan_2047`，这本身是安全的。但仍建议加入显式断言，防止未来改脚本时误用全部数值列。

### 1.4 建议加入代码

```python
FORBIDDEN_FEATURE_COLUMNS = {
    "y",
    "known_label",
    "activity_label",
    "label",
    "split",
    "split_group",
    "nearest_train_tanimoto",
    "within_applicability_domain",
    "near_training_neighbor",
    "leakage_warning",
    "valid_smiles",
}


def assert_no_leakage_features(feature_cols):
    leaked = sorted(set(feature_cols) & FORBIDDEN_FEATURE_COLUMNS)
    if leaked:
        raise ValueError(f"Potential leakage columns found in features: {leaked}")
```

在 `featurize()` 返回后加入：

```python
x_train, feature_cols = featurize(train_df)
assert_no_leakage_features(feature_cols)
```

### 1.5 不推荐写法

```python
X = df.select_dtypes(include=[np.number])
X = df.drop(columns=["Sample"])
```

这两种写法很容易把 `y`、`nearest_train_tanimoto`、`split_group` 等列混入模型。

---

## 2. 措施二：Y-randomization 检查

### 2.1 目标

验证 baseline 高分是否可能由隐藏泄漏或非随机结构偏差造成。

### 2.2 适用脚本

```text
train_morgan_baselines.py
```

### 2.3 新增参数

```python
parser.add_argument("--y_randomization", action="store_true")
parser.add_argument("--y_randomization_repeats", type=int, default=50)
parser.add_argument("--y_randomization_output", default="D:/hyl_Data/Luo/y_randomization_report.json")
```

### 2.4 建议代码

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
            "AUROC": float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) > 1 else None,
            "AUPRC": float(average_precision_score(y_test, prob)) if len(np.unique(y_test)) > 1 else None,
        })
    return rows
```

在 RF / SVM / XGB 训练后运行：

```python
if args.y_randomization:
    yr = {}
    for name, model in models.items():
        yr[name] = run_y_randomization(
            model, x_train, y_train, x_test, y_test,
            repeats=args.y_randomization_repeats,
            seed=args.seed,
        )
    with open(args.y_randomization_output, "w", encoding="utf-8") as f:
        json.dump(yr, f, indent=2)
```

### 2.5 解释标准

```text
理想结果：AUROC ≈ 0.5，AUPRC ≈ 测试集 active ratio
若 AUROC 明显 > 0.6：优先排查标签泄漏、split 泄漏或特征构造错误
```

---

## 3. 措施三：在现有 predict.csv 中构造远邻测试子集

### 3.1 目标

判断 baseline 高分是否主要来自 train-predict 间 analog interpolation。

### 3.2 适用脚本

```text
analyze_predictions.py
baseline_predictions.csv
holdout_predictions.csv
```

### 3.3 当前已有能力

`analyze_predictions.py` 已支持：

```text
Within AD >= 0.4
Outside AD < 0.4
Near train >= 0.8
Non-near train < 0.8
```

但建议进一步增加更严格的 distant bins。

### 3.4 建议增加 CLI 参数

```python
parser.add_argument("--distant_threshold", type=float, default=0.6)
parser.add_argument("--very_distant_threshold", type=float, default=0.5)
```

### 3.5 修改 `build_strata()`

```python
if "nearest_train_tanimoto" in df.columns:
    ad = pd.to_numeric(df["nearest_train_tanimoto"], errors="coerce")
    strata.extend([
        (f"Distant train < {args.distant_threshold}", df[ad < args.distant_threshold]),
        (f"Very distant train < {args.very_distant_threshold}", df[ad < args.very_distant_threshold]),
    ])
```

### 3.6 推荐报告方式

```text
All holdout
Non-near train < 0.8
Distant train < 0.6
Very distant train < 0.5
```

### 3.7 解释标准

```text
如果 All holdout AUROC≈1，但 distant <0.6 明显下降：
说明原始高分主要来自近邻化学空间插值。

如果 distant <0.6 仍 AUROC≈1：
重点怀疑 source bias、label bias 或 active/inactive 本身由显著不同结构空间构成。
```

---

## 4. 措施四：使用 external_independence_audit.py 加强 near-neighbor 风险分级

### 4.1 目标

不只判断 pass/fail，还给出 near-neighbor 风险等级。

### 4.2 当前已有结果

当前审计已包含：

```text
exact_overlap_count
inchikey_overlap_count
same_scaffold_count
near_neighbor_count_tanimoto_ge_0_8
leakage_count_tanimoto_ge_0_999
nearest_tanimoto_summary
pass_strict_independence
```

### 4.3 建议新增函数

```python
def assign_external_risk_grade(report):
    n = max(1, report["n_external"])
    near_rate = report["near_neighbor_count_tanimoto_ge_0_8"] / n

    if report["exact_overlap_count"] > 0 or report["leakage_count_tanimoto_ge_0_999"] > 0:
        return "D", "Exact overlap or leakage detected."
    if near_rate < 0.10:
        return "A", "Low near-neighbor burden."
    if near_rate < 0.30:
        return "B", "Moderate near-neighbor burden; report non-near subset separately."
    return "C", "High near-neighbor burden; not suitable as strong external validation."
```

在 report 中加入：

```python
grade, grade_note = assign_external_risk_grade(report)
report["external_independence_grade"] = grade
report["external_independence_grade_note"] = grade_note
```

### 4.4 解释标准

```text
Grade A：可作为较强 scaffold/chemical-space holdout 证据
Grade B：可用，但必须报告 non-near subset
Grade C：只适合做内部 holdout，不适合强泛化宣称
Grade D：存在泄漏，不能作为验证结果
```

---

## 5. 措施五：构造 hard-negative 测试集

### 5.1 目标

避免 inactive 与 active 结构空间差异过大导致分类过易。

### 5.2 适用数据

```text
input.csv 或 train.csv / predict.csv
```

### 5.3 基本思想

从 inactive 中挑选与 active 相似的分子作为 hard negatives：

```text
inactive_nearest_active_tanimoto >= 0.4 或 >= 0.5
```

这样模型必须区分“结构相近但标签不同”的样本。

### 5.4 建议新增脚本

```text
make_hard_negative_dataset.py
```

### 5.5 关键代码

```python
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd


def fp(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def nearest_similarity_to_set(smiles_list, ref_smiles_list):
    ref_fps = [fp(s) for s in ref_smiles_list]
    ref_fps = [x for x in ref_fps if x is not None]
    scores = []
    for s in smiles_list:
        q = fp(s)
        if q is None or not ref_fps:
            scores.append(np.nan)
        else:
            scores.append(float(max(DataStructs.BulkTanimotoSimilarity(q, ref_fps))))
    return scores


def make_hard_negative_dataset(df, smiles_col="canonical_smiles", label_col="y", threshold=0.4):
    active = df[df[label_col] == 1].copy()
    inactive = df[df[label_col] == 0].copy()
    inactive["nearest_active_tanimoto"] = nearest_similarity_to_set(
        inactive[smiles_col].tolist(),
        active[smiles_col].tolist(),
    )
    hard_inactive = inactive[inactive["nearest_active_tanimoto"] >= threshold].copy()
    out = pd.concat([active, hard_inactive], ignore_index=True)
    return out.sample(frac=1, random_state=42).reset_index(drop=True)
```

### 5.6 推荐输出

```text
hard_negative_input_tanimoto_0p4.csv
hard_negative_input_tanimoto_0p5.csv
```

### 5.7 后续执行

用 hard-negative 数据重新 split，再训练 baseline：

```bash
python train_morgan_baselines.py \
  --train_csv hard_negative_train.csv \
  --test_csv hard_negative_predict.csv \
  --label_col y
```

### 5.8 解释标准

```text
如果 hard-negative 数据上 baseline 明显下降：
说明原始数据中 inactive 太容易区分。

如果 hard-negative 数据上 baseline 仍接近 1：
说明标签和局部结构片段仍强绑定，需进一步做 endpoint/source/assay curation。
```

---

## 6. 措施六：构造 property-matched negative 数据集

### 6.1 目标

减少 active/inactive 在基础理化性质上的差异。

### 6.2 可直接利用的列

如果 `input.csv` 中已有 RDKit descriptor，可直接使用：

```text
MolWt / exact molecular weight 类字段
MolLogP / LogP 类字段
TPSA
NumHDonors / HBD
NumHAcceptors / HBA
NumRotatableBonds / RotB
NumAromaticRings / aromatic ring count
```

不同数据文件列名可能不同，脚本中应做候选列匹配。

### 6.3 建议新增脚本

```text
make_property_matched_dataset.py
```

### 6.4 关键代码框架

```python
def find_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def make_property_matched_set(df, label_col="y", n_neg_per_pos=2, seed=42):
    mw_col = find_first_existing(df, ["MolWt", "MW", "ExactMolWt"])
    logp_col = find_first_existing(df, ["MolLogP", "LogP", "CrippenClogP"])
    tpsa_col = find_first_existing(df, ["TPSA"])

    if not all([mw_col, logp_col, tpsa_col]):
        raise ValueError("Required property columns were not found.")

    active = df[df[label_col] == 1].copy()
    inactive = df[df[label_col] == 0].copy()
    selected = []

    rng = np.random.default_rng(seed)
    for _, a in active.iterrows():
        cand = inactive[
            (inactive[mw_col].sub(a[mw_col]).abs() < 50) &
            (inactive[logp_col].sub(a[logp_col]).abs() < 1.0) &
            (inactive[tpsa_col].sub(a[tpsa_col]).abs() < 30)
        ]
        if len(cand) == 0:
            continue
        n = min(n_neg_per_pos, len(cand))
        selected.extend(rng.choice(cand.index.to_numpy(), size=n, replace=False).tolist())

    matched = pd.concat([active, inactive.loc[sorted(set(selected))]], ignore_index=True)
    return matched.sample(frac=1, random_state=seed).reset_index(drop=True)
```

### 6.5 推荐输出

```text
property_matched_input.csv
property_matched_train.csv
property_matched_predict.csv
property_matching_report.json
```

### 6.6 解释标准

```text
如果 property-matched 后 baseline 明显下降：
说明原始数据存在明显 property bias。

如果仍接近 1：
说明更可能是特定片段、系列或标签来源造成的强相关。
```

---

## 7. 措施七：限制大 scaffold / cluster 的主导效应

### 7.1 目标

避免某些大化学系列主导模型学习。

### 7.2 当前可用字段

如果 `train.csv / predict.csv` 已有 `scaffold` 列，可直接使用。否则用 RDKit 生成。

### 7.3 下采样方案

每个 scaffold 最多保留 N 个样本：

```python
def cap_group_size(df, group_col="scaffold", max_per_group=20, seed=42):
    return (
        df.groupby(group_col, group_keys=False)
          .apply(lambda g: g.sample(n=min(len(g), max_per_group), random_state=seed))
          .reset_index(drop=True)
    )
```

### 7.4 加权方案

按 scaffold size 反比加权：

```python
cluster_size = train_df.groupby("scaffold")["scaffold"].transform("count")
sample_weight = 1.0 / cluster_size
```

用于 RF / XGB：

```python
model.fit(x_train, y_train, sample_weight=sample_weight)
```

### 7.5 解释标准

```text
如果限制大 scaffold 后性能明显下降：
说明原始模型依赖大系列记忆。

如果性能稳定：
说明规律可能跨多个 scaffold 存在。
```

---

## 8. 措施八：单特征 AUC / descriptor bias audit

### 8.1 目标

识别是否存在单个 descriptor 或 Morgan bit 就能高度区分标签。

### 8.2 建议新增脚本

```text
descriptor_bias_audit.py
```

### 8.3 关键代码

```python
from sklearn.metrics import roc_auc_score


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
        })
    return pd.DataFrame(rows).sort_values("single_feature_auc", ascending=False)
```

### 8.4 推荐执行

```text
feature_cols = morgan_* + descriptor columns
```

输出：

```text
descriptor_bias_report.csv
```

### 8.5 解释标准

```text
单特征 AUC > 0.85：强警示
单特征 AUC > 0.90：说明标签可能与某个片段/性质高度绑定
```

这不是说要永久删除这些特征，而是要证明当前任务是否被少数结构片段主导。

---

## 9. 措施九：诊断性 feature-ablation

### 9.1 目标

判断 baseline 是否依赖少数过强特征。

### 9.2 方法

1. 用 `descriptor_bias_report.csv` 找到：

```text
single_feature_auc > 0.85
```

2. 从特征中移除这些列。
3. 重新训练 baseline。

### 9.3 注意

这只是诊断，不是最终推荐建模方案。因为某些强特征可能确实是 SAR 的一部分。

### 9.4 解释标准

```text
移除强特征后性能大幅下降：
模型主要依赖少数片段/性质。

移除强特征后性能仍很高：
标签与整体结构空间仍高度相关。
```

---

## 10. 措施十：统一 baseline 评估到 analyze_predictions.py

### 10.1 目标

让 baseline 和深度模型使用同一套评估逻辑，避免指标不可比。

### 10.2 当前问题

`train_morgan_baselines.py` 输出 `baseline_predictions.csv`，列名通常是：

```text
morgan_rf_probability
morgan_svm_probability
morgan_xgb_probability
```

而 `analyze_predictions.py` 默认寻找：

```text
calibrated_probability
RORC_binding_score_raw
```

### 10.3 推荐执行方式

分别指定 score_col：

```bash
python analyze_predictions.py \
  --input_csv D:/hyl_Data/Luo/baseline_predictions.csv \
  --label_col y \
  --score_col morgan_rf_probability \
  --output_json D:/hyl_Data/Luo/eval_morgan_rf.json \
  --output_markdown D:/hyl_Data/Luo/eval_morgan_rf.md \
  --bootstrap 1000
```

```bash
python analyze_predictions.py \
  --input_csv D:/hyl_Data/Luo/baseline_predictions.csv \
  --label_col y \
  --score_col morgan_xgb_probability \
  --output_json D:/hyl_Data/Luo/eval_morgan_xgb.json \
  --output_markdown D:/hyl_Data/Luo/eval_morgan_xgb.md \
  --bootstrap 1000
```

### 10.4 建议改进

让 `choose_score_cols()` 自动识别：

```python
for col in df.columns:
    if col.endswith("_probability"):
        candidates.append(col)
```

这样可以一次性评估所有 baseline 概率列。

---

## 11. 建议当前立即执行的最小工作流

### Step 1：特征泄漏检查

```bash
python train_morgan_baselines.py \
  --train_csv D:/hyl_Data/Luo/train.csv \
  --test_csv D:/hyl_Data/Luo/predict.csv \
  --label_col y
```

同时确保脚本中加入：

```python
assert_no_leakage_features(feature_cols)
```

---

### Step 2：Y-randomization

```bash
python train_morgan_baselines.py \
  --train_csv D:/hyl_Data/Luo/train.csv \
  --test_csv D:/hyl_Data/Luo/predict.csv \
  --label_col y \
  --y_randomization \
  --y_randomization_repeats 50 \
  --y_randomization_output D:/hyl_Data/Luo/y_randomization_report.json
```

---

### Step 3：baseline 统一评估

```bash
python analyze_predictions.py \
  --input_csv D:/hyl_Data/Luo/baseline_predictions.csv \
  --label_col y \
  --score_col morgan_xgb_probability \
  --output_json D:/hyl_Data/Luo/eval_morgan_xgb_bootstrap.json \
  --output_markdown D:/hyl_Data/Luo/eval_morgan_xgb_bootstrap.md \
  --bootstrap 1000
```

---

### Step 4：远邻子集评估

先在 `analyze_predictions.py` 增加：

```text
Distant train < 0.6
Very distant train < 0.5
```

再重复 Step 3。

---

### Step 5：hard-negative 数据集

```bash
python make_hard_negative_dataset.py \
  --input_csv D:/hyl_Data/Luo/input.csv \
  --smiles_col Sample \
  --label_col y \
  --threshold 0.4 \
  --output_csv D:/hyl_Data/Luo/hard_negative_input_t0p4.csv
```

然后重新 split 和 baseline。

---

### Step 6：property-matched 数据集

```bash
python make_property_matched_dataset.py \
  --input_csv D:/hyl_Data/Luo/input.csv \
  --label_col y \
  --output_csv D:/hyl_Data/Luo/property_matched_input.csv
```

然后重新 split 和 baseline。

---

## 12. 结果解释模板

### 情况 A：Y-randomization 后 AUROC≈0.5

说明无明显直接泄漏，baseline 高分主要来自真实结构-标签关联或数据构造偏差。

### 情况 B：远邻测试集性能下降

说明原始高分主要来自近邻 analog interpolation。应重点报告 distant subset。

### 情况 C：property-matched 后性能下降

说明 active/inactive 的基础理化性质差异推动了高分。需要 matched negatives 或 source-balanced negatives。

### 情况 D：hard-negative 后性能下降

说明原始 inactive 过于容易，模型在更真实的 active-like inactive 上泛化有限。

### 情况 E：所有严格测试中性能仍接近 1

需要重点检查：

```text
1. 标签是否来自化学系列规则
2. 0 类是否是真正 inactive
3. active/inactive 是否来自不同数据源
4. assay/source/year 是否混杂
5. 是否需要 true external experimental set
```

---

## 13. 优先级总结

| 优先级 | 措施 | 是否需大改代码 | 目的 |
|---|---|---:|---|
| 最高 | assert 排除 y / split / AD 列 | 否 | 排除直接泄漏 |
| 最高 | Y-randomization | 小改 | 排查隐藏泄漏 |
| 最高 | baseline 接入 analyze_predictions.py | 否/小改 | 统一评估 |
| 高 | distant subset 评估 | 小改 | 判断近邻插值 |
| 高 | external risk grade | 小改 | 提高验证解释性 |
| 高 | hard-negative dataset | 新增小脚本 | 增加任务难度 |
| 高 | property-matched dataset | 新增小脚本 | 减少性质偏差 |
| 中 | scaffold/cluster size cap | 小改 | 减少大系列主导 |
| 中 | descriptor bias audit | 新增小脚本 | 识别强标签相关片段 |
| 中 | feature-ablation | 小改 | 诊断少数特征依赖 |

---

## 14. 最终原则

不要为了让 baseline 变低而削弱模型；应该通过更严格的数据构造和评估方式，让结果更接近真实 QSAR 泛化：

```text
从：active 系列 vs 明显不同的 inactive
转向：结构相近、性质匹配、无近邻泄漏、来源一致的 active vs inactive
```

在现有数据上，最推荐立即执行：

```text
1. 标签/辅助列泄漏断言
2. Y-randomization
3. Distant subset 评估
4. Hard-negative 构造
5. Property-matched negative 构造
6. Baseline 与深度模型统一评估
```
