# RORγ/RORC QSAR 快速代码落地优化方案

> 目标：将当前 `drug_lamp_ror_gamma.py` 与 `analyze_predictions.py` 快速升级为更接近高水平 QSAR / CNS 类期刊要求的可复现工作流。本文只整理“可以较快通过代码实现”的部分，不包含需要长期实验验证或大规模人工整理的内容。

---

## 0. 当前脚本基础与主要缺口

当前主脚本已经包含：

- SMILES canonicalization
- Murcko scaffold split
- GINE 分子图编码
- ChemBERTa 药物序列编码
- ESM2 RORC LBD 蛋白嵌入
- PGCA / PMMA 注意力模块
- BCEWithLogitsLoss + class weight
- Isotonic calibration
- checkpoint metadata
- nearest-training Tanimoto AD
- leakage warning

当前评估脚本已经包含：

- AUROC
- AUPRC
- EF1%
- EF5%
- BEDROC
- Brier score
- ECE
- AD 内外分层
- near-training / non-near-training 分层
- high-confidence non-near-training hits 输出

但仍缺少以下快速可实现模块：

1. true external validation 独立性审计
2. 更完整的分子标准化、去重与冲突标签处理
3. Butina split
4. baseline 模型：Morgan + RF / XGBoost / SVM
5. bootstrap 置信区间
6. uncertainty estimation
7. ablation study
8. top hits PAINS / ADMET / SA 初筛
9. JSON 化评估报告与可复现 trace

---

## 1. 新增 `curate_rorc_data.py`：分子标准化、重复合并、冲突标签处理

### 1.1 目标

将原始 RORγ/RORC 活性数据统一为可训练的 QSAR 表格。

### 1.2 输入字段建议

```text
smiles
activity_type       # IC50, Ki, Kd, EC50, AC50
activity_value
activity_unit       # nM, uM, M
relation            # =, <, >, <=, >=
assay_id
assay_type          # binding, reporter, coactivator, etc.
source_database
publication_year
```

### 1.3 输出文件

```text
data/curated/rorc_activity_master.csv
results/curation_report.json
results/invalid_smiles.csv
results/duplicate_merged.csv
results/conflict_compounds.csv
```

### 1.4 关键代码实现

```python
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np
import pandas as pd

fragment_chooser = rdMolStandardize.LargestFragmentChooser()
normalizer = rdMolStandardize.Normalizer()
reionizer = rdMolStandardize.Reionizer()
uncharger = rdMolStandardize.Uncharger()
tautomer_enumerator = rdMolStandardize.TautomerEnumerator()


def standardize_molecule(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None, None
    mol = fragment_chooser.choose(mol)
    mol = normalizer.normalize(mol)
    mol = reionizer.reionize(mol)
    mol = uncharger.uncharge(mol)
    mol = tautomer_enumerator.Canonicalize(mol)
    smiles_std = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    inchikey = Chem.MolToInchiKey(mol)
    inchi = Chem.MolToInchi(mol)
    return smiles_std, inchikey, inchi


def convert_to_molar(value, unit):
    unit = str(unit).lower().replace('μ', 'u')
    value = float(value)
    if unit == 'm':
        return value
    if unit == 'mm':
        return value * 1e-3
    if unit in ['um', 'µm']:
        return value * 1e-6
    if unit == 'nm':
        return value * 1e-9
    if unit == 'pm':
        return value * 1e-12
    return np.nan


def make_pactivity(value, unit):
    molar = convert_to_molar(value, unit)
    if pd.isna(molar) or molar <= 0:
        return np.nan
    return -np.log10(molar)


def assign_label(pactivity, active_cutoff=6.0, inactive_cutoff=5.0):
    if pd.isna(pactivity):
        return np.nan
    if pactivity >= active_cutoff:
        return 1
    if pactivity < inactive_cutoff:
        return 0
    return np.nan  # gray zone
```

### 1.5 重复合并逻辑

```python
def merge_duplicates(df):
    grouped = []
    conflicts = []
    for inchikey, g in df.groupby('standard_inchikey'):
        values = g['p_activity'].dropna().values
        if len(values) == 0:
            continue
        spread = values.max() - values.min()
        row = g.iloc[0].copy()
        row['p_activity_median'] = float(np.median(values))
        row['activity_count'] = int(len(values))
        row['activity_spread'] = float(spread)
        row['conflict_flag'] = bool(spread >= 1.0)
        row['activity_label'] = assign_label(row['p_activity_median'])
        if spread >= 1.0:
            conflicts.append(g)
        grouped.append(row)
    merged = pd.DataFrame(grouped)
    conflict_df = pd.concat(conflicts) if conflicts else pd.DataFrame()
    return merged, conflict_df
```

### 1.6 快速落地优先级

优先级：最高。

原因：如果数据 curation 不清楚，后续深度模型结果再好也不可信。

---

## 2. 新增 `external_independence_audit.py`：true external validation 独立性审计

### 2.1 目标

在运行外部测试前，确认 external set 没有泄漏到训练集。

### 2.2 输入

```text
train.csv
external_test.csv
```

### 2.3 输出

```text
results/external_independence_audit.json
results/external_overlap_records.csv
```

### 2.4 检查项目

1. canonical SMILES exact overlap
2. InChIKey overlap
3. scaffold overlap
4. nearest train Tanimoto
5. Tanimoto >= 0.8 near-neighbor rate
6. Tanimoto >= 0.999 leakage count
7. assay_id overlap
8. source / publication overlap

### 2.5 关键代码

```python
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import numpy as np
import json


def ecfp4(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def scaffold(smiles):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return None


def nearest_tanimoto(external_smiles, train_smiles):
    train_fps = [(s, ecfp4(s)) for s in train_smiles]
    train_fps = [(s, fp) for s, fp in train_fps if fp is not None]
    scores = []
    nearest = []
    for s in external_smiles:
        fp = ecfp4(s)
        if fp is None or not train_fps:
            scores.append(np.nan)
            nearest.append(None)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, [x[1] for x in train_fps])
        idx = int(np.argmax(sims))
        scores.append(float(sims[idx]))
        nearest.append(train_fps[idx][0])
    return scores, nearest


def audit_external(train_df, external_df):
    train_smiles = set(train_df['canonical_smiles'])
    ext_smiles = set(external_df['canonical_smiles'])

    exact_overlap = ext_smiles & train_smiles

    train_scaffolds = set(train_df['canonical_smiles'].apply(scaffold))
    ext_scaffolds = external_df['canonical_smiles'].apply(scaffold)

    nt, nearest = nearest_tanimoto(
        external_df['canonical_smiles'].tolist(),
        train_df['canonical_smiles'].tolist()
    )

    external_df = external_df.copy()
    external_df['external_nearest_train_tanimoto'] = nt
    external_df['external_nearest_train_smiles'] = nearest
    external_df['external_same_scaffold_as_train'] = ext_scaffolds.isin(train_scaffolds)
    external_df['external_exact_overlap'] = external_df['canonical_smiles'].isin(exact_overlap)
    external_df['external_near_neighbor'] = external_df['external_nearest_train_tanimoto'] >= 0.8
    external_df['external_leakage'] = external_df['external_nearest_train_tanimoto'] >= 0.999

    report = {
        'n_train': int(len(train_df)),
        'n_external': int(len(external_df)),
        'exact_overlap_count': int(external_df['external_exact_overlap'].sum()),
        'same_scaffold_count': int(external_df['external_same_scaffold_as_train'].sum()),
        'near_neighbor_count_tanimoto_ge_0_8': int(external_df['external_near_neighbor'].sum()),
        'leakage_count_tanimoto_ge_0_999': int(external_df['external_leakage'].sum()),
        'nearest_tanimoto_summary': external_df['external_nearest_train_tanimoto'].describe().to_dict(),
    }
    return external_df, report
```

### 2.6 判定标准

```text
exact_overlap_count 必须为 0
leakage_count_tanimoto_ge_0_999 必须为 0
near_neighbor 样本可以保留，但必须单独分层报告
same_scaffold 样本可以保留，但不能声称是 novel scaffold validation
```

---

## 3. 新增 Butina split：替代或补充 KMeans

### 3.1 目标

建立比随机 split 更接近真实外推的化学簇划分。

### 3.2 输出

```text
data/splits/butina_train.csv
data/splits/butina_valid.csv
data/splits/butina_test.csv
results/butina_split_report.json
```

### 3.3 关键代码

```python
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
import numpy as np
import random


def butina_clusters(smiles_list, cutoff=0.4):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m is not None else None for m in mols]
    valid = [(i, fp) for i, fp in enumerate(fps) if fp is not None]

    dists = []
    for i in range(1, len(valid)):
        sims = DataStructs.BulkTanimotoSimilarity(valid[i][1], [valid[j][1] for j in range(i)])
        dists.extend([1 - x for x in sims])

    clusters = Butina.ClusterData(dists, len(valid), cutoff, isDistData=True)
    index_clusters = [[valid[i][0] for i in c] for c in clusters]
    return index_clusters


def split_clusters(clusters, frac_train=0.7, frac_valid=0.15, seed=42):
    random.seed(seed)
    clusters = sorted(clusters, key=len, reverse=True)
    train, valid, test = [], [], []
    n_total = sum(len(c) for c in clusters)
    for c in clusters:
        if len(train) + len(c) <= frac_train * n_total:
            train.extend(c)
        elif len(valid) + len(c) <= frac_valid * n_total:
            valid.extend(c)
        else:
            test.extend(c)
    return train, valid, test
```

### 3.4 推荐参数

```text
cutoff = 0.4 distance，即 Tanimoto similarity >= 0.6 聚为一类
```

---

## 4. 新增 Morgan baseline：RF / XGBoost / SVM

### 4.1 目标

证明复杂深度模型优于传统 QSAR baseline，或至少给出公平比较。

### 4.2 输出

```text
models/morgan_rf.pkl
models/morgan_xgb.pkl
models/morgan_svm.pkl
results/baseline_predictions.csv
results/baseline_metrics.json
```

### 4.3 关键代码

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV


def morgan_array(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    arr = np.zeros((n_bits,), dtype=np.int8)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
    for i in fp.GetOnBits():
        arr[i] = 1
    return arr


def featurize(df):
    return np.vstack([morgan_array(s) for s in df['canonical_smiles']])


def train_rf(train_df, valid_df, label_col='activity_label'):
    x_train = featurize(train_df)
    y_train = train_df[label_col].astype(int).values
    x_valid = featurize(valid_df)
    y_valid = valid_df[label_col].astype(int).values

    rf = RandomForestClassifier(
        n_estimators=500,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(x_train, y_train)
    prob = rf.predict_proba(x_valid)[:, 1]
    metrics = {
        'AUROC': roc_auc_score(y_valid, prob),
        'AUPRC': average_precision_score(y_valid, prob),
        'Brier': brier_score_loss(y_valid, prob)
    }
    return rf, metrics


def train_svm(train_df, valid_df, label_col='activity_label'):
    x_train = featurize(train_df)
    y_train = train_df[label_col].astype(int).values
    svm = SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced')
    svm.fit(x_train, y_train)
    return svm
```

### 4.4 XGBoost 版本

如果已安装 `xgboost`：

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
```

---

## 5. 扩展 `analyze_predictions.py`：输出 JSON + bootstrap 95% CI

### 5.1 目标

让评估结果可复现、可放入论文表格。

### 5.2 新增 CLI 参数

```python
parser.add_argument('--output_json', default='evaluation_report.json')
parser.add_argument('--bootstrap', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42)
```

### 5.3 关键代码

```python
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


def bootstrap_ci(labels, scores, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    vals = []
    n = len(labels)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        vals.append(metric_fn(labels[idx], scores[idx]))
    if not vals:
        return {'mean': None, 'low': None, 'high': None}
    vals = np.asarray(vals)
    return {
        'mean': float(np.mean(vals)),
        'low': float(np.percentile(vals, 2.5)),
        'high': float(np.percentile(vals, 97.5))
    }
```

### 5.4 JSON 输出格式

```json
{
  "All labeled": {
    "AUROC": 0.91,
    "AUROC_CI95": [0.86, 0.95],
    "AUPRC": 0.88,
    "AUPRC_CI95": [0.82, 0.93]
  },
  "Within AD": {},
  "Outside AD": {},
  "Non-near train": {}
}
```

---

## 6. 新增 uncertainty estimation：ensemble + MC dropout

### 6.1 快速方案 A：多 seed ensemble

训练 5 个模型：

```bash
python drug_lamp_ror_gamma.py --mode train --seed 1 --checkpoint models/seed1.pt
python drug_lamp_ror_gamma.py --mode train --seed 2 --checkpoint models/seed2.pt
python drug_lamp_ror_gamma.py --mode train --seed 3 --checkpoint models/seed3.pt
python drug_lamp_ror_gamma.py --mode train --seed 4 --checkpoint models/seed4.pt
python drug_lamp_ror_gamma.py --mode train --seed 5 --checkpoint models/seed5.pt
```

预测时合并：

```python
def summarize_ensemble(prob_matrix):
    # shape: n_models × n_samples
    mean = prob_matrix.mean(axis=0)
    std = prob_matrix.std(axis=0)
    entropy = -(mean * np.log(mean + 1e-8) + (1 - mean) * np.log(1 - mean + 1e-8))
    return mean, std, entropy
```

输出字段：

```text
prob_mean
prob_std
prob_entropy
uncertainty_flag
```

判定：

```text
prob_mean >= 0.8 且 prob_std <= 0.1：high-confidence active
prob_mean <= 0.2 且 prob_std <= 0.1：high-confidence inactive
prob_std > 0.15：uncertain
```

### 6.2 快速方案 B：MC dropout

预测时开启 dropout：

```python
def mc_dropout_predict(model, batch, n_passes=30):
    model.train()  # keep dropout active
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(*batch)
            preds.append(torch.sigmoid(logits).cpu().numpy())
    preds = np.vstack(preds)
    return preds.mean(axis=0), preds.std(axis=0)
```

注意：BatchNorm 模型中 MC dropout 要小心；你当前模型主要是 LayerNorm / Dropout，较适合快速实现。

---

## 7. 新增 ablation study

### 7.1 目标

证明 PGCA / PMMA / protein embedding / GINE / ChemBERTa 各模块是否必要。

### 7.2 建议新增参数

```python
parser.add_argument('--disable_protein', action='store_true')
parser.add_argument('--disable_pgca', action='store_true')
parser.add_argument('--disable_pmma', action='store_true')
parser.add_argument('--disable_gine', action='store_true')
parser.add_argument('--disable_chemberta', action='store_true')
parser.add_argument('--freeze_chemberta', action='store_true')
```

### 7.3 模型内实现逻辑

```python
if args.freeze_chemberta:
    for p in model.chemberta.parameters():
        p.requires_grad = False
```

融合层建议改为：

```python
features = []
if not disable_chemberta:
    features.append(aligned_plm.squeeze(1))
if not disable_gine:
    features.append(aligned_gcn.squeeze(1))

f_combined = torch.cat(features, dim=-1)
```

分类器输入维度根据启用模块自动调整。

### 7.4 必做 ablation 表

```text
ChemBERTa only
GINE only
ChemBERTa + GINE
ChemBERTa + GINE + global LBD
ChemBERTa + GINE + pocket LBD
Full model without PGCA
Full model without PMMA
Full model
```

---

## 8. 优化 PGCA / PMMA 注意力模块

### 8.1 当前问题

当前 PGCA 和 PMMA 都是简单 MultiheadAttention + residual + MLP。PMMA 中 graph representation 是单 token：

```python
d_feats_graph.unsqueeze(1)
```

这会导致 PMMA 的 key/value 长度为 1，注意力实际上退化为“蛋白 token 对单个图向量的广播对齐”，解释性有限。

### 8.2 快速优化方案

#### 方案 A：GINE 输出 node-level embedding

修改 `DrugGINEEncoder.forward()`，让它同时返回：

```text
graph_pool_embedding
node_embeddings
node_batch_index
```

```python
return global_mean_pool(x, batch), x, batch
```

PMMA 使用 node embeddings 作为 key/value，而不是 pooled graph embedding。

#### 方案 B：attention pooling 加 gating fusion

新增融合层：

```python
self.fusion_gate = nn.Sequential(
    nn.Linear(embed_dim * 2, embed_dim),
    nn.Sigmoid()
)
```

融合：

```python
gate = self.fusion_gate(torch.cat([aligned_plm, aligned_gcn], dim=-1))
f_combined = gate * aligned_plm + (1 - gate) * aligned_gcn
```

这比简单相加更合理。

#### 方案 C：返回 attention weights 用于解释

```python
return logits, {
    'pgca_attention': pgca_weights,
    'pmma_attention': pmma_weights
}
```

预测时可保存：

```text
compound_id
residue_index
token_or_atom_index
attention_score
```

---

## 9. 新增 top hits 快速筛选：PAINS / ADMET / SA

### 9.1 目标

对预测 top hits 做药物化学过滤。

### 9.2 输出

```text
results/top_hits_filtered.csv
results/top_hits_filter_report.json
```

### 9.3 关键代码

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
filter_catalog = FilterCatalog(params)


def calc_filters(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    alerts = filter_catalog.GetMatches(mol)
    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Crippen.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'RotB': Lipinski.NumRotatableBonds(mol),
        'QED': QED.qed(mol),
        'PAINS_BRENK_alert_count': len(alerts),
        'PAINS_BRENK_alerts': ';'.join([a.GetDescription() for a in alerts])
    }
```

### 9.4 推荐筛选条件

```text
calibrated_probability >= 0.8
prob_std <= 0.1
nearest_train_tanimoto < 0.8
leakage_warning == False
PAINS_BRENK_alert_count == 0
MW <= 550
LogP <= 5.5
TPSA <= 140
QED >= 0.35
```

---

## 10. 新增 docking 准备脚本

### 10.1 目标

不是直接完成 docking，而是快速生成 docking 输入。

### 10.2 输出

```text
docking/ligands/top_hits.sdf
docking/ligands/top_hits_3d/
docking/README_docking_protocol.md
```

### 10.3 关键代码

```python
from rdkit import Chem
from rdkit.Chem import AllChem


def make_3d_sdf(smiles, compound_id, out_sdf):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    ok = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if ok != 0:
        return False
    AllChem.MMFFOptimizeMolecule(mol)
    mol.SetProp('_Name', compound_id)
    writer = Chem.SDWriter(out_sdf)
    writer.write(mol)
    writer.close()
    return True
```

必须在 docking protocol 中记录：

```text
protein PDB ID
pocket center
grid box size
protonation method
redocking RMSD
docking software and version
```

---

## 11. 推荐优先级排序

### 第一优先级：一天内可完成

1. `external_independence_audit.py`
2. `analyze_predictions.py` 增加 `--output_json` 和 bootstrap CI
3. Morgan RF baseline
4. top hits PAINS / ADMET 初筛
5. 输出完整 JSON trace

### 第二优先级：2–3 天可完成

1. RDKit MolStandardize curation
2. duplicate / conflict label handling
3. Butina split
4. Morgan XGBoost / SVM baseline
5. ensemble uncertainty

### 第三优先级：一周内完成

1. ablation study CLI
2. PGCA / PMMA node-level attention 优化
3. attention weights 导出
4. docking input preparation
5. 自动生成论文表格和图

---

## 12. 建议最终命令行工作流

```bash
# 1. 数据整理
python curate_rorc_data.py \
  --input_csv data/raw/rorc_raw_all.csv \
  --output_csv data/curated/rorc_activity_master.csv \
  --report_json results/curation_report.json

# 2. 构建 scaffold / Butina split
python make_splits.py \
  --input_csv data/curated/rorc_activity_master.csv \
  --split_type butina \
  --output_dir data/splits/

# 3. 外部独立性审计
python external_independence_audit.py \
  --train_csv data/splits/train.csv \
  --external_csv data/splits/external_test.csv \
  --output_json results/external_independence_audit.json

# 4. 训练 baseline
python train_baselines.py \
  --train_csv data/splits/train.csv \
  --valid_csv data/splits/valid.csv \
  --test_csv data/splits/internal_test.csv \
  --output_dir results/baselines/

# 5. 训练主模型
python drug_lamp_ror_gamma.py \
  --mode train \
  --input_csv data/splits/train_valid_internal.csv \
  --smiles_col canonical_smiles \
  --label_col activity_label \
  --target_lbd_fasta data/target/RORC_LBD.fasta \
  --checkpoint models/druglamp_seed42.pt

# 6. 外部测试预测
python drug_lamp_ror_gamma.py \
  --mode predict \
  --input_csv data/splits/external_test.csv \
  --smiles_col canonical_smiles \
  --label_col activity_label \
  --checkpoint models/druglamp_seed42.pt \
  --output_csv results/external_predictions.csv

# 7. 评估 + CI
python analyze_predictions.py \
  --input_csv results/external_predictions.csv \
  --label_col known_label \
  --score_col calibrated_probability \
  --output_json results/external_eval_bootstrap.json \
  --bootstrap 1000

# 8. top hits 筛选
python filter_top_hits.py \
  --pred_csv results/screening_predictions.csv \
  --output_csv results/top_hits_filtered.csv
```

---

## 13. 最终 CNS 级别产出清单

完成上述代码模块后，应至少能产出：

```text
curation_report.json
split_audit.json
external_independence_audit.json
baseline_metrics.json
main_model_metrics.json
external_eval_bootstrap.json
ablation_results.csv
uncertainty_predictions.csv
top_hits_filtered.csv
docking_ready_ligands.sdf
```

这些文件可以直接支撑论文中的：

1. 数据质量图
2. split 流程图
3. baseline 对比表
4. external validation 表
5. AD 分层图
6. uncertainty calibration 图
7. top hits 筛选流程图
8. docking / ADMET 补充表

---

## 14. 最关键原则

高水平 QSAR 论文不是只看模型复杂度，而是看：

```text
数据是否干净
验证是否独立
baseline 是否公平
不确定性是否可解释
top hits 是否经药物化学过滤
结果是否可复现
```

当前脚本已经有不错的深度学习基础，但下一步应优先加强数据 curation、external independence audit、baseline 和 uncertainty，而不是继续堆叠更复杂的神经网络结构。
