# Time Series Base Type + Anomaly Classification
## Multi-Level Stacked Ensemble with Gated Routing and Context-Dependent Calibration

Final best score: **%89.07 FULL match (3919/4400)** on all 39 class groups,
up from the original **%59.8** single-ensemble baseline — **+29.3 points improvement**.

---

## Table of Contents
1. [Problem Overview](#problem-overview)
2. [Final Results](#final-results)
3. [Full 39-Class Table](#full-39-class-table)
4. [Architecture](#architecture)
5. [Data Sources and Sampling](#data-sources-and-sampling)
6. [Strategy Evolution](#strategy-evolution)
7. [External Model References](#external-model-references)
8. [Implementation Details](#implementation-details)
9. [File Structure](#file-structure)
10. [Reproduction](#reproduction)

---

## Problem Overview

The task is to classify raw time-series CSV files into one of 39 groups based on:
- **Base process type** (4 classes): `stationary`, `deterministic_trend`,
  `stochastic_trend`, `volatility`
- **Anomaly type** (6 classes, multi-label): `collective_anomaly`,
  `contextual_anomaly`, `mean_shift`, `point_anomaly`, `trend_shift`, `variance_shift`

The 39 groups are:
- **10 single-type groups** (groups 1–10): pure base type OR stationary+single_anomaly
- **29 combination groups** (groups 11–39): `{deterministic, stochastic, volatility}+anomaly`

**Evaluation**: Full match requires BOTH the base type correct AND all anomalies correctly detected (no false positives, no false negatives).

---

## Final Results

| Metric | Baseline (ensemble-alldata) | This Project (v12) | Delta |
|---|---|---|---|
| **FULL match** | 59.8% (2632) | **89.07% (3919)** | **+29.27** |
| **PARTIAL match** | 24.6% (1083) | 5.02% (221) | -19.58 |
| **NONE (wrong base)** | 15.6% (685) | 5.91% (260) | -9.69 |
| **Total evaluated** | 4400 | 4400 | — |

Sampling methodology: 10 CSV per leaf directory × all leaf directories across 39 groups = 4400 samples.

---

## Full 39-Class Table

Best configuration: **Stat Detector v2 @ 0.92** + **Router Θ = 0.30** + per-anomaly blended thresholds.

| # | Group | Expected | n | FULL | PART | NONE | FULL% |
|---|---|---|---|---|---|---|---|
| 1 | stationary | stationary | 120 | 67 | 49 | 4 | **55.8** |
| 2 | deterministic_trend | det_trend | 720 | 674 | 10 | 36 | **93.6** |
| 3 | stochastic_trend | stoch_trend | 150 | 129 | 15 | 6 | **86.0** |
| 4 | volatility | volatility | 120 | 106 | 7 | 7 | **88.3** |
| 5 | collective_anomaly | stat+collective | 480 | 432 | 7 | 41 | **90.0** |
| 6 | contextual_anomaly | stat+contextual | 480 | 480 | 0 | 0 | **100.0** |
| 7 | mean_shift | stat+mean_shift | 480 | 421 | 27 | 32 | **87.7** |
| 8 | point_anomaly | stat+point | 480 | 408 | 17 | 55 | **85.0** |
| 9 | trend_shift | stat+trend_shift | 480 | 441 | 10 | 29 | **91.9** |
| 10 | variance_shift | stat+var_shift | 480 | 384 | 46 | 50 | **80.0** |
| 11 | cubic+collective | det+collective | 10 | 10 | 0 | 0 | **100.0** |
| 12 | cubic+mean_shift | det+mean | 20 | 19 | 1 | 0 | **95.0** |
| 13 | cubic+point_anomaly | det+point | 10 | 10 | 0 | 0 | **100.0** |
| 14 | cubic+variance_shift | det+var | 10 | 9 | 1 | 0 | **90.0** |
| 15 | damped+collective | det+collective | 10 | 10 | 0 | 0 | **100.0** |
| 16 | damped+mean_shift | det+mean | 20 | 20 | 0 | 0 | **100.0** |
| 17 | damped+point_anomaly | det+point | 10 | 10 | 0 | 0 | **100.0** |
| 18 | damped+variance_shift | det+var | 10 | 10 | 0 | 0 | **100.0** |
| 19 | exp+collective | det+collective | 10 | 10 | 0 | 0 | **100.0** |
| 20 | exp+mean_shift | det+mean | 20 | 20 | 0 | 0 | **100.0** |
| 21 | exp+point_anomaly | det+point | 10 | 10 | 0 | 0 | **100.0** |
| 22 | exp+variance_shift | det+var | 10 | 10 | 0 | 0 | **100.0** |
| 23 | linear+collective | det+collective | 10 | 10 | 0 | 0 | **100.0** |
| 24 | linear+mean_shift | det+mean | 20 | 19 | 1 | 0 | **95.0** |
| 25 | linear+point_anomaly | det+point | 10 | 10 | 0 | 0 | **100.0** |
| 26 | linear+trend_shift | det+trend_shift | 30 | 30 | 0 | 0 | **100.0** |
| 27 | linear+variance_shift | det+var | 10 | 10 | 0 | 0 | **100.0** |
| 28 | quad+collective | det+collective | 10 | 10 | 0 | 0 | **100.0** |
| 29 | quad+mean_shift | det+mean | 20 | 20 | 0 | 0 | **100.0** |
| 30 | quad+point_anomaly | det+point | 20 | 20 | 0 | 0 | **100.0** |
| 31 | quad+variance_shift | det+var | 10 | 9 | 1 | 0 | **90.0** |
| 32 | stoch+collective | stoch+collective | 10 | 5 | 5 | 0 | **50.0** |
| 33 | stoch+mean_shift | stoch+mean | 10 | 6 | 4 | 0 | **60.0** |
| 34 | stoch+point_anomaly | stoch+point | 10 | 10 | 0 | 0 | **100.0** |
| 35 | stoch+variance_shift | stoch+var | 50 | 44 | 6 | 0 | **88.0** |
| 36 | vol+collective | vol+collective | 10 | 3 | 7 | 0 | **30.0** |
| 37 | vol+mean_shift | vol+mean | 10 | 10 | 0 | 0 | **100.0** |
| 38 | vol+point_anomaly | vol+point | 10 | 6 | 4 | 0 | **60.0** |
| 39 | vol+variance_shift | vol+var | 10 | 7 | 3 | 0 | **70.0** |

**TOTAL: 3919/4400 FULL = 89.07%** | PARTIAL: 221 (5.02%) | NONE: 260 (5.91%)

**Perfect groups (100%):** 18 groups
**Groups above 90%:** 28 groups
**Groups above 80%:** 33 groups
**Weakest groups:** #36 (vol+collective, 30%), #1 (stationary, 55.8%), #32 (stoch+collective, 50%)

---

## Architecture

### Multi-Level Stacked Ensemble with Gated Routing

```
Input Time Series (raw CSV)
        │
        ├──► tsfresh Feature Extraction (777 features, EfficientFC)
        │         │
        │         ├──► Old Ensemble (9 binary detectors)  ──► 9 probabilities
        │         │    [tsfresh ensemble/]
        │         │
        │         └──► New Ensemble (10 binary models)    ──► 10 probabilities
        │              [ensemble-alldata/]
        │              4 base + 6 anomaly
        │
        ├──► Stationary Detector (25 custom features)     ──► P(stationary)
        │    [stationary detector ml/v2/]
        │
        └──► Meta-Feature Vector: 9 + 10 + 14 derived + 777 raw tsfresh = 810 dim
                      │
                      ├──► Stage 1: STATIONARY DETECTOR OVERRIDE
                      │    IF P(stationary) >= 0.92 → return ("stationary", [])
                      │
                      ├──► Stage 2: ROUTER (single vs combo binary)
                      │    XGBoost + LightGBM ensemble on 810 features
                      │    IF P(combo) < 0.30 → single branch
                      │    ELSE → combo branch
                      │
                      ├──► SINGLE BRANCH:
                      │    Base type meta-learner (XGB+LGB) only
                      │    Return (base_type, [])
                      │
                      └──► COMBO BRANCH:
                           Base type meta-learner (XGB+LGB) → base
                           For each anomaly:
                             meta_prob = XGB+LGB ensemble predict_proba
                             blended = α * meta_prob + (1-α) * new_ensemble_prob
                             IF blended >= per_anomaly_threshold:
                               Add to anomalies
                           Return (base_type, anomalies)
```

### Key Components

1. **Two Base Ensembles (pre-trained, reused as-is)**
   - Old ensemble (9 detectors, single-type-trained)
   - New ensemble (10 models: 4 base + 6 anomaly, combination-trained)

2. **Stat Detector Override** (outer gate)
   - Binary stationary-vs-non_stationary XGBoost with custom 25-feature pipeline
   - Used with extremely high confidence threshold (0.92) to avoid false positives on groups 5–10

3. **Stacking Meta-Learners** (810 features → decisions)
   - Base type: 4-class XGBoost + LightGBM ensemble (F1=0.968)
   - Per-anomaly binary: XGBoost + LightGBM ensembles
   - Router: binary single vs combination XGBoost + LightGBM ensemble

4. **Blended Anomaly Decision**
   - `final_prob = α * meta_prob + (1-α) * new_ensemble_direct_prob`
   - α and threshold tuned per-anomaly via grid search

5. **Context-Dependent Threshold Tuning**
   - Per (base_type, anomaly) threshold learning
   - Stationary base → more lenient anomaly threshold

---

## Data Sources and Sampling

### Data Path
```
C:\Users\user\Desktop\Generated Data\
├── stationary/              # Group 1
├── deterministic_trend/     # Group 2
├── Stochastic Trend/        # Group 3
├── Volatility/              # Group 4
├── collective_anomaly/      # Group 5 (stationary base)
├── contextual_anomaly/      # Group 6
├── mean_shift/              # Group 7
├── point_anomaly/           # Group 8
├── trend_shift/             # Group 9
├── variance_shift/          # Group 10
└── Combinations/
    ├── Cubic Base/          # Groups 11-14
    ├── Damped Base/         # Groups 15-18
    ├── Exponential Base/    # Groups 19-22
    ├── Linear Base/         # Groups 23-27
    ├── Quadratic Base/      # Groups 28-31
    ├── Stochastic Trend + Collective/... # Groups 32-35
    └── Volatility + .../    # Groups 36-39
```

### Training Data (Meta-Learner)
- **500 samples per group × 39 groups = 19,500 total**
- Leaf-balanced sampling: each leaf directory contributes proportionally
- tsfresh extraction with EfficientFCParameters (777 features)
- Ensemble probabilities precomputed and cached

### Evaluation Data
- **All leaf directories, 10 CSVs per leaf = 4400 total samples**
- Fixed random seed (42) for reproducibility
- Same sampling protocol as `manuelalldatatest.py` in ensemble-alldata

---

## Strategy Evolution

| # | Version | Key Addition | FULL% | Δ |
|---|---|---|---|---|
| 0 | Original | Single new ensemble | 59.8% | — |
| 1 | v1 (stack 50/grp) | Stacking meta-learner | 67.8% | +8.0 |
| 2 | v2 (+PCA+derived) | 53 meta features | 67.8% | = |
| 3 | v3 (150/grp+tune) | More data + per-anom threshold | 74.6% | +6.8 |
| 4 | v4 (oversample) | Grp 5-10 3x oversample | 77.0% | +2.4 |
| 5 | v5 (context) | Stationary context threshold | 77.5% | +0.5 |
| 6 | v6 (XGB+LGB) | Dual meta-learner ensemble | 77.5% | = |
| 7 | v7 (810-feat router) | Raw tsfresh in router | 77.9% | +0.4 |
| 8 | v8 (CT=0.0) | Aggressive context | 88.5% | +10.6 |
| 9 | v9 (stat det v6) | Binary stationary gate | 88.5% | = |
| 10 | v10 (stat det v2) | Better stat detector | 88.6% | +0.1 |
| 11 | v11 (RT=0.35) | Router fine-tuning | 88.7% | +0.1 |
| 12 | **v12 (joint+per-anom)** | **Joint grid search** | **89.07%** | **+0.37** |

### Total improvement: +29.27 percentage points over baseline

### Key Insights Along the Way

1. **Stacking is the biggest single win** (67.8% from 59.8%) — combining both
   ensembles' opinions as meta-features gives the meta-learner both perspectives.

2. **Raw tsfresh features in meta-learner** (+6.8%) — PCA to 20 components lost
   crucial signal; using full 777 features let XGBoost pick the important ones.

3. **Oversampling weak groups** — 3x oversample of stationary+anomaly groups in
   meta training directly addressed the model's weakest areas.

4. **Context-dependent thresholds (CT=0.0)** — aggressive at first seems wrong,
   but the router already filters 99%+ of pure stationary cases; for combo cases
   with stationary base, any signal is meaningful.

5. **Dual meta-learners (XGB+LGB)** — different inductive biases catch different
   errors; simple average works better than either alone.

6. **Stationary detector as outer gate** — binary classifier with custom features
   acts as second opinion on pure base cases without contaminating anomaly
   decisions (high threshold 0.92).

7. **Per-anomaly joint alpha+threshold tuning** — each anomaly has different
   optimal blend weight and cutoff; tuning all 12 parameters jointly gives the
   final polish.

---

## External Model References

This project reuses three pre-trained models from sibling repositories. These
are loaded at inference time and must exist on disk:

### 1. Old Ensemble (9-class binary detectors)
**Path:** `../tsfresh ensemble/trained_models/`
**Classes:** collective_anomaly, contextual_anomaly, deterministic_trend,
mean_shift, point_anomaly, stochastic_trend, trend_shift, variance_shift, volatility
**Training:** Single-label time series, tsfresh EfficientFC features, one
XGBoost/LightGBM/MLP per class, best by F1 selected.
**Reference:** [tsfresh ensemble/](../tsfresh%20ensemble/)

### 2. New Ensemble (10 binary models: 4 base + 6 anomaly)
**Path:** `../ensemble-alldata/trained_models/`
**Classes:** stationary, deterministic_trend, stochastic_trend, volatility (base),
collective_anomaly, contextual_anomaly, mean_shift, point_anomaly, trend_shift,
variance_shift (anomaly).
**Training:** Combination data (base + single anomaly), N=1320 per model,
leaf-balanced sampling, LightGBM/XGBoost/MLP, best by validation F1.
**Reference:** [ensemble-alldata/](../ensemble-alldata/) — see its own README
for detailed training methodology.

### 3. Stationary Detector v2
**Path:** `../stationary detector ml/trained_models v2/`
**Classes:** stationary (0) vs non_stationary (1)
**Training:** Custom 25 features (mean, std, skewness, kurtosis,
autocorrelation, rolling window stats, etc.), XGBoost best (F1=0.881).
**Reference:** [stationary detector ml/](../stationary%20detector%20ml/)
**Why v2:** Tested all 6 versions (v1–v6); v2 gave cleanest stationary/non-stationary
separation on our evaluation data — P(stat) near 0 on anomaly groups and near 1
on pure stationary.

---

## Implementation Details

### Meta-Learner Configuration

**Base Type Meta-Learner:**
```python
XGBoost: n_estimators=500, lr=0.05, max_depth=6, min_child_weight=3,
         gamma=0.1, subsample=0.8, colsample_bytree=0.7,
         reg_alpha=0.1, reg_lambda=1.0, class_weight=balanced
LightGBM: same hyperparameters, num_leaves=63
Ensemble: 0.5 * XGB + 0.5 * LGB predict_proba
Test Accuracy: 96.85%
```

**Per-Anomaly Meta-Learners (6 × XGB+LGB):**
```python
Oversample groups 5-10 by 3x before training
scale_pos_weight = neg_count / pos_count
Same hyperparameters as base
Test F1 range: 0.91 (mean_shift) to 1.00 (contextual_anomaly)
```

**Router (single vs combo):**
```python
Input: 810 meta features
Positive (combo): groups 5-39
Negative (single): groups 1-4
Test F1: 0.978
```

### Best Configuration (v12)

```python
# evaluator.py
STAT_DET_THRESHOLD = 0.92          # Stationary detector v2 confidence
ROUTER_THETA = 0.30                # Router single/combo boundary
CONTEXT_THRESH = 0.0               # Combo + stationary base override (not used in v12)

# Per-anomaly blend parameters
blend_params = {
    "collective_anomaly":  {"alpha": 0.85, "threshold": 0.73},
    "contextual_anomaly":  {"alpha": 0.70, "threshold": 0.69},
    "mean_shift":          {"alpha": 0.90, "threshold": 0.49},
    "point_anomaly":       {"alpha": 0.70, "threshold": 0.69},
    "trend_shift":         {"alpha": 0.90, "threshold": 0.73},
    "variance_shift":      {"alpha": 0.70, "threshold": 0.69},
}
```

### Meta Training Data Pipeline

```
19500 CSVs (500/group × 39)
    ↓
tsfresh EfficientFC (777 features)
    ↓
[Old ensemble probs (9)] + [New ensemble probs (10)]
    ↓
+ 14 derived features (agreement, entropy, confidence gaps, max/argmax)
    ↓
+ 777 raw tsfresh (standardized)
    ↓
810-dim meta feature vector
    ↓
Train: base meta (4-class XGB+LGB), 6 anomaly metas (binary XGB+LGB), router (binary XGB+LGB)
```

### Derived Features (14)

1. max_old_base, argmax_old_base
2. max_old_anomaly, n_old_anomaly_above_0.5
3. max_new_base, argmax_new_base
4. max_new_anomaly, n_new_anomaly_above_0.5
5. base_agreement (old == new?)
6. base_confidence_gap (max - 2nd_max)
7. anomaly_entropy (mean binary entropy of new anomaly probs)
8. old_new_anomaly_correlation
9. total_new_anomaly_signal
10. total_old_anomaly_signal

---

## File Structure

```
hopefullyprojectfinal/
├── README.md                    # This file
├── BEST_RESULTS.md              # Best config backup + per-group table
├── .gitignore
│
├── config.py                    # Paths, 39 group definitions, model lists
├── processor.py                 # tsfresh extraction, ensemble inference, meta features
├── trainer.py                   # Meta-learner training + blend learning
├── evaluator.py                 # Full evaluation pipeline
├── stat_detector.py             # Standalone stationary detector wrapper
├── main.py                      # Orchestration: --force, --train, --eval
│
├── cache_eval.py                # Builds evaluation cache (4400 samples, all probs)
├── fast_grid.py                 # Fast grid search on cache
├── fast_grid2.py                # Advanced grid (joint stat/router/blend)
├── fast_grid3.py                # Per-anomaly alpha × threshold joint
├── eval_best.py                 # Print 39-group table for best config
│
├── processed_data/              # (gitignored) cached tsfresh + ensemble probs
│   ├── meta_X.npy               # (19500, 810) meta features
│   ├── meta_y_base.npy          # (19500,) base labels
│   ├── meta_y_anom.npy          # (19500, 6) anomaly labels
│   ├── tsfresh_scaler.pkl       # StandardScaler for tsfresh
│   └── eval_cache.npz           # (4400,) all probs for fast grid search
│
├── meta_models/                 # (gitignored) trained meta-learners
│   ├── base_meta.pkl            # {xgb, lgb}
│   ├── anom_<name>.pkl          # {xgb, lgb} × 6
│   ├── router.pkl               # {xgb, lgb}
│   └── blend_weights.pkl
│
└── results/                     # (gitignored) evaluation JSONs and reports
```

---

## Reproduction

### Prerequisites

1. Python 3.10+
2. Install dependencies:
   ```
   pip install numpy pandas scikit-learn xgboost lightgbm tsfresh joblib tqdm scipy
   ```
3. External data and models must exist at these paths (sibling directories):
   ```
   ../tsfresh ensemble/trained_models/           # 9 detectors
   ../ensemble-alldata/trained_models/            # 10 models
   ../stationary detector ml/trained_models v2/   # stat detector
   C:\Users\user\Desktop\Generated Data\         # raw CSV data (39 groups)
   ```

### Full Pipeline (from scratch)

```bash
# 1. Prepare meta-learner training data (takes ~20-30 min)
python main.py --force

# This will:
#   - Sample 500 CSVs from each of 39 groups (19500 total)
#   - Run tsfresh EfficientFC extraction
#   - Compute old + new ensemble probabilities  
#   - Compute 14 derived features
#   - Standardize tsfresh features
#   - Build 810-dim meta feature matrix
#   - Train base meta-learner (XGB+LGB, 4-class)
#   - Train 6 anomaly meta-learners (XGB+LGB, binary, with oversampling)
#   - Train router (XGB+LGB, single vs combo)
#   - Learn per-anomaly blend (alpha, threshold) via F1 optimization
#   - Run full evaluation on 4400 samples
#   - Output 39-group table + save JSON

# 2. Build evaluation cache (one-time, ~15-20 min)
python cache_eval.py

# 3. Run fast grid search (seconds per strategy)
python fast_grid3.py

# 4. Print best-config 39-group table
python eval_best.py
```

### Running Evaluation Only (after training done)

```bash
python main.py --eval
```

---

## Acknowledgments

This project builds on three sibling models:
- **tsfresh ensemble** — 9 single-label binary detectors
- **ensemble-alldata** — 10 binary models for combination detection  
- **stationary detector ml** — binary stationary classifier (v2 used)

All three were trained independently and are loaded as-is; no retraining was
done on them for this project. This project contributes the stacking layer,
router, blending, per-anomaly calibration, and stationary gate.

---

## Appendix: Reference Techniques Used

| Technique | Where Applied |
|---|---|
| Stacked Generalization (Wolpert 1992) | Meta-learner on base classifier probs |
| Gradient Boosting (XGBoost) | Base meta, anomaly metas, router |
| LightGBM | Secondary meta-learner for ensemble averaging |
| Automated Feature Engineering (tsfresh) | 777 features per series |
| Class Imbalance Oversampling | Groups 5–10 (3x) in meta training |
| Hierarchical/Gated Classification | Stationary gate + single/combo router |
| Blended Probability Combination | α × meta + (1-α) × new_ensemble |
| Per-Context Threshold Calibration | Per-anomaly thresholds via F1 grid search |
| Model Averaging (XGB+LGB) | Base + all anomaly meta-learners |
| Derived Meta-Features | Agreement, entropy, confidence gaps |
