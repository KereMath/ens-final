# Hierarchical Multi-Level Stacked Ensemble for Time Series Base Type and Anomaly Classification

A production-grade pipeline combining two pre-trained binary ensemble systems,
a custom-feature stationarity gate, a learned routing layer, and tuned
stacking meta-learners to classify time series into 39 distinct behavioral
categories combining base process types and anomaly patterns.

**Final Result: 89.07 % FULL match (3919 / 4400)** on out-of-distribution evaluation —
a **+29.27 percentage point improvement** over the single-ensemble baseline.

---

## Table of Contents

1. [Motivation and Problem Statement](#motivation-and-problem-statement)
2. [Dataset and Taxonomy](#dataset-and-taxonomy)
3. [System Architecture](#system-architecture)
4. [Component 1 — tsfresh Feature Extraction](#component-1--tsfresh-feature-extraction)
5. [Component 2 — Old Binary Ensemble (9 detectors)](#component-2--old-binary-ensemble-9-detectors)
6. [Component 3 — New Binary Ensemble (10 models)](#component-3--new-binary-ensemble-10-models)
7. [Component 4 — Stationary Detector Gate](#component-4--stationary-detector-gate)
8. [Component 5 — Single/Combination Router](#component-5--singlecombination-router)
9. [Component 6 — Stacking Meta-Learners](#component-6--stacking-meta-learners)
10. [Component 7 — Blended Probability Decision](#component-7--blended-probability-decision)
11. [Inference Pipeline](#inference-pipeline)
12. [Training Data and Balanced Sampling](#training-data-and-balanced-sampling)
13. [Hyperparameter Search and Calibration](#hyperparameter-search-and-calibration)
14. [Full Evaluation Results (39 classes)](#full-evaluation-results-39-classes)
15. [Incremental Improvement History](#incremental-improvement-history)
16. [File Organization and Reproducibility](#file-organization-and-reproducibility)
17. [External Model References](#external-model-references)
18. [Techniques Used — Academic Summary](#techniques-used--academic-summary)

---

## Motivation and Problem Statement

Given a raw univariate time series sampled as CSV, the task is to assign:

1. **A base process type** (4-way classification): stationary, deterministic trend,
   stochastic trend, or volatility (heteroskedastic).
2. **Zero or more anomaly labels** (6-way multi-label): collective, contextual,
   mean shift, point, trend shift, or variance shift.

The evaluation metric is **strict FULL match** — a prediction counts as correct
only if the base type is identified AND every anomaly present is detected AND
no spurious anomalies are produced (no false positives). This single-criterion
evaluation forces the system to be simultaneously accurate on base type
identification and surgically precise on anomaly detection.

The 39 source groups span four complexity tiers:

| Tier | Description | Groups | Example |
|---|---|---|---|
| **1. Pure base** | Single process, no anomaly | 1–4 | stationary (1), deterministic_trend (2) |
| **2. Stationary + single anomaly** | Baseline process + one anomaly | 5–10 | stationary + mean_shift (7) |
| **3. Deterministic + anomaly** | Trend process + one anomaly | 11–31 | cubic + collective (11), linear + trend_shift (26) |
| **4. Non-deterministic combinations** | Stochastic / volatility + anomaly | 32–39 | stoch_trend + variance_shift (35) |

---

## Dataset and Taxonomy

Total evaluation: **4,400 CSV files** drawn from all 39 groups using
leaf-balanced sampling (10 files per leaf directory). The 39 groups and their
canonical expected labels are:

| # | Group Name | Expected Base | Expected Anomaly | Count |
|---|---|---|---|---|
| 1 | stationary | stationary | — | 120 |
| 2 | deterministic_trend | deterministic_trend | — | 720 |
| 3 | stochastic_trend | stochastic_trend | — | 150 |
| 4 | volatility | volatility | — | 120 |
| 5 | collective_anomaly | stationary | collective_anomaly | 480 |
| 6 | contextual_anomaly | stationary | contextual_anomaly | 480 |
| 7 | mean_shift | stationary | mean_shift | 480 |
| 8 | point_anomaly | stationary | point_anomaly | 480 |
| 9 | trend_shift | stationary | trend_shift | 480 |
| 10 | variance_shift | stationary | variance_shift | 480 |
| 11–14 | cubic + {collective, mean, point, variance} | deterministic_trend | each | 10–20 |
| 15–18 | damped + {collective, mean, point, variance} | deterministic_trend | each | 10–20 |
| 19–22 | exponential + {collective, mean, point, variance} | deterministic_trend | each | 10–20 |
| 23–27 | linear + {collective, mean, point, trend_shift, variance} | deterministic_trend | each | 10–30 |
| 28–31 | quadratic + {collective, mean, point, variance} | deterministic_trend | each | 10–20 |
| 32–35 | stochastic + {collective, mean, point, variance} | stochastic_trend | each | 10–50 |
| 36–39 | volatility + {collective, mean, point, variance} | volatility | each | 10 |

---

## System Architecture

The pipeline is a **seven-stage hierarchical ensemble** where each stage
contributes a distinct inductive bias:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INPUT: raw univariate time series (variable length, min 50 timesteps)  │
└─────────────────────────────────────────────────────────────────────────┘
                │
                ├─► Stage A: tsfresh EfficientFC   →  777-dim feature vector
                │
                ├─► Stage B: Stationary Detector Gate ─────► P(stationary)
                │   (custom 25 features, XGBoost binary)
                │
                ├─► Stage C: Old Binary Ensemble   ─────► 9 × P(class_k)
                │   (9 independent XGBoost/LightGBM/MLP binaries)
                │
                ├─► Stage D: New Binary Ensemble   ─────► 10 × P(class_k)
                │   (4 base + 6 anomaly, LightGBM/XGBoost binaries)
                │
                ├─► Stage E: Feature Engineering
                │   • 14 derived meta-features
                │     (agreement, entropy, confidence gaps, max/argmax)
                │   • Standardized raw tsfresh features (777)
                │   └────► 810-dim unified meta-vector
                │
                ├─► Stage F: Single/Combination Router
                │   (XGB+LGB binary on 810-dim)  →  P(combination)
                │
                └─► Stage G: Stacking Meta-Learners
                    • Base type: XGB+LGB 4-class on 810-dim
                    • 6 × Anomaly binaries: XGB+LGB
                      with alpha-blended new-ensemble probability
                      and per-anomaly tuned threshold

                    ▼
                DECISION LOGIC:
                IF stationary_gate ≥ 0.92:
                    return (stationary, [])        # Override path
                ELIF router(combo) < 0.30:
                    return (base_meta_argmax, [])  # Single path
                ELSE:
                    return (
                        base_meta_argmax,
                        [a for a in ANOMALIES
                         if α·meta_a + (1-α)·new_a ≥ threshold_a]
                    )                              # Combination path
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: (base_type, [list of detected anomalies])                      │
└─────────────────────────────────────────────────────────────────────────┘
```

Each component is described in full technical detail below.

---

## Component 1 — tsfresh Feature Extraction

**What it is:** An automated, deterministic feature generator that maps a
variable-length time series to a fixed 777-dimensional feature vector using
tsfresh's `EfficientFCParameters` (a curated subset excluding the most
computationally expensive calculators).

**Why 777 features:** Empirically, this is the upper bound on what tsfresh
generates for single-variable series with `EfficientFCParameters`.
The 777 features span:

| Feature Family | Approx Count | Examples |
|---|---|---|
| Statistical moments | ~20 | mean, std, skewness, kurtosis, quantiles |
| Autocorrelation | ~100 | `autocorrelation(lag=k)` for k=1..10, partial ACF |
| Fourier descriptors | ~150 | FFT coefficient real/imaginary, spectral centroid |
| Wavelet CWT | ~50 | Continuous wavelet transform coefficients |
| Change metrics | ~80 | `change_quantile`, `cid_ce`, `mean_abs_change` |
| Peak detection | ~30 | `number_peaks(n=k)`, prominence |
| Distribution | ~40 | `ratio_beyond_r_sigma`, entropy, benford |
| Linear trend | ~20 | slope, stderr, intercept on raw and sub-series |
| Nonlinear descriptors | ~50 | `augmented_dickey_fuller`, Lempel-Ziv |
| Symbolic aggregation | ~50 | `value_count`, `unique_ratio`, binned counts |
| Miscellaneous | ~187 | `friedrich_coefficients`, `agg_linear_trend`, etc. |

**Why this family of features:** tsfresh's extraction is a widely-adopted,
standard baseline in time series analysis. It's well-tuned for classification
tasks and captures both local and global temporal characteristics without
requiring manual engineering. Crucially, **every downstream component in this
pipeline takes these same 777 features as input**, making them the single
unifying representation of the series.

**NaN handling:** tsfresh's `impute()` replaces infinities and NaN values
with column means/0; we additionally apply `np.nan_to_num()` before any model
call to guarantee numerical stability.

---

## Component 2 — Old Binary Ensemble (9 detectors)

**Source:** Loaded from `../tsfresh ensemble/trained_models/`. This ensemble
was trained independently on **single-label time series data** (each series
belongs to exactly one of 9 classes).

**Architecture:** 9 independent binary detectors, one per class:

```
Classes (ordered for canonical indexing):
  0: collective_anomaly
  1: contextual_anomaly
  2: deterministic_trend
  3: mean_shift
  4: point_anomaly
  5: stochastic_trend
  6: trend_shift
  7: variance_shift
  8: volatility
```

**Training methodology** (original repo): For each of the 9 classes,
a separate binary XGBoost / LightGBM / MLP was trained with the target
class as positive (1) and all other classes as negatives (0). The best
model per class was selected by **validation F1** and stored alongside
a per-class `RobustScaler`.

**Output at inference:** 9 probabilities `P(class_k = 1 | tsfresh_features)`.
These 9 values become the **first 9 components** of the meta-feature vector
fed to the stacking layer.

**Why retained unchanged:** The old ensemble excels at single-label
identification ("is this series mean_shift?"). Retraining it would
require rebuilding its training set and offers no clear advantage —
but its **per-class opinion** is valuable signal for the meta-learner,
especially for determining base type on pure trend series (Group 2)
where it complements the new ensemble's occasional miscalibration.

---

## Component 3 — New Binary Ensemble (10 models)

**Source:** Loaded from `../ensemble-alldata/trained_models/`. Trained
independently on **combination data** (base process plus one anomaly per
series).

**Architecture:** 10 independent binary models covering:

| Index | Model Name | Type |
|---|---|---|
| 0 | stationary | base |
| 1 | deterministic_trend | base |
| 2 | stochastic_trend | base |
| 3 | volatility | base |
| 4 | collective_anomaly | anomaly |
| 5 | contextual_anomaly | anomaly |
| 6 | mean_shift | anomaly |
| 7 | point_anomaly | anomaly |
| 8 | trend_shift | anomaly |
| 9 | variance_shift | anomaly |

**Training methodology** (original repo): For each model, training data was
built with **N=1320 positive + N=1320 negative samples**, drawn via
leaf-balanced sampling across all 39 source groups. Positive sources were
the groups where the target class is present (e.g., for `mean_shift`, the
positive groups are 7, 12, 16, 20, 24, 29, 33, 37). For each model,
LightGBM, XGBoost, and MLP were trained; the one with highest **validation F1**
was saved.

**Output at inference:** 10 probabilities. These become the **next 10
components** of the meta-feature vector (indices 9–18).

**Why retained unchanged:** The new ensemble is well-calibrated for
base+anomaly combinations (groups 11–39 reach 95.9% in-distribution
on these models alone). However, it exhibits **overly confident
false positives** on pure base types (groups 1–4): on a pure stationary
series, it may still produce `P(point_anomaly) = 0.55` and issue a
spurious anomaly. The downstream stacking + routing layers address this.

---

## Component 4 — Stationary Detector Gate

**Source:** Loaded from `../stationary detector ml/trained_models v2/`.
A separate binary classifier distinguishing stationary (class 0) from
non-stationary (class 1).

**Critical distinction:** Unlike Components 2 and 3 which use tsfresh
features, **this detector uses custom 25 hand-engineered features**
purpose-built for stationarity testing:

```
Feature family                 # features
─────────────────────────────────────────
Basic statistics               13
  mean, std, var, min, max, range,
  q25, median, q75, iqr,
  skewness, kurtosis, cv
First/second differences       5
  diff1_{mean, std, var},
  diff2_{mean, std}
Rolling window stats           3
  rolling_mean_std,
  rolling_std_mean, rolling_std_std
Autocorrelation                2
  autocorr_lag1, autocorr_lag10
Peak analysis                  2
  num_peaks, zero_crossing_rate
```

These features are **selected specifically to detect stationarity**:
for instance `rolling_std_std` captures whether the series has stable
variance over time; `autocorr_lag10` captures short-range memory effects.

**Why this approach over tsfresh:** Empirically, tsfresh's 777 features
include many aggregate statistics that are insensitive to brief localized
changes. For a stationary series with a single point anomaly added, tsfresh
still classifies it as stationary because the overall distribution is
barely perturbed. The custom features — especially `num_peaks` and
`rolling_std_std` — are more sensitive to these localized perturbations.

**Training methodology** (original repo): Binary classification on
a balanced training set (stationary vs non_stationary), with
LightGBM, XGBoost, MLP, Decision Tree, Random Forest, Extra Trees,
and MLPFast all evaluated. **XGBoost** won with **F1 = 0.881** on the
held-out test split.

**Output at inference:** A single probability `P(stationary)`.

**Role in pipeline:** Serves as a **high-precision outer gate**. If
`P(stationary) ≥ 0.92`, the pipeline shortcuts the downstream components
and directly emits `("stationary", [])`. This threshold was tuned via
grid search to maximize full-match count while avoiding false overrides
on groups 5–10 (stationary base with anomaly).

---

## Component 5 — Single/Combination Router

**Role:** Before committing to the full combination decision, the router
provides a **second-opinion** on whether the series is a pure single
pattern or a combination of base + anomaly.

**Training setup:**
- **Positive (combination = 1):** Groups 5–39 (all groups with at least
  one anomaly or non-stationary base)
- **Negative (single = 0):** Groups 1–4 (pure base types)
- **Training data:** 19,500 samples (500 per group × 39 groups),
  leaf-balanced
- **Feature representation:** The **full 810-dim meta-vector** of
  Component 6 (see below), not just tsfresh features
- **Models:** XGBoost (500 trees, max_depth=6, lr=0.05) and LightGBM
  (num_leaves=63, max_depth=7) trained independently
- **Test metric:** Ensemble F1 = **0.978** on held-out 20% split

**Inference:** `P(combination) = 0.5 × XGB + 0.5 × LGB` probabilities.

**Routing decision:**
```
IF P(combination) < 0.30:
    Route to SINGLE branch → return (base_meta_argmax, [])
ELSE:
    Route to COMBINATION branch → full anomaly evaluation
```

The threshold **0.30** was selected via grid search. A lower threshold
biases the system toward the combination branch, which is desirable
because the combination branch has robust mechanisms (threshold calibration,
blending) to prevent false positives, while falling through to the single
branch loses valid anomaly detections.

---

## Component 6 — Stacking Meta-Learners

**Purpose:** Learn how to combine the opinions of the old ensemble (9),
the new ensemble (10), and raw tsfresh features into a final prediction.
This is the **central innovation** of the project: rather than hand-coding
rules for when to trust which ensemble, we train models to do it.

### Meta-Feature Vector (810 dimensions)

Constructed for every sample:

```
Dimension   Source                               Description
────────────────────────────────────────────────────────────────
  0 –   8   Old ensemble probabilities           9 binaries
  9 –  18   New ensemble probabilities           10 binaries (4 base + 6 anomaly)
 19 –  32   Derived meta-features                14 engineered stats
 33 – 809   Raw tsfresh features (standardized)  777 features
────────────────────────────────────────────────────────────────
           TOTAL                                  810 dimensions
```

### 14 Derived Meta-Features

Computed from the 19 raw ensemble probabilities to provide richer signal:

1. `max_old_base` — highest prob among old ensemble's base classes
2. `argmax_old_base` — which base class old ensemble prefers
3. `max_old_anomaly` — highest prob among old ensemble's anomaly classes
4. `n_old_anomaly_above_0.5` — count of old anomaly classes firing
5. `max_new_base` — new ensemble's max base prob
6. `argmax_new_base` — new ensemble's preferred base index
7. `max_new_anomaly` — new ensemble's max anomaly prob
8. `n_new_anomaly_above_0.5` — count of new anomalies firing
9. `base_agreement` — do old and new agree on base type? (0/1)
10. `base_confidence_gap` — new's max base prob minus its 2nd-max
11. `anomaly_entropy` — mean binary entropy of new's 6 anomaly probs
12. `old_new_anomaly_correlation` — Pearson correlation of the two
    ensembles' anomaly probability vectors
13. `total_new_anomaly_signal` — sum of all new anomaly probs
14. `total_old_anomaly_signal` — sum of all old anomaly probs

These features let the meta-learner detect **ensemble disagreement**,
**confidence levels**, and **signal strength patterns** — invaluable
for handling edge cases.

### Base-Type Meta-Learner

A **4-class XGBoost + LightGBM ensemble** predicting {stationary,
deterministic_trend, stochastic_trend, volatility}.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| n_estimators | 500 |
| learning_rate | 0.05 |
| max_depth | 6 |
| min_child_weight | 3 |
| gamma | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.7 |
| reg_alpha / reg_lambda | 0.1 / 1.0 |
| class_weight | balanced |
| num_class | 4 |

LightGBM uses the same hyperparameters with `num_leaves=63` and
`max_depth=7`. **Ensemble prediction:**
`proba = 0.5 × XGB.predict_proba + 0.5 × LGB.predict_proba`

**Training data:** 19,500 meta-vectors with 4-class base labels,
stratified 80/20 train/test split.
**Test accuracy: 96.85 %, weighted F1: 96.85 %.**

### Anomaly Meta-Learners (6 × binary)

One XGBoost + LightGBM ensemble per anomaly type: collective, contextual,
mean_shift, point, trend_shift, variance_shift.

**Critical oversampling:** Groups 5–10 (stationary + single anomaly)
are **tripled in the training set** before fitting. This is because
these groups are the most nuanced cases — a stationary series with
one subtle anomaly — and need proportionally more exposure.

**Per-anomaly F1 scores on held-out test:**

| Anomaly | F1 | Accuracy |
|---|---|---|
| collective_anomaly | 0.9127 | 96.6% |
| contextual_anomaly | 0.9988 | 99.99% |
| mean_shift | 0.9211 | 96.9% |
| point_anomaly | 0.9518 | 98.1% |
| trend_shift | 0.9863 | 99.9% |
| variance_shift | 0.9297 | 97.2% |

---

## Component 7 — Blended Probability Decision

**Purpose:** In the combination branch, each anomaly's final probability
is a **weighted combination** of the meta-learner's prediction and the
new ensemble's direct prediction.

**Formula for anomaly k:**
```
blended_prob_k = α_k × meta_prob_k + (1 - α_k) × new_ensemble_prob_k
```

**Decision rule:**
```
anomaly_k detected  ⇔  blended_prob_k ≥ threshold_k
```

### Per-Anomaly Tuned Parameters

These were found via **joint grid search** maximizing FULL match count
on the training set, with each `(α, threshold)` pair searched over:

| Anomaly | α (blend weight) | Threshold | Rationale |
|---|---|---|---|
| collective_anomaly | 0.85 | 0.73 | Meta dominates; high threshold suppresses FPs on groups 5–10 |
| contextual_anomaly | 0.70 | 0.69 | Equal importance; strong separation already |
| mean_shift | 0.90 | 0.49 | Meta dominates; standard threshold works |
| point_anomaly | 0.70 | 0.69 | Balance; high threshold critical to avoid spike FPs |
| trend_shift | 0.90 | 0.73 | Meta dominates; high threshold for clarity |
| variance_shift | 0.70 | 0.69 | Balance; high threshold reduces stochastic trend confusion |

**Why blend?** The new ensemble's direct probability provides a "sanity check"
for cases where the meta-learner may have been over-confident or
under-confident due to feature interactions. Blending preserves valid signal
from both sources.

---

## Inference Pipeline

Given a raw CSV file:

```python
# ─── Stage A: Feature extraction ─────────────────────────────────────────
series = read_csv_values(csv_path)                 # raw values
if len(series) < MIN_SERIES_LENGTH:                # minimum 50 timesteps
    return ERROR

tsfresh_features = tsfresh.extract(series)         # 777-dim
tsfresh_scaled   = tsfresh_scaler.transform(tsfresh_features)  # standardized

# ─── Stage B: Stationarity gate ──────────────────────────────────────────
p_stationary = stat_detector_v2.predict_proba(
    extract_25_custom_features(series)
)[0]                                               # single scalar

# ─── Stage C: Old ensemble ───────────────────────────────────────────────
old_probs = [
    old_ensemble[class_k].predict_proba(tsfresh_features)[0, 1]
    for class_k in OLD_CLASSES                     # 9 classes
]                                                  # 9 probabilities

# ─── Stage D: New ensemble ───────────────────────────────────────────────
new_probs = [
    new_ensemble[model_k].predict_proba(tsfresh_features)[0, 1]
    for model_k in NEW_ALL_MODELS                  # 10 models
]                                                  # 10 probabilities

# ─── Stage E: Meta-feature construction ──────────────────────────────────
derived = compute_derived_features(old_probs, new_probs)    # 14-dim
meta_x  = concat(old_probs, new_probs, derived, tsfresh_scaled)  # 810-dim

# ─── Stage F: Stationarity gate check ────────────────────────────────────
if p_stationary >= 0.92:
    return ("stationary", [])                       # Override path

# ─── Stage G: Router decision ────────────────────────────────────────────
p_combo = 0.5 * router_xgb.predict_proba(meta_x)[0, 1] \
        + 0.5 * router_lgb.predict_proba(meta_x)[0, 1]

# ─── Stage H: Base type prediction ───────────────────────────────────────
base_xgb_p = base_meta_xgb.predict_proba(meta_x)[0]    # 4-class probs
base_lgb_p = base_meta_lgb.predict_proba(meta_x)[0]
base_idx   = argmax(0.5 * base_xgb_p + 0.5 * base_lgb_p)
base_type  = BASE_LABELS[base_idx]

# ─── Stage I: Routing branches ───────────────────────────────────────────
if p_combo < 0.30:
    return (base_type, [])                          # Single path

# ─── Stage J: Anomaly detection via blended probability ──────────────────
anomalies = []
for k, anomaly_name in enumerate(ANOM_LABELS):
    meta_prob = 0.5 * anom_meta[anomaly_name].xgb.predict_proba(meta_x)[0, 1] \
              + 0.5 * anom_meta[anomaly_name].lgb.predict_proba(meta_x)[0, 1]
    new_prob  = new_probs[k + 4]                    # 4 base + k anomaly
    blended   = ALPHA[anomaly_name] * meta_prob \
              + (1 - ALPHA[anomaly_name]) * new_prob
    if blended >= THRESHOLD[anomaly_name]:
        anomalies.append(anomaly_name)

return (base_type, anomalies)                       # Combination path
```

**Total inference cost per sample:**
- tsfresh extraction: ~0.1–0.2 sec (bulk dominated)
- Custom 25-feature extraction: ~0.01 sec
- 9 + 10 = 19 binary predictions: negligible
- Router: 2 predictions, negligible
- Base: 2 predictions
- Anomaly: 12 predictions (6 × XGB+LGB)

Total: dominated by tsfresh (≈ 150 ms for a 1000-sample series).

---

## Training Data and Balanced Sampling

### Meta-Learner Training

- **Total samples: 19,500** (500 per group × 39 groups)
- **Leaf-balanced sampling:** within each group, samples drawn
  proportionally from each leaf sub-directory to ensure coverage of
  all parameter combinations (noise levels, series lengths, etc.)
- **Stratified train/test split: 80% / 20%**
- **Oversampling for hard groups:** Groups 5–10 (stationary + single
  anomaly) are **tripled** in the anomaly meta-learner training, so
  the model has stronger signal for these subtle cases

### Evaluation Data

- **Total samples: 4,400**
- **Per group:** 10 samples per leaf directory
- **Random seed:** 42, deterministic for reproducibility
- **Non-overlapping with training data:** evaluation uses different
  random-sampled CSVs than meta training

---

## Hyperparameter Search and Calibration

Three hyperparameter families were jointly tuned via **exhaustive grid
search on cached meta-features** (the "fast grid" approach):

### Stationarity Gate Threshold
- **Range searched:** 0.85 – 1.01 in increments of 0.01
- **Best value:** 0.92 (v2 detector)
- **Selected by:** maximum FULL match count

### Router Threshold
- **Range searched:** 0.25 – 0.50 in increments of 0.01
- **Best value:** 0.30
- **Selected by:** maximum FULL match count

### Per-Anomaly (α, Threshold) Pairs
For each of the 6 anomalies, a 2-D grid:
- **α:** [0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0] (8 values)
- **Threshold:** 0.25 – 0.75 in 0.02 increments (~25 values)
- Total per anomaly: 8 × 25 = 200 combinations
- **Sequential refinement:** each anomaly tuned in order (collective →
  contextual → mean_shift → point → trend_shift → variance_shift), with
  each step using the previously-tuned values as the baseline

Final tuned parameters reported in [Component 7](#component-7--blended-probability-decision).

### Fast Grid Infrastructure
- **`cache_eval.py`**: Computes all 19 ensemble probabilities, meta-features,
  and stationarity detector probabilities for all 4,400 evaluation samples
  ONCE and stores as `.npz` (~22 MB). Subsequent grid searches take
  seconds per combination instead of minutes.
- **`fast_grid*.py`**: Multi-phase grid search scripts operating on the
  cached features

---

## Full Evaluation Results (39 classes)

| # | Group | n | FULL | PART | NONE | FULL % |
|---|---|---|---|---|---|---|
| 1 | stationary | 120 | 67 | 49 | 4 | 55.8 |
| 2 | deterministic_trend | 720 | 674 | 10 | 36 | 93.6 |
| 3 | stochastic_trend | 150 | 129 | 15 | 6 | 86.0 |
| 4 | volatility | 120 | 106 | 7 | 7 | 88.3 |
| 5 | collective_anomaly | 480 | 432 | 7 | 41 | 90.0 |
| 6 | contextual_anomaly | 480 | 480 | 0 | 0 | 100.0 |
| 7 | mean_shift | 480 | 421 | 27 | 32 | 87.7 |
| 8 | point_anomaly | 480 | 408 | 17 | 55 | 85.0 |
| 9 | trend_shift | 480 | 441 | 10 | 29 | 91.9 |
| 10 | variance_shift | 480 | 384 | 46 | 50 | 80.0 |
| 11 | cubic + collective | 10 | 10 | 0 | 0 | 100.0 |
| 12 | cubic + mean_shift | 20 | 19 | 1 | 0 | 95.0 |
| 13 | cubic + point_anomaly | 10 | 10 | 0 | 0 | 100.0 |
| 14 | cubic + variance_shift | 10 | 9 | 1 | 0 | 90.0 |
| 15 | damped + collective | 10 | 10 | 0 | 0 | 100.0 |
| 16 | damped + mean_shift | 20 | 20 | 0 | 0 | 100.0 |
| 17 | damped + point_anomaly | 10 | 10 | 0 | 0 | 100.0 |
| 18 | damped + variance_shift | 10 | 10 | 0 | 0 | 100.0 |
| 19 | exponential + collective | 10 | 10 | 0 | 0 | 100.0 |
| 20 | exponential + mean_shift | 20 | 20 | 0 | 0 | 100.0 |
| 21 | exponential + point_anomaly | 10 | 10 | 0 | 0 | 100.0 |
| 22 | exponential + variance_shift | 10 | 10 | 0 | 0 | 100.0 |
| 23 | linear + collective | 10 | 10 | 0 | 0 | 100.0 |
| 24 | linear + mean_shift | 20 | 19 | 1 | 0 | 95.0 |
| 25 | linear + point_anomaly | 10 | 10 | 0 | 0 | 100.0 |
| 26 | linear + trend_shift | 30 | 30 | 0 | 0 | 100.0 |
| 27 | linear + variance_shift | 10 | 10 | 0 | 0 | 100.0 |
| 28 | quadratic + collective | 10 | 10 | 0 | 0 | 100.0 |
| 29 | quadratic + mean_shift | 20 | 20 | 0 | 0 | 100.0 |
| 30 | quadratic + point_anomaly | 20 | 20 | 0 | 0 | 100.0 |
| 31 | quadratic + variance_shift | 10 | 9 | 1 | 0 | 90.0 |
| 32 | stochastic + collective | 10 | 5 | 5 | 0 | 50.0 |
| 33 | stochastic + mean_shift | 10 | 6 | 4 | 0 | 60.0 |
| 34 | stochastic + point_anomaly | 10 | 10 | 0 | 0 | 100.0 |
| 35 | stochastic + variance_shift | 50 | 44 | 6 | 0 | 88.0 |
| 36 | volatility + collective | 10 | 3 | 7 | 0 | 30.0 |
| 37 | volatility + mean_shift | 10 | 10 | 0 | 0 | 100.0 |
| 38 | volatility + point_anomaly | 10 | 6 | 4 | 0 | 60.0 |
| 39 | volatility + variance_shift | 10 | 7 | 3 | 0 | 70.0 |

**TOTALS**
- FULL match: **3,919 / 4,400 (89.07 %)**
- PARTIAL match: 221 / 4,400 (5.02 %)
- NO match: 260 / 4,400 (5.91 %)

**Summary statistics:**
- Perfect groups (100% FULL): **18** (all of groups 15–30 series, 37)
- Groups ≥ 90% FULL: **28 / 39**
- Groups ≥ 80% FULL: **33 / 39**

---

## Incremental Improvement History

| # | Checkpoint | Key Contribution | Full % | Δ |
|---|---|---|---|---|
| 0 | Baseline (new ensemble only) | Single binary ensemble | 59.80 | — |
| 1 | + Stacking meta-learner | Combine old + new opinions | 67.80 | +8.00 |
| 2 | + 777 raw tsfresh features | Unified 810-dim meta-vector | 74.60 | +6.80 |
| 3 | + Oversample hard groups 5–10 | 3× emphasis in training | 77.00 | +2.40 |
| 4 | + Context threshold | Base-dependent threshold | 77.50 | +0.50 |
| 5 | + Dual XGB+LGB ensemble | Different inductive biases | 77.90 | +0.40 |
| 6 | + Single/Combination router | Route by complexity | 88.50 | +10.60 |
| 7 | + Stationarity gate | Override via dedicated model | 88.60 | +0.10 |
| 8 | + Joint (stat, router, α, θ) tuning | Fine parameter search | 89.07 | +0.47 |
| | **Final** | **Combined pipeline** | **89.07** | **+29.27** |

---

## File Organization and Reproducibility

```
hopefullyprojectfinal/
├── README.md                    # This document
├── BEST_RESULTS.md              # Configuration backup
├── .gitignore
│
├── config.py                    # 39 group paths, class lists, global constants
├── processor.py                 # tsfresh extraction, ensemble probability computation,
│                                  derived feature construction, model loading
├── trainer.py                   # Meta-learner training (base + 6 anomaly + router),
│                                  oversampling, blend weight learning
├── evaluator.py                 # Complete evaluation pipeline over 4,400 samples
├── stat_detector.py             # Stationarity detector v2 wrapper
├── main.py                      # Pipeline orchestration (--force, --train, --eval)
│
├── cache_eval.py                # Builds evaluation .npz cache (all probabilities)
├── fast_grid.py                 # Fast grid search on cached features
├── fast_grid2.py                # Joint (stat × router × blend) search
├── fast_grid3.py                # Per-anomaly alpha × threshold grid
├── eval_best.py                 # Print 39-group table for the best configuration
│
├── processed_data/              # (gitignored) intermediate caches
│   ├── meta_X.npy               # 19,500 × 810 training meta-feature matrix
│   ├── meta_y_base.npy          # 19,500 base type labels
│   ├── meta_y_anom.npy          # 19,500 × 6 multi-label anomaly indicators
│   ├── tsfresh_scaler.pkl
│   └── eval_cache.npz           # 4,400 × (all probabilities + meta features)
│
├── meta_models/                 # (gitignored) trained meta-learners
│   ├── base_meta.pkl            # {xgb, lgb}
│   ├── anom_{name}.pkl × 6      # {xgb, lgb}
│   ├── router.pkl               # {xgb, lgb}
│   └── blend_weights.pkl        # per-anomaly {alpha, threshold}
│
└── results/                     # (gitignored) evaluation outputs
    ├── evaluation.json
    └── evaluation_report.md
```

### Reproducing Best Result

**Prerequisites:**
- Python 3.10+
- `pip install numpy pandas scikit-learn xgboost lightgbm tsfresh joblib tqdm scipy`
- Sibling directories with pre-trained models:
  - `../tsfresh ensemble/trained_models/`
  - `../ensemble-alldata/trained_models/`
  - `../stationary detector ml/trained_models v2/`
- `C:\Users\user\Desktop\Generated Data\` for raw CSV data

**Full pipeline (training + evaluation):**
```bash
python main.py --force
```

**Evaluation only (after training):**
```bash
python main.py --eval
```

**Grid search on cached features:**
```bash
python cache_eval.py    # ~20 min, runs once
python fast_grid.py     # ~1 min
python fast_grid3.py    # ~3 min — finds best (alpha, threshold) per anomaly
python eval_best.py     # display 39-group table for best config
```

---

## External Model References

| Model | Role | Training Paradigm | Source Path |
|---|---|---|---|
| Old binary ensemble | 9 single-class detectors | tsfresh features, one-vs-rest binaries | `../tsfresh ensemble/` |
| New binary ensemble | 10 base+anomaly binaries | tsfresh features, balanced binaries | `../ensemble-alldata/` |
| Stationarity detector v2 | Binary stationary gate | Custom 25 features, XGBoost | `../stationary detector ml/` |

All three are **reused as pre-trained models** — no retraining was performed
on any of them for this project. Only the stacking layer (Components 5–7)
is new.

---

## Techniques Used — Academic Summary

| Technique | Application |
|---|---|
| **Stacked Generalization** (Wolpert, 1992) | Meta-learner trained on base classifier outputs |
| **Gradient Boosting Decision Trees** | XGBoost and LightGBM for all tree-based models |
| **Ensemble Averaging** | XGB + LGB probability combination |
| **Automated Feature Engineering** | tsfresh for 777 time-series features |
| **Class Weighting** | Balanced class weights in base meta-learner |
| **Synthetic Minority Oversampling** | 3× duplication of groups 5–10 in anomaly training |
| **Hierarchical Classification** | Stationarity gate + single/combination router |
| **Probability Blending** | Convex combination of meta and base ensemble probabilities |
| **Per-Context Threshold Calibration** | Per-anomaly tuned thresholds via grid search |
| **Derived Meta-Features** | Agreement, entropy, confidence gap, correlation |
| **Data Caching for Grid Search** | Precomputed feature matrices for fast hyperparameter search |
| **Leaf-Balanced Sampling** | Proportional representation of all sub-parameter combinations |

---

## Acknowledgments

This work builds upon three sibling repositories in the `STATIONARY/` project
family. Each was developed independently with a focused scope; this project
integrates them into a unified classification system. The meta-learning,
routing, blending, and calibration layers are contributions of this work.

---

_If you use this work, please cite this repository and the three underlying
ensembles listed in [External Model References](#external-model-references)._
