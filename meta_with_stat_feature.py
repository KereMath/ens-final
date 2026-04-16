"""
Meta-learner'a stat_prob'u EKSTRA FEATURE olarak ekle.
810 -> 811 dim.

Hipotez: Meta-learner stat_prob'u bilirse, Group 1'in zor örneklerini
(stat orta-yüksek, anomali belirsiz) DAHA IYI sınıflandırır.
"""
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb

from config import (
    ANOM_LABELS, BASE_LABELS, GROUP_EXPECTED, GROUP_NAMES,
    META_MODELS_DIR, PROCESSED_DIR, RANDOM_STATE,
)
from fast_grid import match_type
from stat_detector import extract_feature_vector, load_stationary_detector
from processor import read_series, get_leaf_csvs
from config import GROUP_PATHS, META_N_PER_GROUP, MIN_SERIES_LENGTH, SOURCE_GROUPS
import random

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent / "meta_models_811"
MODELS_DIR.mkdir(exist_ok=True)


def compute_train_stat_probs():
    """Training sample'lar icin stat_prob hesapla."""
    cache_path = PROCESSED_DIR / "meta_stat_probs_train.npy"
    if cache_path.exists():
        print(f"  Cache: {cache_path}")
        return np.load(cache_path)

    print("  Fresh sampling for stat probs (aynı seed)...")
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Aynı sampling with processor.sample_group
    def sample_g(gid, n):
        leaf_map = {}
        for root in GROUP_PATHS[gid]:
            if root.exists():
                for leaf, csvs in get_leaf_csvs(root).items():
                    leaf_map.setdefault(leaf, []).extend(csvs)
        if not leaf_map:
            return []
        leaves = sorted(leaf_map.keys())
        per_leaf_max = max(10, n // len(leaves))
        phase1, leftover = [], {}
        for leaf in leaves:
            pool = sorted(leaf_map[leaf])
            if pool:
                c = random.choice(pool)
                phase1.append(c)
                leftover[leaf] = [f for f in pool if f != c]
            else:
                leftover[leaf] = []
        if len(phase1) >= n:
            random.shuffle(phase1)
            return phase1[:n]
        phase2 = []
        for leaf in leaves:
            b = n - len(phase1) - len(phase2)
            if b <= 0:
                break
            k = min(per_leaf_max - 1, len(leftover[leaf]), b)
            if k > 0:
                phase2.extend(random.sample(leftover[leaf], k))
        result = phase1 + phase2
        if len(result) < n:
            all_csvs = [f for csvs in leaf_map.values() for f in csvs]
            remaining = [f for f in all_csvs if f not in set(result)]
            extra = min(n - len(result), len(remaining))
            if extra > 0:
                result.extend(random.sample(remaining, extra))
        random.shuffle(result)
        return result[:n]

    all_csvs = []
    for gid, _, _ in SOURCE_GROUPS:
        all_csvs.extend(sample_g(gid, META_N_PER_GROUP))

    stat_model, stat_scaler, stat_selector = load_stationary_detector()
    exp_feat = stat_scaler.n_features_in_

    probs = []
    from tqdm import tqdm
    for csv_path in tqdm(all_csvs, desc="Stat"):
        d = read_series(csv_path)
        if len(d) < MIN_SERIES_LENGTH:
            probs.append(0.0)
            continue
        fv = extract_feature_vector(d)
        if fv is None or len(fv) != exp_feat:
            if fv is not None and len(fv) < exp_feat:
                fv = np.pad(fv, (0, exp_feat - len(fv)), 'constant')
            else:
                probs.append(0.0)
                continue
        x = fv.reshape(1, -1)
        x_s = stat_scaler.transform(x)
        if stat_selector is not None:
            x_s = stat_selector.transform(x_s)
        p = stat_model.predict_proba(x_s)[0, 0]  # class 0 = stationary
        probs.append(float(p))

    arr = np.array(probs)
    np.save(cache_path, arr)
    print(f"  Saved: {arr.shape}")
    return arr


def train_base(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    classes, counts = np.unique(y_tr, return_counts=True)
    total = len(y_tr)
    weights = {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_w = np.array([weights[int(y)] for y in y_tr])
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        min_child_weight=3, gamma=0.1, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, num_class=4, objective="multi:softprob",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    xgb_clf.fit(X_tr, y_tr, sample_weight=sample_w)
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        num_leaves=63, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    lgb_clf.fit(X_tr, y_tr)
    pred = np.argmax(0.5 * xgb_clf.predict_proba(X_te) + 0.5 * lgb_clf.predict_proba(X_te), axis=1)
    print(f"  Base F1={f1_score(y_te, pred, average='weighted', zero_division=0):.4f}")
    return {"xgb": xgb_clf, "lgb": lgb_clf}


def train_anom(X, y_anom, gids):
    mask = np.isin(gids, [5, 6, 7, 8, 9, 10])
    aug_X = np.vstack([X, np.tile(X[mask], (2, 1))])
    aug_y = np.vstack([y_anom, np.tile(y_anom[mask], (2, 1))])
    results = {}
    for i, anom_name in enumerate(ANOM_LABELS):
        y = aug_y[:, i]
        X_tr, X_te, y_tr, y_te = train_test_split(aug_X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        pos_w = max(int((y_tr == 0).sum()) / max(int(y_tr.sum()), 1), 1.0)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            scale_pos_weight=pos_w, random_state=RANDOM_STATE,
            n_jobs=-1, eval_metric="logloss", verbosity=0)
        xgb_clf.fit(X_tr, y_tr)
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            num_leaves=63, scale_pos_weight=pos_w,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        lgb_clf.fit(X_tr, y_tr)
        ens = 0.5 * xgb_clf.predict_proba(X_te)[:, 1] + 0.5 * lgb_clf.predict_proba(X_te)[:, 1]
        f1 = f1_score(y_te, (ens >= 0.5).astype(int), average="binary", zero_division=0)
        print(f"  {anom_name:<25} F1={f1:.4f}")
        results[anom_name] = {"xgb": xgb_clf, "lgb": lgb_clf}
    return results


def train_router(X, gids):
    y = np.where(np.isin(gids, [1, 2, 3, 4]), 0, 1).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    pos_w = max(int((y_tr == 0).sum()) / max(int(y_tr.sum()), 1), 1.0)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        scale_pos_weight=pos_w, random_state=RANDOM_STATE,
        n_jobs=-1, eval_metric="logloss", verbosity=0)
    xgb_clf.fit(X_tr, y_tr)
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        scale_pos_weight=pos_w, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1)
    lgb_clf.fit(X_tr, y_tr)
    ens = 0.5 * xgb_clf.predict_proba(X_te)[:, 1] + 0.5 * lgb_clf.predict_proba(X_te)[:, 1]
    f1 = f1_score(y_te, (ens >= 0.5).astype(int), average="binary", zero_division=0)
    print(f"  Router F1={f1:.4f}")
    return {"xgb": xgb_clf, "lgb": lgb_clf}


def main():
    print("=" * 70)
    print("  META-LEARNER + STAT_PROB FEATURE (811-dim)")
    print("=" * 70)

    meta_X = np.load(PROCESSED_DIR / "meta_X.npy")
    y_base = np.load(PROCESSED_DIR / "meta_y_base.npy")
    y_anom = np.load(PROCESSED_DIR / "meta_y_anom.npy")
    gids = np.load(PROCESSED_DIR / "meta_gids.npy")
    print(f"  meta_X: {meta_X.shape}")

    print("\n[1] Training stat_prob hesaplama...")
    stat_probs_train = compute_train_stat_probs()
    n = min(len(meta_X), len(stat_probs_train))
    meta_X_811 = np.hstack([meta_X[:n], stat_probs_train[:n].reshape(-1, 1)])
    y_base, y_anom, gids = y_base[:n], y_anom[:n], gids[:n]
    print(f"  meta_X_811: {meta_X_811.shape}")

    print("\n[2] Base meta")
    base_meta = train_base(meta_X_811, y_base)
    joblib.dump(base_meta, MODELS_DIR / "base_meta.pkl")

    print("\n[3] Anomaly meta (6x oversampled)")
    anom_metas = train_anom(meta_X_811, y_anom, gids)
    for n_a, m in anom_metas.items():
        joblib.dump(m, MODELS_DIR / f"anom_{n_a}.pkl")

    print("\n[4] Router")
    router = train_router(meta_X_811, gids)
    joblib.dump(router, MODELS_DIR / "router.pkl")

    # Eval
    print("\n[5] Eval (811-dim)")
    cache = dict(np.load(PROCESSED_DIR / "eval_cache.npz"))
    stat_eval = cache["stat_v2"]
    eval_811 = np.hstack([cache["meta_X"], stat_eval.reshape(-1, 1)])

    blend = {
        "collective_anomaly":  {"alpha": 0.85, "threshold": 0.73},
        "contextual_anomaly":  {"alpha": 0.70, "threshold": 0.69},
        "mean_shift":          {"alpha": 0.90, "threshold": 0.49},
        "point_anomaly":       {"alpha": 0.70, "threshold": 0.69},
        "trend_shift":         {"alpha": 0.90, "threshold": 0.73},
        "variance_shift":      {"alpha": 0.70, "threshold": 0.69},
    }

    xgb_bp = base_meta["xgb"].predict_proba(eval_811)
    lgb_bp = base_meta["lgb"].predict_proba(eval_811)
    base_pred = np.argmax(0.5 * xgb_bp + 0.5 * lgb_bp, axis=1)
    xgb_r = router["xgb"].predict_proba(eval_811)[:, 1]
    lgb_r = router["lgb"].predict_proba(eval_811)[:, 1]
    router_p = 0.5 * xgb_r + 0.5 * lgb_r
    anom_probs = {}
    for a, m in anom_metas.items():
        xp = m["xgb"].predict_proba(eval_811)[:, 1]
        lp = m["lgb"].predict_proba(eval_811)[:, 1]
        anom_probs[a] = 0.5 * xp + 0.5 * lp
    raw_new_anom = cache["raw_new"][:, 4:]

    results_per_gid = defaultdict(lambda: {"full": 0, "partial": 0, "none": 0})
    for i in range(len(cache["gid"])):
        gid = int(cache["gid"][i])
        expected = GROUP_EXPECTED[gid]
        base_type = BASE_LABELS[base_pred[i]]
        if stat_eval[i] >= 0.92:
            pb, pa = "stationary", []
        elif router_p[i] < 0.30:
            pb, pa = base_type, []
        else:
            anomalies = []
            for j, anom_name in enumerate(ANOM_LABELS):
                p = blend.get(anom_name, {"alpha": 1.0, "threshold": 0.5})
                meta_p = anom_probs[anom_name][i]
                new_p = float(raw_new_anom[i, j])
                blended = p["alpha"] * meta_p + (1 - p["alpha"]) * new_p
                eff_t = min(p["threshold"], 0.0) if base_type == "stationary" else p["threshold"]
                if blended >= eff_t:
                    anomalies.append(anom_name)
            pb, pa = base_type, anomalies
        mt = match_type(pb, pa, expected)
        results_per_gid[gid][mt.lower()] += 1

    total = sum(sum(v.values()) for v in results_per_gid.values())
    full = sum(v["full"] for v in results_per_gid.values())
    print(f"\n  FULL: {full}/{total} ({100*full/total:.2f}%)  [v12: 3919/89.07%]")
    print(f"  Delta vs v12: {full - 3919:+d}")

    print(f"\n  PER-GROUP:")
    for gid in sorted(results_per_gid.keys()):
        r = results_per_gid[gid]
        tot = r["full"] + r["partial"] + r["none"]
        pct = 100 * r["full"] / tot if tot else 0
        print(f"  Grp {gid:2d} ({GROUP_NAMES[gid][:25]:<25}) FULL={r['full']}/{tot} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
