"""
Meta-learner'i Group 1 oversample ile yeniden egit.
Amac: Group 1 (stationary, anomali yok) pattern'ini daha iyi ogrensin.

Onceki: Grup 5-10 x3 oversample (9000 sample), Grup 1 sadece 500.
Yeni: Grup 1 x5 oversample (2500), gruplar 5-10 x3 ayni.
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

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent / "meta_models_g1_oversample"
MODELS_DIR.mkdir(exist_ok=True)


def train_base_g1oversample(X, y_base, gids, g1_factor=5):
    """Group 1 oversample."""
    mask_g1 = (gids == 1)
    n_g1 = mask_g1.sum()
    if n_g1 > 0 and g1_factor > 1:
        extra_X = np.tile(X[mask_g1], (g1_factor - 1, 1))
        extra_y = np.tile(y_base[mask_g1], g1_factor - 1)
        X_aug = np.vstack([X, extra_X])
        y_aug = np.concatenate([y_base, extra_y])
    else:
        X_aug, y_aug = X, y_base
    print(f"  Base train: {len(X)} -> {len(X_aug)} (+{g1_factor}x Group 1)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=RANDOM_STATE, stratify=y_aug)
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


def train_anom_g1oversample(X, y_anom, gids, g1_factor=5, gp510_factor=3):
    """Hem Group 1 (5x) hem Gruplar 5-10 (3x) oversample."""
    mask_g510 = np.isin(gids, [5, 6, 7, 8, 9, 10])
    mask_g1 = (gids == 1)

    aug_X = [X]
    aug_y = [y_anom]
    if mask_g510.any() and gp510_factor > 1:
        aug_X.append(np.tile(X[mask_g510], (gp510_factor - 1, 1)))
        aug_y.append(np.tile(y_anom[mask_g510], (gp510_factor - 1, 1)))
    if mask_g1.any() and g1_factor > 1:
        aug_X.append(np.tile(X[mask_g1], (g1_factor - 1, 1)))
        aug_y.append(np.tile(y_anom[mask_g1], (g1_factor - 1, 1)))

    aug_X = np.vstack(aug_X)
    aug_y = np.vstack(aug_y)
    print(f"  Anom train: {len(X)} -> {len(aug_X)} "
          f"(G1 x{g1_factor}, G5-10 x{gp510_factor})")

    results = {}
    for i, anom_name in enumerate(ANOM_LABELS):
        y = aug_y[:, i]
        X_tr, X_te, y_tr, y_te = train_test_split(
            aug_X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
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


def train_router_g1oversample(X, gids, g1_factor=5):
    """Router'i Group 1 oversample ile egit - 'single' class'i daha kuvvetli."""
    y = np.where(np.isin(gids, [1, 2, 3, 4]), 0, 1).astype(int)
    mask_g1 = (gids == 1)
    aug_X = [X]
    aug_y = [y]
    if mask_g1.any() and g1_factor > 1:
        aug_X.append(np.tile(X[mask_g1], (g1_factor - 1, 1)))
        aug_y.append(np.tile(y[mask_g1], g1_factor - 1))
    aug_X = np.vstack(aug_X)
    aug_y = np.concatenate(aug_y)
    print(f"  Router train: {len(X)} -> {len(aug_X)} (+{g1_factor}x G1)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        aug_X, aug_y, test_size=0.2, random_state=RANDOM_STATE, stratify=aug_y)
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


def evaluate(meta_eval, stat_probs, base_meta, anom_metas, router, blend, gids_eval, cache):
    xgb_bp = base_meta["xgb"].predict_proba(meta_eval)
    lgb_bp = base_meta["lgb"].predict_proba(meta_eval)
    base_pred = np.argmax(0.5 * xgb_bp + 0.5 * lgb_bp, axis=1)
    xgb_r = router["xgb"].predict_proba(meta_eval)[:, 1]
    lgb_r = router["lgb"].predict_proba(meta_eval)[:, 1]
    router_p = 0.5 * xgb_r + 0.5 * lgb_r
    anom_p = {}
    for a, m in anom_metas.items():
        xp = m["xgb"].predict_proba(meta_eval)[:, 1]
        lp = m["lgb"].predict_proba(meta_eval)[:, 1]
        anom_p[a] = 0.5 * xp + 0.5 * lp
    raw_new_a = cache["raw_new"][:, 4:]

    results_per_gid = defaultdict(lambda: {"full": 0, "partial": 0, "none": 0})
    for i in range(len(gids_eval)):
        gid = int(gids_eval[i])
        expected = GROUP_EXPECTED[gid]
        base_type = BASE_LABELS[base_pred[i]]
        if stat_probs[i] >= 0.92:
            pb, pa = "stationary", []
        elif router_p[i] < 0.30:
            pb, pa = base_type, []
        else:
            anomalies = []
            for j, anom in enumerate(ANOM_LABELS):
                p = blend.get(anom, {"alpha": 1.0, "threshold": 0.5})
                meta_pp = anom_p[anom][i]
                new_pp = float(raw_new_a[i, j])
                blended = p["alpha"] * meta_pp + (1 - p["alpha"]) * new_pp
                eff_t = min(p["threshold"], 0.0) if base_type == "stationary" else p["threshold"]
                if blended >= eff_t:
                    anomalies.append(anom)
            pb, pa = base_type, anomalies
        mt = match_type(pb, pa, expected)
        results_per_gid[gid][mt.lower()] += 1
    return results_per_gid


def main():
    print("=" * 70)
    print("  META-LEARNER RETRAIN — Group 1 OVERSAMPLE")
    print("=" * 70)

    X = np.load(PROCESSED_DIR / "meta_X.npy")
    y_base = np.load(PROCESSED_DIR / "meta_y_base.npy")
    y_anom = np.load(PROCESSED_DIR / "meta_y_anom.npy")
    gids = np.load(PROCESSED_DIR / "meta_gids.npy")
    n = min(len(X), len(y_base))
    X, y_base, y_anom, gids = X[:n], y_base[:n], y_anom[:n], gids[:n]
    print(f"  meta_X: {X.shape}")
    print(f"  G1 total: {(gids==1).sum()}")

    for G1_FACTOR in [3, 5, 8, 10]:
        print(f"\n{'='*70}")
        print(f"  G1 OVERSAMPLE FACTOR: {G1_FACTOR}")
        print(f"{'='*70}")

        print("\n[1] Base meta")
        base_meta = train_base_g1oversample(X, y_base, gids, g1_factor=G1_FACTOR)

        print("\n[2] Anomaly meta")
        anom_metas = train_anom_g1oversample(X, y_anom, gids,
                                              g1_factor=G1_FACTOR, gp510_factor=3)

        print("\n[3] Router")
        router = train_router_g1oversample(X, gids, g1_factor=G1_FACTOR)

        blend_v12 = {
            "collective_anomaly":  {"alpha": 0.85, "threshold": 0.73},
            "contextual_anomaly":  {"alpha": 0.70, "threshold": 0.69},
            "mean_shift":          {"alpha": 0.90, "threshold": 0.49},
            "point_anomaly":       {"alpha": 0.70, "threshold": 0.69},
            "trend_shift":         {"alpha": 0.90, "threshold": 0.73},
            "variance_shift":      {"alpha": 0.70, "threshold": 0.69},
        }

        print("\n[4] Eval")
        cache = dict(np.load(PROCESSED_DIR / "eval_cache.npz"))
        per_gid = evaluate(cache["meta_X"], cache["stat_v2"],
                           base_meta, anom_metas, router, blend_v12,
                           cache["gid"], cache)
        total = sum(sum(v.values()) for v in per_gid.values())
        full = sum(v["full"] for v in per_gid.values())
        g1 = per_gid.get(1, {}).get("full", 0)
        g5 = per_gid.get(5, {}).get("full", 0)
        g7 = per_gid.get(7, {}).get("full", 0)
        g8 = per_gid.get(8, {}).get("full", 0)
        g10 = per_gid.get(10, {}).get("full", 0)
        print(f"\n  G1 factor={G1_FACTOR}: FULL={full}/{total} ({100*full/total:.2f}%)  "
              f"G1={g1}/120 G5={g5} G7={g7} G8={g8} G10={g10}  [v12: 3919/89.07%]")


if __name__ == "__main__":
    main()
