"""
hopefullyprojectfinal - Meta-Learner Egitimi
Stacking: 19 meta-feature (9 eski + 10 yeni ensemble olasiliklari)
→ 1 base type classifier (4-class XGBoost)
→ 6 anomaly classifier (binary XGBoost)
"""
import json
import warnings

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score)
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb

from config import (ANOM_LABELS, BASE_LABELS, META_MODELS_DIR,
                    RANDOM_STATE, RESULTS_DIR, TEST_SIZE)

warnings.filterwarnings("ignore")


def train_base_meta(meta_X, y_base):
    """4-class base type: XGBoost + LightGBM ensemble."""
    print(f"\n{'='*62}")
    print(f"  BASE TYPE META-LEARNER (4-class, XGB+LGB ensemble)")
    print(f"{'='*62}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        meta_X, y_base, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_base)

    classes, counts = np.unique(y_tr, return_counts=True)
    total = len(y_tr)
    weights = {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_w = np.array([weights[int(y)] for y in y_tr])

    # XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        min_child_weight=3, gamma=0.1,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        num_class=4, objective="multi:softprob",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    xgb_clf.fit(X_tr, y_tr, sample_weight=sample_w)
    xgb_pred = xgb_clf.predict(X_te)
    xgb_f1 = f1_score(y_te, xgb_pred, average="weighted", zero_division=0)
    print(f"  XGBoost:  F1={xgb_f1:.4f}  Acc={accuracy_score(y_te, xgb_pred):.4f}")

    # LightGBM
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        num_leaves=63, subsample=0.8, colsample_bytree=0.7,
        class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    lgb_clf.fit(X_tr, y_tr)
    lgb_pred = lgb_clf.predict(X_te)
    lgb_f1 = f1_score(y_te, lgb_pred, average="weighted", zero_division=0)
    print(f"  LightGBM: F1={lgb_f1:.4f}  Acc={accuracy_score(y_te, lgb_pred):.4f}")

    # Ensemble: average probabilities
    xgb_proba = xgb_clf.predict_proba(X_te)
    lgb_proba = lgb_clf.predict_proba(X_te)
    ens_proba = 0.5 * xgb_proba + 0.5 * lgb_proba
    ens_pred = np.argmax(ens_proba, axis=1)
    ens_f1 = f1_score(y_te, ens_pred, average="weighted", zero_division=0)
    ens_acc = accuracy_score(y_te, ens_pred)
    print(f"  Ensemble: F1={ens_f1:.4f}  Acc={ens_acc:.4f}")
    print(classification_report(
        y_te, ens_pred, target_names=BASE_LABELS, digits=4, zero_division=0))

    # Her iki modeli kaydet
    ensemble = {"xgb": xgb_clf, "lgb": lgb_clf}
    return ensemble, {"acc": round(float(ens_acc), 4), "f1": round(float(ens_f1), 4)}


def train_anomaly_meta(meta_X, y_anom):
    """6 binary anomaly: XGBoost + LightGBM ensemble."""
    results = {}

    for i, anom_name in enumerate(ANOM_LABELS):
        y = y_anom[:, i]
        pos = int(y.sum())
        neg = int((y == 0).sum())

        print(f"\n  ANOMALY META: {anom_name}  (pos={pos}, neg={neg})")

        if pos < 5:
            results[anom_name] = None
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(
            meta_X, y, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=y)

        pos_w = max(int((y_tr == 0).sum()) / max(int(y_tr.sum()), 1), 1.0)

        # XGBoost
        xgb_clf = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            min_child_weight=3, gamma=0.1,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=pos_w,
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric="logloss", verbosity=0,
        )
        xgb_clf.fit(X_tr, y_tr)

        # LightGBM
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            num_leaves=63, subsample=0.8, colsample_bytree=0.7,
            scale_pos_weight=pos_w,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        lgb_clf.fit(X_tr, y_tr)

        # Ensemble prob
        xgb_p = xgb_clf.predict_proba(X_te)[:, 1]
        lgb_p = lgb_clf.predict_proba(X_te)[:, 1]
        ens_p = 0.5 * xgb_p + 0.5 * lgb_p
        ens_pred = (ens_p >= 0.5).astype(int)
        f1 = f1_score(y_te, ens_pred, average="binary", zero_division=0)
        acc = accuracy_score(y_te, ens_pred)
        print(f"    Ensemble F1={f1:.4f}  Acc={acc:.4f}")

        results[anom_name] = {
            "model": {"xgb": xgb_clf, "lgb": lgb_clf},
            "f1": round(float(f1), 4),
            "acc": round(float(acc), 4),
        }

    return results


def learn_blend_weights(meta_X, y_anom, anom_results):
    """
    Her anomali icin optimal (alpha, threshold) cifti ogren:
    blended = alpha * meta_prob + (1-alpha) * new_ensemble_prob
    final = blended >= threshold
    Grid search: alpha x threshold, F1 maximize ederek.
    """
    NEW_ANOM_START = 9 + 4
    blend_params = {}

    print(f"\n  BLEND WEIGHT + THRESHOLD OGRENIMI")
    for i, anom_name in enumerate(ANOM_LABELS):
        if anom_results.get(anom_name) is None:
            blend_params[anom_name] = {"alpha": 0.0, "threshold": 0.50}
            continue

        y = y_anom[:, i]
        models = anom_results[anom_name]["model"]
        new_probs = meta_X[:, NEW_ANOM_START + i]
        # Ensemble prob: average of XGB + LGB
        xgb_p = models["xgb"].predict_proba(meta_X)[:, 1]
        lgb_p = models["lgb"].predict_proba(meta_X)[:, 1]
        meta_probs = 0.5 * xgb_p + 0.5 * lgb_p

        best_alpha = 0.5
        best_threshold = 0.5
        best_f1 = 0.0

        for alpha in np.arange(0.0, 1.01, 0.05):
            blended = alpha * meta_probs + (1 - alpha) * new_probs
            for threshold in np.arange(0.38, 0.56, 0.02):
                pred = (blended >= threshold).astype(int)
                f1 = f1_score(y, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_alpha = alpha
                    best_threshold = threshold

        blend_params[anom_name] = {
            "alpha": round(float(best_alpha), 2),
            "threshold": round(float(best_threshold), 2),
        }
        print(f"    {anom_name:<25}  alpha={best_alpha:.2f}  thresh={best_threshold:.2f}  F1={best_f1:.4f}")

    return blend_params


def learn_calibrated_thresholds(meta_X, y_base, y_anom, base_clf):
    """
    Hibrit yaklasim: meta-learner base type tahminini kullanarak
    yeni ensemble'in anomali olasikliklari icin optimal esik ogren.
    meta_X icerisinde new ensemble anomali probs: index 13-18
    (9 old + 4 new_base = 13, sonraki 6 = anomali)
    """
    print(f"\n{'='*62}")
    print(f"  CALIBRATED THRESHOLD OGRENIMI (base x anomaly)")
    print(f"{'='*62}")

    NEW_ANOM_START = 9 + 4  # meta_X icinde yeni ensemble anomali prob baslangici

    # Meta-learner'in base tahminleri
    y_base_pred = base_clf.predict(meta_X)

    threshold_matrix = {}

    for base_idx in range(4):
        base_mask = (y_base_pred == base_idx)
        n_base = int(base_mask.sum())
        if n_base < 10:
            continue

        for anom_idx in range(6):
            anom_probs = meta_X[base_mask, NEW_ANOM_START + anom_idx]
            anom_true  = y_anom[base_mask, anom_idx]

            pos_count = int(anom_true.sum())
            neg_count = n_base - pos_count

            # F1 ile optimal esik bul
            best_t  = 0.50
            best_f1 = 0.0
            for t in np.arange(0.25, 0.85, 0.01):
                pred = (anom_probs >= t).astype(int)
                f1 = f1_score(anom_true, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t  = float(t)

            # Eger hic pozitif yoksa esik yukselt (FP bastirma)
            if pos_count == 0:
                best_t = max(best_t, 0.65)

            threshold_matrix[(base_idx, anom_idx)] = round(best_t, 2)

    # Tablo olarak yazdir
    base_names = ["stationary", "det_trend", "stoch_trend", "volatility"]
    anom_names = ["collective", "contextual", "mean_shift", "point", "trend_shift", "var_shift"]

    print(f"\n  {'':>15}", end="")
    for a in anom_names:
        print(f"  {a:>11}", end="")
    print()
    for bi, bn in enumerate(base_names):
        print(f"  {bn:>15}", end="")
        for ai in range(6):
            t = threshold_matrix.get((bi, ai), 0.50)
            marker = " *" if t != 0.50 else "  "
            print(f"  {t:>9.2f}{marker}", end="")
        print()

    return threshold_matrix


def oversample_anomaly_groups(meta_X, y_base, y_anom, gids):
    """
    Zayif gruplari oversample:
    - Gruplar 5-10 (stationary + anomali): 4x
    - Gruplar 32-39 (stoch/vol kombi): 3x
    """
    aug_X, aug_yb, aug_ya = meta_X.copy(), y_base.copy(), y_anom.copy()

    # Grup 5-10: 3x (stationary + anomali — en cok iyilesme buradan geldi)
    mask1 = np.isin(gids, [5, 6, 7, 8, 9, 10])
    n1 = int(mask1.sum())
    if n1 > 0:
        extra_X = np.tile(meta_X[mask1], (2, 1))  # 2 extra = 3x total
        extra_yb = np.tile(y_base[mask1], 2)
        extra_ya = np.tile(y_anom[mask1], (2, 1))
        aug_X = np.vstack([aug_X, extra_X])
        aug_yb = np.concatenate([aug_yb, extra_yb])
        aug_ya = np.vstack([aug_ya, extra_ya])
        print(f"  Oversample: grup 5-10 x3 ({n1} -> {n1 * 3})")

    print(f"  Augmented: {meta_X.shape[0]} -> {aug_X.shape[0]} ornek")
    return aug_X, aug_yb, aug_ya


def train_single_combo_router(meta_X, gids):
    """
    810 feature uzerinde tekli(0) vs combo(1) router.
    Grup 1-4 = single, Grup 5-39 = combo.
    """
    print(f"\n{'='*62}")
    print(f"  SINGLE vs COMBO ROUTER (810 feature, XGB+LGB)")
    print(f"{'='*62}")

    y_router = np.where(np.isin(gids, [1, 2, 3, 4]), 0, 1).astype(int)
    print(f"  Single: {(y_router==0).sum()}  |  Combo: {(y_router==1).sum()}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        meta_X, y_router, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_router)

    pos_w = max(int((y_tr == 0).sum()) / max(int(y_tr.sum()), 1), 1.0)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        min_child_weight=3, gamma=0.1,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=pos_w,
        random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric="logloss", verbosity=0,
    )
    xgb_clf.fit(X_tr, y_tr)

    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        num_leaves=63, subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=pos_w,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    lgb_clf.fit(X_tr, y_tr)

    # Ensemble
    xgb_p = xgb_clf.predict_proba(X_te)[:, 1]
    lgb_p = lgb_clf.predict_proba(X_te)[:, 1]
    ens_p = 0.5 * xgb_p + 0.5 * lgb_p
    ens_pred = (ens_p >= 0.5).astype(int)

    f1 = f1_score(y_te, ens_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_te, ens_pred)
    print(f"  Router Ensemble: F1={f1:.4f}  Acc={acc:.4f}")
    print(classification_report(
        y_te, ens_pred, target_names=["single", "combo"], digits=4, zero_division=0))

    router = {"xgb": xgb_clf, "lgb": lgb_clf}
    return router, {"f1": round(float(f1), 4), "acc": round(float(acc), 4)}


def train_all_meta(meta_X, y_base, y_anom, gids=None):
    print("\n" + "=" * 62)
    print("  STACKING META-LEARNER EGITIMI (HIBRIT v7)")
    print(f"  Input: {meta_X.shape[1]} meta-feature  |  {meta_X.shape[0]} ornek")
    print(f"  Base type: meta-learner  |  Anomali: blended + context threshold")
    print("=" * 62)

    # --- Single vs Combo Router ---
    if gids is not None:
        router, router_stats = train_single_combo_router(meta_X, gids)
    else:
        router, router_stats = None, {}

    # --- Base type meta-learner (orijinal veri ile) ---
    base_clf, base_stats = train_base_meta(meta_X, y_base)

    # --- Oversample gruplar 5-10 icin anomali meta-learner ---
    if gids is not None:
        aug_X, aug_yb, aug_ya = oversample_anomaly_groups(meta_X, y_base, y_anom, gids)
    else:
        aug_X, aug_yb, aug_ya = meta_X, y_base, y_anom

    # --- Anomaly meta-learners (augmented data ile) ---
    print(f"\n{'='*62}")
    print(f"  ANOMALY META-LEARNERS (6 binary, oversampled)")
    print(f"{'='*62}")
    anom_results = train_anomaly_meta(aug_X, aug_ya)

    # --- Blend weights + thresholds (augmented data ile) ---
    blend_params = learn_blend_weights(aug_X, aug_ya, anom_results)

    # --- Kaydet ---
    META_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(base_clf, META_MODELS_DIR / "base_meta.pkl")
    joblib.dump(blend_params, META_MODELS_DIR / "blend_weights.pkl")
    if router is not None:
        joblib.dump(router, META_MODELS_DIR / "router.pkl")
    for anom_name, res in anom_results.items():
        if res is not None:
            joblib.dump(res["model"], META_MODELS_DIR / f"anom_{anom_name}.pkl")  # dict with xgb + lgb

    # --- Ozet ---
    print(f"\n{'='*62}")
    print(f"  HIBRIT v3 OZET")
    print(f"{'='*62}")
    print(f"  Base type meta:  Acc={base_stats['acc']}  F1={base_stats['f1']}")
    for anom_name, res in anom_results.items():
        if res:
            bp = blend_params.get(anom_name, {})
            print(f"  {anom_name:<25}  F1={res['f1']}  Acc={res['acc']}  "
                  f"alpha={bp.get('alpha',0.5)}  thresh={bp.get('threshold',0.5)}")

    # JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "base_meta": base_stats,
        "approach": "full stacking v4: meta base + blended anomaly (meta*alpha + new*(1-alpha))",
        "blend_params": {k: v for k, v in blend_params.items()},
        "anomaly_meta": {
            k: {"f1": v["f1"], "acc": v["acc"]} if v else None
            for k, v in anom_results.items()
        },
        "n_samples": int(meta_X.shape[0]),
        "n_features": int(meta_X.shape[1]),
    }
    with open(RESULTS_DIR / "meta_training.json", "w") as f:
        json.dump(summary, f, indent=2)

    return base_clf, anom_results


if __name__ == "__main__":
    from processor import prepare_meta_data
    meta_X, y_base, y_anom, gids = prepare_meta_data()
    train_all_meta(meta_X, y_base, y_anom, gids)
