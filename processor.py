"""
hopefullyprojectfinal - Veri Isleme
Stacking meta-learner icin:
  1. Her 39 gruptan ornekle
  2. tsfresh feature extraction
  3. Eski + Yeni ensemble uzerinden 19 meta-feature uret
  4. Ground truth etiketlerini kaydet
"""
import json
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tsfresh import extract_features as tsfresh_extract
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

from config import (
    GROUP_EXPECTED, GROUP_PATHS, META_N_PER_GROUP,
    MIN_SERIES_LENGTH, NEW_ALL_MODELS, NEW_ANOMALY_MODELS,
    NEW_BASE_MODELS, NEW_ENSEMBLE_DIR,
    OLD_CLASSES, OLD_ENSEMBLE_DIR, PROCESSED_DIR,
    RANDOM_STATE, SOURCE_GROUPS, STATIONARY_DETECTOR_DIR,
)


warnings.filterwarnings("ignore")
random.seed(RANDOM_STATE)


# ===================================================================
# CSV okuma (tum projede paylasilan)
# ===================================================================
def read_series(csv_path: Path) -> np.ndarray:
    try:
        df = pd.read_csv(csv_path)
        for col in ("data", "value", "values", "y"):
            if col in df.columns:
                return df[col].dropna().values.astype(float)
        num_cols = df.select_dtypes(include=[float, int]).columns
        if len(num_cols) > 0:
            return df[num_cols[0]].dropna().values.astype(float)
    except Exception:
        pass
    return np.array([])


# ===================================================================
# Leaf CSV listeleme
# ===================================================================
def get_leaf_csvs(root: Path) -> Dict[Path, List[Path]]:
    leaf_map: Dict[Path, List[Path]] = {}
    for f in root.rglob("*.csv"):
        if f.name == "metadata.csv":
            continue
        leaf_map.setdefault(f.parent, []).append(f)
    return leaf_map


def sample_group(group_id: int, n: int) -> List[Path]:
    """Bir kaynak grubundan n CSV sec (leaf-balanced)."""
    leaf_map: Dict[Path, List[Path]] = {}
    for root in GROUP_PATHS[group_id]:
        if root.exists():
            for leaf, csvs in get_leaf_csvs(root).items():
                leaf_map.setdefault(leaf, []).extend(csvs)
    if not leaf_map:
        return []

    leaves = sorted(leaf_map.keys())
    per_leaf = max(10, n // len(leaves))

    phase1 = []
    leftover = {}
    for leaf in leaves:
        pool = sorted(leaf_map[leaf])
        if pool:
            chosen = random.choice(pool)
            phase1.append(chosen)
            leftover[leaf] = [f for f in pool if f != chosen]
        else:
            leftover[leaf] = []

    if len(phase1) >= n:
        random.shuffle(phase1)
        return phase1[:n]

    phase2 = []
    for leaf in leaves:
        budget = n - len(phase1) - len(phase2)
        if budget <= 0:
            break
        pool = leftover[leaf]
        k = min(per_leaf - 1, len(pool), budget)
        if k > 0:
            phase2.extend(random.sample(pool, k))

    result = phase1 + phase2
    if len(result) < n:
        all_csvs = [f for csvs in leaf_map.values() for f in csvs]
        remaining = [f for f in all_csvs if f not in set(result)]
        extra = min(n - len(result), len(remaining))
        if extra > 0:
            result.extend(random.sample(remaining, extra))

    random.shuffle(result)
    return result[:n]


# ===================================================================
# Batch tsfresh extraction
# ===================================================================
def extract_batch(series_list: List[np.ndarray], n_jobs: int = 4) -> np.ndarray:
    dfs = []
    for i, s in enumerate(series_list):
        dfs.append(pd.DataFrame({"id": i, "time": np.arange(len(s)), "value": s}))
    combined = pd.concat(dfs, ignore_index=True)
    X_df = tsfresh_extract(
        combined,
        column_id="id", column_sort="time", column_value="value",
        default_fc_parameters=EfficientFCParameters(),
        disable_progressbar=True, n_jobs=n_jobs,
    )
    impute(X_df)
    return X_df.values


# ===================================================================
# Eski + Yeni ensemble yukle  (paylasilan)
# ===================================================================
def load_old_ensemble() -> Dict[str, Any]:
    models = {}
    for class_name in OLD_CLASSES:
        detector_dir = OLD_ENSEMBLE_DIR / class_name
        try:
            with open(detector_dir / "best_model_info.json") as f:
                best_info = json.load(f)
            best_name = best_info["best_model"]
            model  = joblib.load(detector_dir / f"{best_name}.joblib")
            scaler = joblib.load(detector_dir / "scalers.pkl")["scaler"]
            models[class_name] = {"model": model, "scaler": scaler}
        except Exception as e:
            print(f"  [WARN] Eski ensemble {class_name}: {e}")
    print(f"  Eski ensemble: {len(models)}/9 detector")
    return models


def load_new_ensemble() -> Dict[str, Any]:
    models = {}
    for name in NEW_ALL_MODELS:
        path = NEW_ENSEMBLE_DIR / f"{name}.pkl"
        if path.exists():
            models[name] = joblib.load(path)
        else:
            print(f"  [WARN] Yeni ensemble {name}: bulunamadi")
    print(f"  Yeni ensemble: {len(models)}/10 model")
    return models


def get_old_probs(models: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Tek ornek icin eski ensemble 9 olasilik."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    probs = []
    for name in OLD_CLASSES:
        bundle = models[name]
        X_s = bundle["scaler"].transform(X_clean)
        p = float(bundle["model"].predict_proba(X_s)[0, 1])
        probs.append(p)
    return np.array(probs)


def get_new_probs(models: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Tek ornek icin yeni ensemble 10 olasilik."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    probs = []
    for name in NEW_ALL_MODELS:
        bundle = models[name]
        X_s = bundle["scaler"].transform(X_clean)
        p = float(bundle["model"].predict_proba(X_s)[0, 1])
        probs.append(p)
    return np.array(probs)


# ===================================================================
# Meta-learner egitim verisi hazirlama
# ===================================================================
def prepare_meta_data(force: bool = False):
    """
    Her 39 gruptan META_N_PER_GROUP ornek al.
    tsfresh → eski + yeni ensemble → 19 meta-feature.
    Ground truth: base_type (int), 6x anomaly (binary).
    """
    meta_x_path   = PROCESSED_DIR / "meta_X.npy"
    meta_yb_path  = PROCESSED_DIR / "meta_y_base.npy"
    meta_ya_path  = PROCESSED_DIR / "meta_y_anom.npy"
    gid_path      = PROCESSED_DIR / "meta_gids.npy"

    if not force and meta_x_path.exists():
        meta_X    = np.load(meta_x_path)
        meta_yb   = np.load(meta_yb_path)
        meta_ya   = np.load(meta_ya_path)
        meta_gids = np.load(gid_path)
        print(f"  [Meta] Cache mevcut -> {meta_X.shape}")
        return meta_X, meta_yb, meta_ya, meta_gids

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Meta-learner verisi: {META_N_PER_GROUP}/grup x 39 grup")

    # --- 1. Ornekleme ---
    all_csvs = []
    all_gids = []
    for gid, gname, _ in SOURCE_GROUPS:
        sampled = sample_group(gid, META_N_PER_GROUP)
        all_csvs.extend(sampled)
        all_gids.extend([gid] * len(sampled))
        print(f"    Grup {gid:2d} ({gname[:30]:<30}): {len(sampled)} CSV")

    # --- 2. CSV oku ---
    series_list = []
    valid_gids  = []
    failed = 0
    for csv_path, gid in tqdm(zip(all_csvs, all_gids), total=len(all_csvs), desc="  CSV okuma"):
        data = read_series(csv_path)
        if len(data) < MIN_SERIES_LENGTH:
            failed += 1
            continue
        series_list.append(data)
        valid_gids.append(gid)

    print(f"  Basarili: {len(series_list)}  |  Atlanan: {failed}")

    # --- 3. tsfresh ---
    print(f"  tsfresh extraction ({len(series_list)} seri)...")
    X_tsfresh = extract_batch(series_list, n_jobs=4)

    # --- 4. Her iki ensemble'dan olasiliklar ---
    print("  Ensemble modelleri yukleniyor...")
    old_models = load_old_ensemble()
    new_models = load_new_ensemble()

    n = len(series_list)
    raw_old = np.zeros((n, 9))
    raw_new = np.zeros((n, 10))

    print(f"  Ensemble inference ({n} ornek)...")
    for i in tqdm(range(n), desc="  Inference"):
        raw_old[i] = get_old_probs(old_models, X_tsfresh[i])
        raw_new[i] = get_new_probs(new_models, X_tsfresh[i])

    # --- 4b. Turetilmis meta-feature'lar ---
    print("  Turetilmis meta-feature'lar hesaplaniyor...")
    derived = _compute_derived_features(raw_old, raw_new)

    # --- 4c. Ham tsfresh features (777 boyut, PCA yerine direkt) ---
    X_clean = np.nan_to_num(X_tsfresh, nan=0.0, posinf=0.0, neginf=0.0)
    tsfresh_scaler = StandardScaler()
    X_scaled = tsfresh_scaler.fit_transform(X_clean)
    joblib.dump(tsfresh_scaler, PROCESSED_DIR / "tsfresh_scaler.pkl")

    # --- 4d. Birlestir: 9 old + 10 new + derived + 777 tsfresh ---
    meta_features = np.hstack([raw_old, raw_new, derived, X_scaled])
    print(f"  Meta-features boyut: {meta_features.shape[1]} "
          f"(9 old + 10 new + {derived.shape[1]} derived + 777 tsfresh)")

    # --- 5. Ground truth etiketler ---
    BASE_MAP = {"stationary": 0, "deterministic_trend": 1,
                "stochastic_trend": 2, "volatility": 3}
    ANOM_LIST = ["collective_anomaly", "contextual_anomaly", "mean_shift",
                 "point_anomaly", "trend_shift", "variance_shift"]

    y_base = np.zeros(n, dtype=int)
    y_anom = np.zeros((n, 6), dtype=int)
    gids_arr = np.array(valid_gids, dtype=int)

    for i, gid in enumerate(valid_gids):
        exp = GROUP_EXPECTED[gid]
        y_base[i] = BASE_MAP[exp["base"]]
        for j, a in enumerate(ANOM_LIST):
            if a in exp["anomalies"]:
                y_anom[i, j] = 1

    # --- 6. Kaydet ---
    np.save(meta_x_path,  meta_features)
    np.save(meta_yb_path, y_base)
    np.save(meta_ya_path, y_anom)
    np.save(gid_path,     gids_arr)
    print(f"  Meta-features: {meta_features.shape} -> {PROCESSED_DIR}")
    print(f"  Base dagilimi: {dict(zip(*np.unique(y_base, return_counts=True)))}")
    print(f"  Anomali dagilimi: {y_anom.sum(axis=0).tolist()}")

    return meta_features, y_base, y_anom, gids_arr


def _compute_derived_features(raw_old: np.ndarray, raw_new: np.ndarray) -> np.ndarray:
    """
    19 ham olasiliktan turetilmis meta-feature'lar.
    Her satir icin ~14 ek feature.
    """
    n = raw_old.shape[0]
    feats = []

    # Old ensemble: base-related indices (det_trend=2, stoch=5, volatility=8)
    old_base_idx = [2, 5, 8]  # det_trend, stoch_trend, volatility in OLD_CLASSES order
    old_anom_idx = [0, 1, 3, 4, 6, 7]  # collective, contextual, mean, point, trend, variance

    # New ensemble: base indices 0-3, anomaly indices 4-9
    new_base_idx = [0, 1, 2, 3]
    new_anom_idx = [4, 5, 6, 7, 8, 9]

    for i in range(n):
        old = raw_old[i]
        new = raw_new[i]
        row = []

        # 1. Max old base prob + argmax
        old_base_probs = old[old_base_idx]
        row.append(np.max(old_base_probs))                    # max_old_base
        row.append(float(np.argmax(old_base_probs)))           # argmax_old_base

        # 2. Max old anomaly prob
        old_anom_probs = old[old_anom_idx]
        row.append(np.max(old_anom_probs))                     # max_old_anom
        row.append(float(np.sum(old_anom_probs > 0.5)))        # n_old_anom_above_05

        # 3. Max new base prob + argmax
        new_base_probs = new[new_base_idx]
        row.append(np.max(new_base_probs))                     # max_new_base
        row.append(float(np.argmax(new_base_probs)))           # argmax_new_base

        # 4. Max new anomaly prob
        new_anom_probs = new[new_anom_idx]
        row.append(np.max(new_anom_probs))                     # max_new_anom
        row.append(float(np.sum(new_anom_probs > 0.5)))        # n_new_anom_above_05

        # 5. Base agreement: old ve new ayni base mi?
        # Old: argmax of (det, stoch, vol) → 1,2,3; if all < 0.5 → 0 (stationary)
        old_base_map = [1, 2, 3]  # det, stoch, vol
        if np.max(old_base_probs) < 0.5:
            old_base_choice = 0  # stationary (no strong signal)
        else:
            old_base_choice = old_base_map[np.argmax(old_base_probs)]
        new_base_choice = int(np.argmax(new_base_probs))
        row.append(float(old_base_choice == new_base_choice))  # base_agreement

        # 6. Confidence gap: max_base - 2nd_max_base (new ensemble)
        sorted_base = np.sort(new_base_probs)[::-1]
        row.append(sorted_base[0] - sorted_base[1])           # base_confidence_gap

        # 7. Anomaly entropy (new) — yuksekse belirsizlik var
        anom_clipped = np.clip(new_anom_probs, 1e-6, 1 - 1e-6)
        anom_binary = np.stack([1 - anom_clipped, anom_clipped], axis=1)
        row.append(float(np.mean([sp_entropy(ab) for ab in anom_binary])))  # anom_entropy

        # 8. Old-new anomaly correlation (agreement)
        # old anom order: collective, contextual, mean, point, trend, variance
        # new anom order: same
        if np.std(old_anom_probs) > 0 and np.std(new_anom_probs) > 0:
            row.append(float(np.corrcoef(old_anom_probs, new_anom_probs)[0, 1]))
        else:
            row.append(0.0)                                     # anom_correlation

        # 9. Toplam anomali sinyal gucu
        row.append(float(np.sum(new_anom_probs)))              # total_new_anom_signal
        row.append(float(np.sum(old_anom_probs)))              # total_old_anom_signal

        feats.append(row)

    return np.array(feats)


if __name__ == "__main__":
    prepare_meta_data()
