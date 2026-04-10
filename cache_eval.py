"""
Evaluation verisini bir kez cache'le.
4400 seri icin:
- tsfresh features
- Meta features (810 boyut, eski + yeni ensemble + derived + tsfresh)
- Raw new probs (10 boyut)
- Base meta prediction (4-class argmax + probs)
- Anomaly meta predictions (6 binary xgb+lgb probs)
- Router predictions (P(combo))
- Stat detector probs for v1-v6
- Ground truth metadata (gid, leaf, csv, expected)
"""
import random
import warnings
from pathlib import Path

import joblib
import numpy as np

from config import (
    ANOM_LABELS, GROUP_EXPECTED, MIN_SERIES_LENGTH,
    META_MODELS_DIR, PROCESSED_DIR, RANDOM_STATE,
    SOURCE_GROUPS, STATIONARY_DETECTOR_DIR,
)
from evaluator import SAMPLES_PER_LEAF, build_meta_features_batch, load_meta_models
from processor import (
    get_leaf_csvs, load_new_ensemble,
    load_old_ensemble, read_series,
)

# Custom batch extraction with n_jobs=1 for stability
import pandas as pd
from tsfresh import extract_features as tsfresh_extract
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute


def extract_batch(series_list, n_jobs=1):
    dfs = []
    for i, s in enumerate(series_list):
        dfs.append(pd.DataFrame({"id": i, "time": np.arange(len(s)), "value": s}))
    combined = pd.concat(dfs, ignore_index=True)
    X_df = tsfresh_extract(
        combined, column_id="id", column_sort="time", column_value="value",
        default_fc_parameters=EfficientFCParameters(),
        disable_progressbar=True, n_jobs=n_jobs,
    )
    impute(X_df)
    return X_df.values

warnings.filterwarnings("ignore")


def _load_stat_v(version: str):
    """trained_models vX'i yukle."""
    import pickle
    mp = STATIONARY_DETECTOR_DIR.parent / f"trained_models {version}" / "xgboost_fast.joblib"
    sp = STATIONARY_DETECTOR_DIR.parent / f"trained_models {version}" / "scalers.pkl"
    if not mp.exists():
        return None
    model = joblib.load(mp)
    with open(sp, "rb") as f:
        scalers = pickle.load(f)
    return {"model": model, "scaler": scalers.get("main"), "selector": scalers.get("selector")}


def _stat_probs_batch(bundle, series_list):
    from stat_detector import extract_feature_vector
    expected = bundle["scaler"].n_features_in_
    feat = np.zeros((len(series_list), expected))
    for i, s in enumerate(series_list):
        fv = extract_feature_vector(s)
        if fv is None:
            continue
        if len(fv) != expected:
            if len(fv) < expected:
                fv = np.pad(fv, (0, expected - len(fv)), 'constant')
            else:
                fv = fv[:expected]
        feat[i] = fv
    scaled = bundle["scaler"].transform(feat)
    if bundle["selector"] is not None:
        scaled = bundle["selector"].transform(scaled)
    probs = bundle["model"].predict_proba(scaled)
    return probs[:, 0]  # P(stationary)


def main():
    cache_path = PROCESSED_DIR / "eval_cache.npz"

    import sys
    print("=" * 62, flush=True)
    print("  EVALUATION CACHE BUILD", flush=True)
    print("=" * 62, flush=True)

    # Tsfresh scaler
    tsfresh_scaler = joblib.load(PROCESSED_DIR / "tsfresh_scaler.pkl")

    # Modelleri yukle
    print("  Modeller yukleniyor...")
    old_models = load_old_ensemble()
    new_models = load_new_ensemble()
    base_meta, anom_metas, blend_params, router = load_meta_models()

    stat_versions = {}
    for v in ["v1", "v2", "v3", "v4", "v5", "v6"]:
        b = _load_stat_v(v)
        if b is not None:
            stat_versions[v] = b
            print(f"  Stat {v} yuklendi")

    # Topla
    random.seed(RANDOM_STATE)
    all_meta_X = []
    all_raw_new = []
    all_gid = []
    all_leaf = []
    all_csv = []
    all_stat_probs = {v: [] for v in stat_versions}
    all_base_proba = []   # base meta 4-class proba
    all_anom_proba = {a: [] for a in ANOM_LABELS}   # per anomaly blended meta prob (xgb+lgb avg)
    all_router_proba = []

    for gid, gname, roots in SOURCE_GROUPS:
        leaf_map = {}
        for root in roots:
            if root.exists():
                for leaf, csvs in get_leaf_csvs(root).items():
                    leaf_map.setdefault(leaf, []).extend(csvs)
        if not leaf_map:
            continue

        selected = []
        leaf_names = []
        for leaf in sorted(leaf_map.keys()):
            pool = sorted(leaf_map[leaf])
            valid = [f for f in pool if f.name != "metadata.csv"]
            k = min(SAMPLES_PER_LEAF, len(valid))
            if k > 0:
                chosen = random.sample(valid, k)
                for c in chosen:
                    selected.append(c)
                    leaf_names.append(leaf.name)

        series_list = []
        valid_csvs = []
        valid_leaves = []
        for csv_path, lname in zip(selected, leaf_names):
            data = read_series(csv_path)
            if len(data) >= MIN_SERIES_LENGTH:
                series_list.append(data)
                valid_csvs.append(str(csv_path))
                valid_leaves.append(lname)

        if not series_list:
            continue

        # tsfresh -> meta features -> raw new probs
        X_batch = extract_batch(series_list)
        meta_X, raw_new = build_meta_features_batch(
            old_models, new_models, tsfresh_scaler, X_batch)

        # Base meta probs (xgb+lgb avg)
        xgb_bp = base_meta["xgb"].predict_proba(meta_X)
        lgb_bp = base_meta["lgb"].predict_proba(meta_X)
        base_proba = 0.5 * xgb_bp + 0.5 * lgb_bp

        # Router probs
        xgb_r = router["xgb"].predict_proba(meta_X)[:, 1]
        lgb_r = router["lgb"].predict_proba(meta_X)[:, 1]
        router_proba = 0.5 * xgb_r + 0.5 * lgb_r

        # Anomaly meta probs (xgb+lgb avg) - per anomaly
        anom_proba_batch = {}
        for anom_name in ANOM_LABELS:
            if anom_name in anom_metas:
                models = anom_metas[anom_name]
                xp = models["xgb"].predict_proba(meta_X)[:, 1]
                lp = models["lgb"].predict_proba(meta_X)[:, 1]
                anom_proba_batch[anom_name] = 0.5 * xp + 0.5 * lp
            else:
                anom_proba_batch[anom_name] = np.zeros(len(series_list))

        # Stat probs her version icin
        stat_batch = {}
        for v, bundle in stat_versions.items():
            stat_batch[v] = _stat_probs_batch(bundle, series_list)

        # Hepsini ekle
        all_meta_X.append(meta_X)
        all_raw_new.append(raw_new)
        all_base_proba.append(base_proba)
        all_router_proba.extend(router_proba.tolist())
        for anom_name in ANOM_LABELS:
            all_anom_proba[anom_name].extend(anom_proba_batch[anom_name].tolist())
        for v in stat_versions:
            all_stat_probs[v].extend(stat_batch[v].tolist())
        all_gid.extend([gid] * len(series_list))
        all_leaf.extend(valid_leaves)
        all_csv.extend(valid_csvs)

        print(f"  Grup {gid:2d} cached: {len(series_list)}")

    # Concatenate
    meta_X_arr = np.vstack(all_meta_X)
    raw_new_arr = np.vstack(all_raw_new)
    base_proba_arr = np.vstack(all_base_proba)
    router_proba_arr = np.array(all_router_proba)
    gid_arr = np.array(all_gid, dtype=int)

    save_dict = {
        "meta_X": meta_X_arr,
        "raw_new": raw_new_arr,
        "base_proba": base_proba_arr,
        "router_proba": router_proba_arr,
        "gid": gid_arr,
    }
    for anom_name in ANOM_LABELS:
        save_dict[f"anom_{anom_name}"] = np.array(all_anom_proba[anom_name])
    for v in stat_versions:
        save_dict[f"stat_{v}"] = np.array(all_stat_probs[v])

    np.savez_compressed(cache_path, **save_dict)

    # Leaf/csv metadata ayri pickle
    import pickle
    with open(PROCESSED_DIR / "eval_cache_meta.pkl", "wb") as f:
        pickle.dump({"leaf": all_leaf, "csv": all_csv}, f)

    print(f"\n  Total: {len(gid_arr)} sample")
    print(f"  Cache: {cache_path}")
    print(f"  Size: {cache_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
