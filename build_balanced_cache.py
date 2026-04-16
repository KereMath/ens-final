"""
BALANCED eval cache olustur:
Her gruptan 1000 sample (kucuk gruplarda tamami) = ~39000 sample.

tsfresh + old + new ensemble + stat detector + meta_X - hepsi cache'lenir.
Sonra hizli threshold sweep yapilir.
"""
import json
import random
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from tsfresh import extract_features as tsfresh_extract
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

from config import (
    GROUP_EXPECTED, GROUP_NAMES, GROUP_PATHS, MIN_SERIES_LENGTH,
    NEW_ALL_MODELS, PROCESSED_DIR, RANDOM_STATE, SOURCE_GROUPS,
)
from processor import (
    _compute_derived_features, get_leaf_csvs, get_new_probs, get_old_probs,
    load_new_ensemble, load_old_ensemble, read_series,
)
from stat_detector import load_stationary_detector, predict_stationary_batch

warnings.filterwarnings("ignore")
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

OUT_PATH = PROCESSED_DIR / "balanced_eval_cache.npz"
N_PER_GROUP = 1000
CHUNK_SIZE = 1000


def collect_balanced_csvs():
    """Her gruptan N_PER_GROUP CSV leaf-balanced sec."""
    all_csvs = []
    total_avail = 0
    for gid, gname, roots in SOURCE_GROUPS:
        leaf_map = {}
        for root in roots:
            if root.exists():
                for leaf, csvs in get_leaf_csvs(root).items():
                    leaf_map.setdefault(leaf, []).extend(csvs)
        valid_all = []
        for leaf in sorted(leaf_map.keys()):
            valid_all.extend([c for c in leaf_map[leaf] if c.name != "metadata.csv"])
        total_avail += len(valid_all)

        # Leaf-balanced sampling
        leaves = sorted(leaf_map.keys())
        per_leaf = max(1, N_PER_GROUP // len(leaves))
        selected = []
        for leaf in leaves:
            valid = sorted([c for c in leaf_map[leaf] if c.name != "metadata.csv"])
            k = min(per_leaf, len(valid))
            if k > 0:
                selected.extend(random.sample(valid, k))
        if len(selected) < N_PER_GROUP and len(valid_all) >= N_PER_GROUP:
            remaining = [c for c in valid_all if c not in set(selected)]
            extra = min(N_PER_GROUP - len(selected), len(remaining))
            if extra > 0:
                selected.extend(random.sample(remaining, extra))
        selected = selected[:N_PER_GROUP]
        for c in selected:
            all_csvs.append((str(c), gid))
        print(f"  Grup {gid:2d} {gname[:28]:<28} {len(selected):>4}/{len(valid_all):<6} ({100*len(selected)/len(valid_all):.1f}%)")

    print(f"\n  Total: {len(all_csvs):,} from {total_avail:,} available")
    return all_csvs


def extract_batch_tsfresh(series_list, n_jobs=4):
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


def main():
    print("=" * 70)
    print(f"  BALANCED EVAL CACHE BUILD (N={N_PER_GROUP}/grup × 39)")
    print("=" * 70)

    print("\n[1] Modeller...")
    old_models = load_old_ensemble()
    new_models = load_new_ensemble()
    stat_model, stat_scaler, stat_selector = load_stationary_detector()
    tsfresh_scaler = joblib.load(PROCESSED_DIR / "tsfresh_scaler.pkl")

    print("\n[2] CSV sampling...")
    all_csvs = collect_balanced_csvs()

    # Accumulator
    meta_X_list = []
    raw_new_list = []
    stat_probs_list = []
    gid_list = []

    print(f"\n[3] Processing in chunks of {CHUNK_SIZE}...")
    pbar = tqdm(range(0, len(all_csvs), CHUNK_SIZE), desc="Chunks")
    for start in pbar:
        end = min(start + CHUNK_SIZE, len(all_csvs))
        chunk = all_csvs[start:end]

        # Read series
        series_list = []
        valid_gids = []
        for csv_path, gid in chunk:
            d = read_series(Path(csv_path))
            if len(d) >= MIN_SERIES_LENGTH:
                series_list.append(d)
                valid_gids.append(gid)

        if not series_list:
            continue

        # tsfresh
        X_tsfresh = extract_batch_tsfresh(series_list, n_jobs=4)

        # Ensemble
        n = len(series_list)
        raw_old = np.zeros((n, 9))
        raw_new = np.zeros((n, 10))
        for i in range(n):
            raw_old[i] = get_old_probs(old_models, X_tsfresh[i])
            raw_new[i] = get_new_probs(new_models, X_tsfresh[i])

        # Derived + tsfresh standardized
        derived = _compute_derived_features(raw_old, raw_new)
        X_clean = np.nan_to_num(X_tsfresh, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = tsfresh_scaler.transform(X_clean)
        meta_X_chunk = np.hstack([raw_old, raw_new, derived, X_scaled])

        # Stat detector
        stat_p = predict_stationary_batch(stat_model, stat_scaler, stat_selector, series_list)

        meta_X_list.append(meta_X_chunk)
        raw_new_list.append(raw_new)
        stat_probs_list.extend(stat_p.tolist())
        gid_list.extend(valid_gids)

        pbar.set_postfix({"n": sum(len(m) for m in meta_X_list)})

    meta_X = np.vstack(meta_X_list)
    raw_new = np.vstack(raw_new_list)
    stat_probs = np.array(stat_probs_list)
    gids = np.array(gid_list, dtype=int)

    print(f"\n  Final: meta_X={meta_X.shape}, raw_new={raw_new.shape}")
    np.savez_compressed(
        OUT_PATH,
        meta_X=meta_X,
        raw_new=raw_new,
        stat_v2=stat_probs,
        gid=gids,
    )
    print(f"  Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
