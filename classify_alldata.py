"""
v12 mevcut pipeline ile TÜM 847K CSV'yi classify et.
Chunked processing — kesilirse kaldigi yerden devam.
Sonuc: results/alldataresults.md + per-sample JSON.
"""
import json
import sys
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from tsfresh import extract_features as tsfresh_extract
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

from config import (
    ANOM_LABELS, BASE_LABELS, GROUP_EXPECTED, GROUP_NAMES, GROUP_PATHS,
    META_MODELS_DIR, MIN_SERIES_LENGTH, NEW_ALL_MODELS, NEW_ENSEMBLE_DIR,
    OLD_CLASSES, OLD_ENSEMBLE_DIR, PROCESSED_DIR, RESULTS_DIR, SOURCE_GROUPS,
)
from processor import (
    _compute_derived_features, get_leaf_csvs, get_new_probs, get_old_probs,
    load_new_ensemble, load_old_ensemble, read_series,
)
from stat_detector import (
    load_stationary_detector, predict_stationary_batch,
)

warnings.filterwarnings("ignore")

CHUNK_SIZE = 1000   # her chunk'ta tsfresh + inference
SAMPLE_FRACTION = 0.10   # her leaf'in 1/10 sample
PROGRESS_FILE = RESULTS_DIR / "alldata_progress.json"
RESULTS_FILE = RESULTS_DIR / "alldata_results.jsonl"
MD_FILE = RESULTS_DIR / "alldataresults.md"


def load_all_models():
    print("[1] Models loading...")
    old_models = load_old_ensemble()
    new_models = load_new_ensemble()
    base_meta = joblib.load(META_MODELS_DIR / "base_meta.pkl")
    anom_metas = {a: joblib.load(META_MODELS_DIR / f"anom_{a}.pkl")
                   for a in ANOM_LABELS
                   if (META_MODELS_DIR / f"anom_{a}.pkl").exists()}
    router = joblib.load(META_MODELS_DIR / "router.pkl")
    blend_params = joblib.load(META_MODELS_DIR / "blend_weights.pkl")
    tsfresh_scaler = joblib.load(PROCESSED_DIR / "tsfresh_scaler.pkl")
    stat_model, stat_scaler, stat_selector = load_stationary_detector()
    return (old_models, new_models, base_meta, anom_metas, router,
            blend_params, tsfresh_scaler, stat_model, stat_scaler, stat_selector)


def collect_all_csvs():
    """Her leaf'ten 1/10 sample (deterministik: her 10. dosyayi al)."""
    import random
    random.seed(42)
    print(f"[2] Collecting CSVs (sampling {SAMPLE_FRACTION:.0%} per leaf)...")
    csvs_with_meta = []
    total_available = 0
    for gid, gname, roots in SOURCE_GROUPS:
        leaf_map = {}
        for root in roots:
            if root.exists():
                for leaf, csvs in get_leaf_csvs(root).items():
                    leaf_map.setdefault(leaf, []).extend(csvs)
        for leaf in sorted(leaf_map.keys()):
            valid = sorted([c for c in leaf_map[leaf] if c.name != "metadata.csv"])
            total_available += len(valid)
            n_sample = max(1, int(len(valid) * SAMPLE_FRACTION))
            sampled = random.sample(valid, min(n_sample, len(valid)))
            for csv_path in sorted(sampled):
                csvs_with_meta.append((str(csv_path), gid, leaf.name))
    print(f"  Available: {total_available:,}, Sampled: {len(csvs_with_meta):,} "
          f"({100*len(csvs_with_meta)/total_available:.1f}%)")
    return csvs_with_meta


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


def classify_chunk(chunk, models):
    """Bir chunk'i classify et, sonuc liste donder."""
    (old_models, new_models, base_meta, anom_metas, router,
     blend_params, tsfresh_scaler, stat_model, stat_scaler, stat_selector) = models

    # 1. Read series
    series_list = []
    chunk_meta = []
    for csv_path, gid, leaf_name in chunk:
        d = read_series(Path(csv_path))
        if len(d) >= MIN_SERIES_LENGTH:
            series_list.append(d)
            chunk_meta.append((csv_path, gid, leaf_name))

    if not series_list:
        return []

    # 2. tsfresh
    X_tsfresh = extract_batch_tsfresh(series_list, n_jobs=4)

    # 3. Old + New ensemble
    n = len(series_list)
    raw_old = np.zeros((n, 9))
    raw_new = np.zeros((n, 10))
    for i in range(n):
        raw_old[i] = get_old_probs(old_models, X_tsfresh[i])
        raw_new[i] = get_new_probs(new_models, X_tsfresh[i])

    # 4. Derived + standardized tsfresh
    derived = _compute_derived_features(raw_old, raw_new)
    X_clean = np.nan_to_num(X_tsfresh, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = tsfresh_scaler.transform(X_clean)
    meta_X = np.hstack([raw_old, raw_new, derived, X_scaled])

    # 5. Stat detector
    stat_probs = predict_stationary_batch(stat_model, stat_scaler, stat_selector, series_list)

    # 6. Meta-learner inference
    xgb_bp = base_meta["xgb"].predict_proba(meta_X)
    lgb_bp = base_meta["lgb"].predict_proba(meta_X)
    base_pred = np.argmax(0.5 * xgb_bp + 0.5 * lgb_bp, axis=1)

    xgb_r = router["xgb"].predict_proba(meta_X)[:, 1]
    lgb_r = router["lgb"].predict_proba(meta_X)[:, 1]
    router_p = 0.5 * xgb_r + 0.5 * lgb_r

    anom_probs_d = {}
    for a, m in anom_metas.items():
        xp = m["xgb"].predict_proba(meta_X)[:, 1]
        lp = m["lgb"].predict_proba(meta_X)[:, 1]
        anom_probs_d[a] = 0.5 * xp + 0.5 * lp

    # 7. Decision logic (v12 pipeline)
    raw_new_anom = raw_new[:, 4:]
    results = []
    for i, (csv_path, gid, leaf_name) in enumerate(chunk_meta):
        expected = GROUP_EXPECTED[gid]
        base_type = BASE_LABELS[base_pred[i]]

        if stat_probs[i] >= 0.92:
            pred_base, pred_anoms = "stationary", []
        elif router_p[i] < 0.30:
            pred_base, pred_anoms = base_type, []
        else:
            anomalies = []
            for j, anom_name in enumerate(ANOM_LABELS):
                params = blend_params.get(anom_name, {"alpha": 1.0, "threshold": 0.5})
                meta_p = anom_probs_d[anom_name][i]
                new_p = float(raw_new_anom[i, j])
                blended = params["alpha"] * meta_p + (1 - params["alpha"]) * new_p
                eff_t = min(params["threshold"], 0.0) if base_type == "stationary" else params["threshold"]
                if blended >= eff_t:
                    anomalies.append(anom_name)
            pred_base, pred_anoms = base_type, anomalies

        # Match
        true_base = expected["base"]
        true_anoms = expected["anomalies"]
        base_ok = (pred_base == true_base)
        if not true_anoms:
            mt = "FULL" if base_ok and not pred_anoms else ("PARTIAL" if base_ok else "NONE")
        else:
            all_anom = all(a in pred_anoms for a in true_anoms)
            if base_ok and all_anom:
                mt = "FULL"
            elif base_ok or all_anom:
                mt = "PARTIAL"
            else:
                mt = "NONE"

        results.append({
            "csv": Path(csv_path).name,
            "leaf": leaf_name,
            "gid": gid,
            "true_base": true_base,
            "true_anoms": list(true_anoms),
            "pred_base": pred_base,
            "pred_anoms": pred_anoms,
            "match": mt,
        })

    return results


def write_markdown(per_gid_stats, total_full, total_part, total_none, total):
    lines = []
    lines.append(f"# All-Data Classification Results\n")
    lines.append(f"**v12 pipeline** üzerinde **{total:,}** CSV classify edildi.\n")
    lines.append(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## Özet\n")
    lines.append("| Metrik | Değer |")
    lines.append("|---|---|")
    lines.append(f"| Toplam | {total:,} |")
    lines.append(f"| FULL | {total_full:,} ({100*total_full/total:.2f}%) |")
    lines.append(f"| PARTIAL | {total_part:,} ({100*total_part/total:.2f}%) |")
    lines.append(f"| NONE | {total_none:,} ({100*total_none/total:.2f}%) |")
    lines.append("\n## Grup Bazlı Sonuçlar\n")
    lines.append("| # | Grup | Beklenen | n | FULL | PART | NONE | FULL% |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for gid in sorted(per_gid_stats.keys()):
        r = per_gid_stats[gid]
        n = r["full"] + r["partial"] + r["none"]
        pct = 100 * r["full"] / n if n else 0
        exp = GROUP_EXPECTED[gid]
        exp_str = exp["base"]
        if exp["anomalies"]:
            exp_str += " + " + " + ".join(exp["anomalies"])
        lines.append(f"| {gid} | {GROUP_NAMES[gid]} | `{exp_str}` | {n:,} | "
                      f"{r['full']:,} | {r['partial']:,} | {r['none']:,} | {pct:.2f} |")

    with open(MD_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    models = load_all_models()
    all_csvs = collect_all_csvs()

    # Resume
    processed_idx = 0
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            processed_idx = json.load(f).get("processed", 0)
        print(f"[3] Resuming from index {processed_idx}")

    # Open results jsonl in append mode
    results_f = open(RESULTS_FILE, "a")

    # Stats accumulator
    per_gid_stats = defaultdict(lambda: {"full": 0, "partial": 0, "none": 0})
    total_full = total_part = total_none = 0

    # If resuming, accumulate existing results stats
    if processed_idx > 0 and RESULTS_FILE.exists():
        print("  Accumulating existing stats...")
        with open(RESULTS_FILE) as f:
            for line in f:
                r = json.loads(line)
                gid = r["gid"]
                m = r["match"]
                per_gid_stats[gid][m.lower()] += 1
                if m == "FULL": total_full += 1
                elif m == "PARTIAL": total_part += 1
                else: total_none += 1
        print(f"  Existing: FULL={total_full} PART={total_part} NONE={total_none}")

    # Chunk loop
    print(f"[4] Processing {len(all_csvs) - processed_idx:,} samples in chunks of {CHUNK_SIZE}")
    pbar = tqdm(range(processed_idx, len(all_csvs), CHUNK_SIZE), desc="Chunks")
    for start in pbar:
        end = min(start + CHUNK_SIZE, len(all_csvs))
        chunk = all_csvs[start:end]
        try:
            results = classify_chunk(chunk, models)
        except Exception as e:
            print(f"  Chunk {start}-{end} ERROR: {e}")
            continue

        # Write incrementally
        for r in results:
            results_f.write(json.dumps(r) + "\n")
            per_gid_stats[r["gid"]][r["match"].lower()] += 1
            if r["match"] == "FULL": total_full += 1
            elif r["match"] == "PARTIAL": total_part += 1
            else: total_none += 1
        results_f.flush()

        # Save progress
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"processed": end, "total": len(all_csvs)}, f)

        # Update markdown periodically
        total = total_full + total_part + total_none
        if total > 0 and (start // CHUNK_SIZE) % 10 == 0:
            write_markdown(per_gid_stats, total_full, total_part, total_none, total)

        pbar.set_postfix({
            "FULL%": f"{100*total_full/(total_full+total_part+total_none):.2f}",
            "n": total_full + total_part + total_none,
        })

    results_f.close()

    # Final write
    total = total_full + total_part + total_none
    write_markdown(per_gid_stats, total_full, total_part, total_none, total)

    print(f"\n[DONE] Total {total:,} samples classified")
    print(f"  FULL:    {total_full:,} ({100*total_full/total:.2f}%)")
    print(f"  PARTIAL: {total_part:,} ({100*total_part/total:.2f}%)")
    print(f"  NONE:    {total_none:,} ({100*total_none/total:.2f}%)")
    print(f"  Output: {MD_FILE}")


if __name__ == "__main__":
    main()
