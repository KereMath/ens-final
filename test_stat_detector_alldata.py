"""
Stationary detector v2 (custom 25 feature) - tum 387K stationary CSV uzerinde test.
Tek soru: kac sample 'stationary' (P >= 0.5) tespit edilecek?
"""
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from config import GROUP_PATHS, MIN_SERIES_LENGTH, RESULTS_DIR
from processor import get_leaf_csvs, read_series
from stat_detector import (
    extract_feature_vector, load_stationary_detector,
)

warnings.filterwarnings("ignore")

CHUNK_SIZE = 5000
SAMPLE_FRACTION = 0.10  # her leaf'in %10'u


def main():
    print("=" * 70)
    print("  STATIONARY DETECTOR v2 — ALL 387K STATIONARY CSVs TEST")
    print("=" * 70)

    # 1. Load detector
    model, scaler, selector = load_stationary_detector()
    expected_features = scaler.n_features_in_
    print(f"  Expected features: {expected_features}")

    # 2. Collect all stationary CSVs
    print("\n[1] Collecting stationary CSVs...")
    leaf_map = {}
    for root in GROUP_PATHS[1]:  # Group 1 = stationary
        if root.exists():
            for leaf, csvs in get_leaf_csvs(root).items():
                leaf_map.setdefault(leaf, []).extend(csvs)

    import random
    random.seed(42)
    all_csvs = []
    leaf_counts = {}
    total_avail = 0
    for leaf in sorted(leaf_map.keys()):
        valid = sorted([c for c in leaf_map[leaf] if c.name != "metadata.csv"])
        total_avail += len(valid)
        n_sample = max(1, int(len(valid) * SAMPLE_FRACTION))
        sampled = random.sample(valid, min(n_sample, len(valid)))
        all_csvs.extend([(c, leaf.name) for c in sampled])
        leaf_counts[leaf.name] = len(sampled)

    print(f"  Available: {total_avail:,}  Sampled: {len(all_csvs):,} ({100*len(all_csvs)/total_avail:.1f}%) from {len(leaf_map)} leaves")

    # 3. Process in chunks
    print(f"\n[2] Processing in chunks of {CHUNK_SIZE}...")
    leaf_stats = defaultdict(lambda: {"total": 0, "stat": 0,
                                       "p_stat_sum": 0.0, "p_stat_high": 0})
    total_stat = 0
    total_processed = 0
    p_stat_all = []

    pbar = tqdm(range(0, len(all_csvs), CHUNK_SIZE), desc="Chunks")
    for start in pbar:
        end = min(start + CHUNK_SIZE, len(all_csvs))
        chunk = all_csvs[start:end]

        # Read + extract features
        feature_matrix = []
        valid_leaves = []
        for csv_path, leaf_name in chunk:
            data = read_series(csv_path)
            if len(data) < MIN_SERIES_LENGTH:
                continue
            fv = extract_feature_vector(data)
            if fv is None:
                continue
            if len(fv) != expected_features:
                if len(fv) < expected_features:
                    fv = np.pad(fv, (0, expected_features - len(fv)), 'constant')
                else:
                    fv = fv[:expected_features]
            feature_matrix.append(fv)
            valid_leaves.append(leaf_name)

        if not feature_matrix:
            continue

        # Predict
        X = np.array(feature_matrix)
        X_scaled = scaler.transform(X)
        if selector is not None:
            X_scaled = selector.transform(X_scaled)
        probs = model.predict_proba(X_scaled)
        p_stat = probs[:, 0]  # P(stationary) = class 0

        for leaf_name, p in zip(valid_leaves, p_stat):
            leaf_stats[leaf_name]["total"] += 1
            leaf_stats[leaf_name]["p_stat_sum"] += float(p)
            if p >= 0.5:
                leaf_stats[leaf_name]["stat"] += 1
                total_stat += 1
            if p >= 0.9:
                leaf_stats[leaf_name]["p_stat_high"] += 1
            p_stat_all.append(float(p))
            total_processed += 1

        pbar.set_postfix({
            "stat%": f"{100*total_stat/total_processed:.2f}",
            "n": total_processed,
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"  STATIONARY DETECTOR — ALL DATA RESULTS")
    print(f"{'='*70}")
    p_arr = np.array(p_stat_all)
    print(f"  Total processed:        {total_processed:,}")
    print(f"  Predicted STATIONARY (P>=0.5): {total_stat:,} ({100*total_stat/total_processed:.2f}%)")
    print(f"  Predicted NON-STAT  (P<0.5):   {total_processed-total_stat:,} "
          f"({100*(total_processed-total_stat)/total_processed:.2f}%)")
    print(f"\n  P(stationary) distribution:")
    print(f"    Mean:         {p_arr.mean():.4f}")
    print(f"    Median:       {np.median(p_arr):.4f}")
    print(f"    > 0.50:       {(p_arr >= 0.5).sum():,} ({100*(p_arr >= 0.5).sum()/len(p_arr):.2f}%)")
    print(f"    > 0.70:       {(p_arr >= 0.7).sum():,} ({100*(p_arr >= 0.7).sum()/len(p_arr):.2f}%)")
    print(f"    > 0.90:       {(p_arr >= 0.9).sum():,} ({100*(p_arr >= 0.9).sum()/len(p_arr):.2f}%)")
    print(f"    > 0.95:       {(p_arr >= 0.95).sum():,} ({100*(p_arr >= 0.95).sum()/len(p_arr):.2f}%)")

    # Per-leaf
    print(f"\n  PER-LEAF (top 30 by sample count):")
    print(f"  {'Leaf':<40} {'n':>8}  {'stat%':>7}  {'p_mean':>7}  {'>0.9%':>6}")
    sorted_leaves = sorted(leaf_stats.items(), key=lambda x: -x[1]["total"])
    for leaf_name, s in sorted_leaves[:30]:
        n = s["total"]
        pct = 100 * s["stat"] / n if n else 0
        p_mean = s["p_stat_sum"] / n if n else 0
        high_pct = 100 * s["p_stat_high"] / n if n else 0
        print(f"  {leaf_name[:40]:<40} {n:>8,}  {pct:>6.2f}%  {p_mean:>7.4f}  {high_pct:>5.2f}%")

    # Save full per-leaf
    md_lines = []
    md_lines.append("# Stationary Detector v2 — All 387K Stationary Test\n")
    md_lines.append("**Custom 25-feature XGBoost binary detector**\n")
    md_lines.append("Tüm `stationary/` klasörü altındaki 387,000 CSV üzerinde test.\n")
    md_lines.append("## Özet\n")
    md_lines.append(f"| Metrik | Değer |")
    md_lines.append(f"|---|---|")
    md_lines.append(f"| Total processed | {total_processed:,} |")
    md_lines.append(f"| Predicted **stationary** (P>=0.5) | {total_stat:,} ({100*total_stat/total_processed:.2f}%) |")
    md_lines.append(f"| Predicted non-stationary | {total_processed-total_stat:,} ({100*(total_processed-total_stat)/total_processed:.2f}%) |")
    md_lines.append(f"| P(stat) Mean | {p_arr.mean():.4f} |")
    md_lines.append(f"| P(stat) Median | {np.median(p_arr):.4f} |")
    md_lines.append(f"| P(stat) > 0.7 | {(p_arr >= 0.7).sum():,} ({100*(p_arr >= 0.7).sum()/len(p_arr):.2f}%) |")
    md_lines.append(f"| P(stat) > 0.9 | {(p_arr >= 0.9).sum():,} ({100*(p_arr >= 0.9).sum()/len(p_arr):.2f}%) |")
    md_lines.append(f"| P(stat) > 0.95 | {(p_arr >= 0.95).sum():,} ({100*(p_arr >= 0.95).sum()/len(p_arr):.2f}%) |")
    md_lines.append("\n## Per-Leaf Results\n")
    md_lines.append("| Leaf | n | Predicted stat% | P(stat) mean | P>0.9 % |")
    md_lines.append("|---|---|---|---|---|")
    for leaf_name, s in sorted_leaves:
        n = s["total"]
        pct = 100 * s["stat"] / n if n else 0
        p_mean = s["p_stat_sum"] / n if n else 0
        high_pct = 100 * s["p_stat_high"] / n if n else 0
        md_lines.append(f"| `{leaf_name}` | {n:,} | {pct:.2f}% | {p_mean:.4f} | {high_pct:.2f}% |")

    out_md = RESULTS_DIR / "stat_detector_alldata.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\nOutput: {out_md}")


if __name__ == "__main__":
    main()
