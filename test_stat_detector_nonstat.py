"""
Stat detector experiment: 38 non-stationary class'tan 1000'er sample -> P(stationary) dağılımı.
Hangi class detector'i kandırıyor? Hangisi temiz?

Cikti: results/stat_detector_nonstat_test.md
"""
import random
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from config import (
    GROUP_EXPECTED, GROUP_NAMES, GROUP_PATHS,
    MIN_SERIES_LENGTH, RESULTS_DIR, SOURCE_GROUPS,
)
from processor import get_leaf_csvs, read_series
from stat_detector import (
    extract_feature_vector, load_stationary_detector,
)

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

N_PER_GROUP = 1000
CHUNK_SIZE = 2000


def main():
    print("=" * 70)
    print("  STATIONARY DETECTOR — 38 NON-STATIONARY CLASS TEST")
    print("  1000 sample / class -> P(stationary) per class")
    print("=" * 70)

    model, scaler, selector = load_stationary_detector()
    expected_features = scaler.n_features_in_

    # Sampling: 1000 from each non-stationary group (2-39)
    print(f"\n[1] Sampling {N_PER_GROUP} per group...")
    all_csvs = []
    for gid in range(2, 40):
        paths = GROUP_PATHS[gid]
        leaf_map = {}
        for root in paths:
            if root.exists():
                for leaf, csvs in get_leaf_csvs(root).items():
                    leaf_map.setdefault(leaf, []).extend(csvs)
        valid = []
        for leaf in sorted(leaf_map.keys()):
            valid.extend([c for c in leaf_map[leaf] if c.name != "metadata.csv"])
        n_take = min(N_PER_GROUP, len(valid))
        sampled = random.sample(valid, n_take)
        for c in sampled:
            all_csvs.append((c, gid))
        print(f"  Grup {gid:2d} {GROUP_NAMES[gid][:30]:<30}  {n_take} sample (avail: {len(valid):,})")

    print(f"\n  Total: {len(all_csvs):,}")

    # Process
    print(f"\n[2] Processing {len(all_csvs):,} samples...")
    per_group_stats = defaultdict(lambda: {"n": 0, "p_sum": 0.0,
                                            "stat_05": 0, "stat_07": 0,
                                            "stat_09": 0, "stat_092": 0, "stat_095": 0})

    for start in tqdm(range(0, len(all_csvs), CHUNK_SIZE), desc="Chunks"):
        end = min(start + CHUNK_SIZE, len(all_csvs))
        chunk = all_csvs[start:end]
        feature_matrix = []
        valid_gids = []
        for csv_path, gid in chunk:
            d = read_series(csv_path)
            if len(d) < MIN_SERIES_LENGTH:
                continue
            fv = extract_feature_vector(d)
            if fv is None:
                continue
            if len(fv) != expected_features:
                if len(fv) < expected_features:
                    fv = np.pad(fv, (0, expected_features - len(fv)), 'constant')
                else:
                    fv = fv[:expected_features]
            feature_matrix.append(fv)
            valid_gids.append(gid)

        if not feature_matrix:
            continue

        X = np.array(feature_matrix)
        X_scaled = scaler.transform(X)
        if selector is not None:
            X_scaled = selector.transform(X_scaled)
        probs = model.predict_proba(X_scaled)
        p_stat = probs[:, 0]

        for gid, p in zip(valid_gids, p_stat):
            s = per_group_stats[gid]
            s["n"] += 1
            s["p_sum"] += float(p)
            if p >= 0.5: s["stat_05"] += 1
            if p >= 0.7: s["stat_07"] += 1
            if p >= 0.9: s["stat_09"] += 1
            if p >= 0.92: s["stat_092"] += 1
            if p >= 0.95: s["stat_095"] += 1

    # Report
    print(f"\n{'='*100}")
    print(f"  PER-CLASS RESULTS (FP rate = stat detector tarafından stationary diye işaretleme oranı)")
    print(f"{'='*100}")
    print(f"  {'#':>3} {'Grup':<28} {'Beklenen Anomali':<25} {'n':>5}  "
          f"{'P_mean':>7}  {'>0.5%':>6} {'>0.7%':>6} {'>0.9%':>6} {'>0.92%':>7} {'>0.95%':>7}")
    print(f"  {'-'*3} {'-'*28} {'-'*25} {'-'*5}  {'-'*7}  {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")

    for gid in sorted(per_group_stats.keys()):
        s = per_group_stats[gid]
        n = s["n"]
        if n == 0:
            continue
        exp = GROUP_EXPECTED[gid]
        anom_str = " + ".join(exp["anomalies"]) if exp["anomalies"] else exp["base"]
        print(f"  {gid:>3} {GROUP_NAMES[gid][:28]:<28} {anom_str[:25]:<25} {n:>5}  "
              f"{s['p_sum']/n:>7.4f}  "
              f"{100*s['stat_05']/n:>5.2f}% {100*s['stat_07']/n:>5.2f}% "
              f"{100*s['stat_09']/n:>5.2f}% {100*s['stat_092']/n:>6.2f}% "
              f"{100*s['stat_095']/n:>6.2f}%")

    # Markdown report
    md = []
    md.append("# Stationary Detector — Non-Stationary Class FP Test\n")
    md.append("**38 non-stationary class** her birinden 1000 sample -> "
              "stat detector P(stationary) dağılımı.\n")
    md.append("Yüksek değer = detector yanlış 'stationary' diyor (FP).\n")
    md.append("Düşük değer = detector doğru 'non-stationary' diyor.\n")
    md.append("\n## Sonuçlar\n")
    md.append("| # | Grup | Beklenen | n | P_mean | P>0.5 % | P>0.7 % | P>0.9 % | **P>0.92 %** | P>0.95 % |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for gid in sorted(per_group_stats.keys()):
        s = per_group_stats[gid]
        n = s["n"]
        if n == 0:
            continue
        exp = GROUP_EXPECTED[gid]
        anom_str = " + ".join(exp["anomalies"]) if exp["anomalies"] else exp["base"]
        md.append(f"| {gid} | {GROUP_NAMES[gid]} | `{anom_str}` | {n} | "
                  f"{s['p_sum']/n:.4f} | "
                  f"{100*s['stat_05']/n:.2f}% | {100*s['stat_07']/n:.2f}% | "
                  f"{100*s['stat_09']/n:.2f}% | **{100*s['stat_092']/n:.2f}%** | "
                  f"{100*s['stat_095']/n:.2f}% |")

    out = RESULTS_DIR / "stat_detector_nonstat_test.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
