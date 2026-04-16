"""
BASIT pipeline — meta-learner olmadan:
1. Stat detector @ 0.5 → stationary
2. Else new ensemble direkt: base=argmax(4 base models), anomaly=P>=0.5

Meta-learner ve complex stacking YOK. Sadece temiz mantık.
"""
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np

from config import (
    ANOM_LABELS, BASE_LABELS, GROUP_EXPECTED, GROUP_NAMES, PROCESSED_DIR,
)
from fast_grid import match_type

warnings.filterwarnings("ignore")


def main():
    cache = dict(np.load(PROCESSED_DIR / "eval_cache.npz"))
    stat_probs = cache["stat_v2"]
    raw_new = cache["raw_new"]  # (N, 10): 4 base + 6 anomaly

    base_probs = raw_new[:, :4]   # stationary, det_trend, stoch_trend, volatility
    anom_probs = raw_new[:, 4:]   # 6 anomaly probs

    print("=" * 70)
    print("  BASIT PIPELINE — Meta-learner YOK")
    print("=" * 70)

    # Strateji 1: Stat@X override + new ensemble direkt
    print("\n[Strateji 1] stat>=X override, else new_ensemble direkt (threshold 0.5)")
    print(f"  {'stat_t':>7}  {'anom_t':>7}  {'FULL':>5}  {'pct':>6}  "
          f"{'G1':>7} {'G5':>7} {'G7':>7} {'G8':>7} {'G10':>7}")

    for stat_t in [0.50, 0.70, 0.85, 0.90, 0.92, 0.95]:
        for anom_t in [0.50, 0.55, 0.60, 0.65, 0.70]:
            results_per_gid = defaultdict(lambda: {"full": 0, "partial": 0, "none": 0})
            for i in range(len(cache["gid"])):
                gid = int(cache["gid"][i])
                expected = GROUP_EXPECTED[gid]

                # Stat override
                if stat_probs[i] >= stat_t:
                    pb, pa = "stationary", []
                else:
                    # New ensemble direkt
                    base_idx = int(np.argmax(base_probs[i]))
                    pb = BASE_LABELS[base_idx]
                    pa = [ANOM_LABELS[j] for j in range(6) if anom_probs[i, j] >= anom_t]

                mt = match_type(pb, pa, expected)
                results_per_gid[gid][mt.lower()] += 1

            full = sum(v["full"] for v in results_per_gid.values())
            total = sum(sum(v.values()) for v in results_per_gid.values())
            g1 = results_per_gid.get(1, {}).get("full", 0)
            g5 = results_per_gid.get(5, {}).get("full", 0)
            g7 = results_per_gid.get(7, {}).get("full", 0)
            g8 = results_per_gid.get(8, {}).get("full", 0)
            g10 = results_per_gid.get(10, {}).get("full", 0)
            print(f"  {stat_t:>7.2f}  {anom_t:>7.2f}  {full:>5}  "
                  f"{100*full/total:>5.2f}%  {g1:>3}/120 {g5:>3}/480 "
                  f"{g7:>3}/480 {g8:>3}/480 {g10:>3}/480")

    # Strateji 2: Context threshold benzeri (stationary base'de anom esigi 0)
    print("\n[Strateji 2] + context threshold (base=stationary -> anom_t=0)")
    best = (0, None, None)
    for stat_t in [0.50, 0.70, 0.90, 0.92, 0.95]:
        for anom_t in [0.50, 0.60, 0.70]:
            results_per_gid = defaultdict(lambda: {"full": 0, "partial": 0, "none": 0})
            for i in range(len(cache["gid"])):
                gid = int(cache["gid"][i])
                expected = GROUP_EXPECTED[gid]

                if stat_probs[i] >= stat_t:
                    pb, pa = "stationary", []
                else:
                    base_idx = int(np.argmax(base_probs[i]))
                    pb = BASE_LABELS[base_idx]
                    eff_t = 0.0 if pb == "stationary" else anom_t
                    pa = [ANOM_LABELS[j] for j in range(6) if anom_probs[i, j] >= eff_t]

                mt = match_type(pb, pa, expected)
                results_per_gid[gid][mt.lower()] += 1

            full = sum(v["full"] for v in results_per_gid.values())
            total = sum(sum(v.values()) for v in results_per_gid.values())
            g1 = results_per_gid.get(1, {}).get("full", 0)
            marker = ""
            if full > best[0]:
                best = (full, stat_t, anom_t)
                marker = " ***"
            print(f"  stat>={stat_t:.2f} anom>={anom_t:.2f}  FULL={full} ({100*full/total:.2f}%)  G1={g1}/120{marker}")

    print(f"\n  BEST SIMPLE: stat>={best[1]} anom>={best[2]}  FULL={best[0]}")
    print(f"  vs v12 meta-learner: 3919 (89.07%)")
    print(f"  Delta: {best[0] - 3919:+d}")


if __name__ == "__main__":
    main()
