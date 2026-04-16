"""
Mevcut v12 pipeline + farkli stat_t degerleri.
Hic bir sey degistirmeden, sadece stat detector threshold'unu sweep et.
"""
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np

from config import (
    ANOM_LABELS, BASE_LABELS, GROUP_EXPECTED, GROUP_NAMES,
    META_MODELS_DIR, PROCESSED_DIR,
)
from fast_grid import match_type

warnings.filterwarnings("ignore")


def main():
    cache = dict(np.load(PROCESSED_DIR / "eval_cache.npz"))
    base_meta = joblib.load(META_MODELS_DIR / "base_meta.pkl")
    anom_metas = {a: joblib.load(META_MODELS_DIR / f"anom_{a}.pkl") for a in ANOM_LABELS
                   if (META_MODELS_DIR / f"anom_{a}.pkl").exists()}
    router = joblib.load(META_MODELS_DIR / "router.pkl")

    stat_probs = cache["stat_v2"]
    meta_X = cache["meta_X"]

    xgb_bp = base_meta["xgb"].predict_proba(meta_X)
    lgb_bp = base_meta["lgb"].predict_proba(meta_X)
    base_pred = np.argmax(0.5 * xgb_bp + 0.5 * lgb_bp, axis=1)
    xgb_r = router["xgb"].predict_proba(meta_X)[:, 1]
    lgb_r = router["lgb"].predict_proba(meta_X)[:, 1]
    router_p = 0.5 * xgb_r + 0.5 * lgb_r
    anom_probs = {}
    for a, m in anom_metas.items():
        xp = m["xgb"].predict_proba(meta_X)[:, 1]
        lp = m["lgb"].predict_proba(meta_X)[:, 1]
        anom_probs[a] = 0.5 * xp + 0.5 * lp
    raw_new_anom = cache["raw_new"][:, 4:]

    blend = {
        "collective_anomaly":  {"alpha": 0.85, "threshold": 0.73},
        "contextual_anomaly":  {"alpha": 0.70, "threshold": 0.69},
        "mean_shift":          {"alpha": 0.90, "threshold": 0.49},
        "point_anomaly":       {"alpha": 0.70, "threshold": 0.69},
        "trend_shift":         {"alpha": 0.90, "threshold": 0.73},
        "variance_shift":      {"alpha": 0.70, "threshold": 0.69},
    }

    def run(stat_t):
        results_per_gid = defaultdict(lambda: {"full": 0, "partial": 0, "none": 0})
        for i in range(len(cache["gid"])):
            gid = int(cache["gid"][i])
            expected = GROUP_EXPECTED[gid]
            base_type = BASE_LABELS[base_pred[i]]
            if stat_probs[i] >= stat_t:
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
        return results_per_gid

    print("=" * 100)
    print("  v12 PIPELINE + STAT THRESHOLD SWEEP")
    print("=" * 100)
    print(f"  {'stat_t':>7}  {'FULL':>5}  {'pct':>6}  "
          f"{'G1':>7} {'G5':>7} {'G7':>7} {'G8':>7} {'G10':>7}")

    best = (0, None)
    for stat_t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98]:
        per_gid = run(stat_t)
        full = sum(v["full"] for v in per_gid.values())
        total = sum(sum(v.values()) for v in per_gid.values())
        g1 = per_gid.get(1, {}).get("full", 0)
        g5 = per_gid.get(5, {}).get("full", 0)
        g7 = per_gid.get(7, {}).get("full", 0)
        g8 = per_gid.get(8, {}).get("full", 0)
        g10 = per_gid.get(10, {}).get("full", 0)
        marker = ""
        if full > best[0]:
            best = (full, stat_t)
            marker = " ***"
        print(f"  {stat_t:>7.2f}  {full:>5}  {100*full/total:>5.2f}%  "
              f"{g1:>3}/120 {g5:>3}/480 {g7:>3}/480 {g8:>3}/480 {g10:>3}/480{marker}")

    print(f"\n  BEST: stat_t={best[1]} FULL={best[0]}")


if __name__ == "__main__":
    main()
