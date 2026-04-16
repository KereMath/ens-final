"""
Akilli stat override testi:
Pipeline'da stat_prob >= X VE max_meta_anom < Y ise override → stationary.
Bu Group 1'i kurtarirken Group 5-10'u korur.
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

    # Precompute
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

    # Max anomali meta prob per sample
    all_anom_mat = np.stack([anom_probs[a] for a in ANOM_LABELS], axis=1)  # (N, 6)
    max_anom_per_sample = all_anom_mat.max(axis=1)

    raw_new_anom = cache["raw_new"][:, 4:]

    blend = {
        "collective_anomaly":  {"alpha": 0.85, "threshold": 0.73},
        "contextual_anomaly":  {"alpha": 0.70, "threshold": 0.69},
        "mean_shift":          {"alpha": 0.90, "threshold": 0.49},
        "point_anomaly":       {"alpha": 0.70, "threshold": 0.69},
        "trend_shift":         {"alpha": 0.90, "threshold": 0.73},
        "variance_shift":      {"alpha": 0.70, "threshold": 0.69},
    }

    def run(stat_t_primary, stat_t_fallback, max_anom_cutoff):
        """
        stat_t_primary: stat_prob >= bu → check max_anom
        max_anom_cutoff: max meta anom < bu ise override
        stat_t_fallback: yuksek threshold (ek override)
        """
        results_per_gid = defaultdict(lambda: {"full": 0, "partial": 0, "none": 0})
        for i in range(len(cache["gid"])):
            gid = int(cache["gid"][i])
            expected = GROUP_EXPECTED[gid]
            base_type = BASE_LABELS[base_pred[i]]

            # Yeni akilli kural
            overridden = False
            if stat_probs[i] >= stat_t_fallback:
                overridden = True  # cok emin: override
            elif stat_probs[i] >= stat_t_primary and max_anom_per_sample[i] < max_anom_cutoff:
                overridden = True  # stat high + no anomali → override

            if overridden:
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
        total = sum(sum(v.values()) for v in results_per_gid.values())
        full = sum(v["full"] for v in results_per_gid.values())
        return full, total, results_per_gid

    # Baseline v12
    f_base, _, base_gid = run(1.01, 0.92, 0.0)  # sadece stat >= 0.92
    g1_base = base_gid.get(1, {}).get("full", 0)
    print(f"BASELINE v12 (stat>=0.92): FULL={f_base}  G1={g1_base}/120")

    print("\nSMART HYBRID: stat_prob >= X AND max_meta_anom < Y -> override")
    print(f"  {'stat_t':>7} {'anom<':>6}  {'FULL':>5}  {'pct':>6}  "
          f"{'G1':>7} {'G5':>7} {'G7':>7} {'G8':>7} {'G10':>7}")

    best = (f_base, None, None)
    for stat_t in [0.50, 0.60, 0.70, 0.80, 0.85, 0.88, 0.90]:
        for anom_t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            f, _, per_gid = run(stat_t, 0.92, anom_t)
            g1 = per_gid.get(1, {}).get("full", 0)
            g5 = per_gid.get(5, {}).get("full", 0)
            g7 = per_gid.get(7, {}).get("full", 0)
            g8 = per_gid.get(8, {}).get("full", 0)
            g10 = per_gid.get(10, {}).get("full", 0)
            marker = ""
            if f > best[0]:
                best = (f, stat_t, anom_t)
                marker = " ***"
            if f >= f_base:
                print(f"  {stat_t:>7.2f} {anom_t:>6.2f}   {f:>5}  "
                      f"{100*f/4400:>5.2f}%  {g1:>3}/120 {g5:>3}/480 "
                      f"{g7:>3}/480 {g8:>3}/480 {g10:>3}/480{marker}")

    print(f"\n  BEST: stat_t={best[1]} anom<{best[2]} FULL={best[0]} (delta vs v12: {best[0]-f_base:+d})")

    if best[1] is not None:
        _, _, best_gid = run(best[1], 0.92, best[2])
        print("\n  PER-GROUP BEST:")
        for gid in sorted(best_gid.keys()):
            r = best_gid[gid]
            tot = r["full"] + r["partial"] + r["none"]
            pct = 100 * r["full"] / tot if tot else 0
            old = base_gid.get(gid, {})
            old_pct = 100 * old.get("full", 0) / tot if tot else 0
            delta = r["full"] - old.get("full", 0)
            mark = " ↑" if delta > 0 else (" ↓" if delta < 0 else "")
            print(f"  Grp {gid:2d} ({GROUP_NAMES[gid][:25]:<25}) "
                  f"FULL={r['full']:3d}/{tot:<3d} ({pct:.1f}%)  [was {old.get('full', 0)}/{old_pct:.1f}%{mark}]")


if __name__ == "__main__":
    main()
