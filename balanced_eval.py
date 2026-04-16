"""
Balanced evaluation: her gruptan 10 sample = 390 sample total.
Mevcut 4400 eval'dan random 10 sample/grup ornekle.

Tum pipeline params (stat_t, blend, vs) bu balanced eval'da test et.
"""
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np

from config import (
    ANOM_LABELS, BASE_LABELS, GROUP_EXPECTED, GROUP_NAMES,
    META_MODELS_DIR, PROCESSED_DIR, RANDOM_STATE,
)
from fast_grid import match_type

warnings.filterwarnings("ignore")


def main():
    cache = dict(np.load(PROCESSED_DIR / "eval_cache.npz"))
    gids = cache["gid"]

    # Her gruptan 10 sample al - BALANCED
    rng = np.random.RandomState(RANDOM_STATE)
    balanced_idx = []
    for gid in sorted(set(gids)):
        gid_idx = np.where(gids == gid)[0]
        n_take = min(10, len(gid_idx))
        chosen = rng.choice(gid_idx, size=n_take, replace=False)
        balanced_idx.extend(chosen.tolist())
    balanced_idx = np.array(balanced_idx)
    print(f"Balanced eval: {len(balanced_idx)} sample (10/grup x 39 grup)")

    # Subset cache
    stat_probs = cache["stat_v2"][balanced_idx]
    meta_X = cache["meta_X"][balanced_idx]
    gids_eval = gids[balanced_idx]
    raw_new_anom = cache["raw_new"][balanced_idx, 4:]

    # Modeller
    base_meta = joblib.load(META_MODELS_DIR / "base_meta.pkl")
    anom_metas = {a: joblib.load(META_MODELS_DIR / f"anom_{a}.pkl") for a in ANOM_LABELS
                   if (META_MODELS_DIR / f"anom_{a}.pkl").exists()}
    router = joblib.load(META_MODELS_DIR / "router.pkl")

    # Inference
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

    blend = {
        "collective_anomaly":  {"alpha": 0.85, "threshold": 0.73},
        "contextual_anomaly":  {"alpha": 0.70, "threshold": 0.69},
        "mean_shift":          {"alpha": 0.90, "threshold": 0.49},
        "point_anomaly":       {"alpha": 0.70, "threshold": 0.69},
        "trend_shift":         {"alpha": 0.90, "threshold": 0.73},
        "variance_shift":      {"alpha": 0.70, "threshold": 0.69},
    }

    def run(stat_t):
        pg = defaultdict(lambda: {"full": 0, "partial": 0, "none": 0})
        for i in range(len(gids_eval)):
            gid = int(gids_eval[i])
            expected = GROUP_EXPECTED[gid]
            base_type = BASE_LABELS[base_pred[i]]
            if stat_probs[i] >= stat_t:
                pb, pa = "stationary", []
            elif router_p[i] < 0.30:
                pb, pa = base_type, []
            else:
                anomalies = []
                for j, anom in enumerate(ANOM_LABELS):
                    p = blend.get(anom, {"alpha": 1.0, "threshold": 0.5})
                    meta_pp = anom_probs[anom][i]
                    new_pp = float(raw_new_anom[i, j])
                    blended = p["alpha"] * meta_pp + (1 - p["alpha"]) * new_pp
                    eff_t = min(p["threshold"], 0.0) if base_type == "stationary" else p["threshold"]
                    if blended >= eff_t:
                        anomalies.append(anom)
                pb, pa = base_type, anomalies
            mt = match_type(pb, pa, expected)
            pg[gid][mt.lower()] += 1
        return pg

    print("\n" + "=" * 100)
    print("  BALANCED EVAL (390 sample, 10/grup) — STAT THRESHOLD SWEEP")
    print("=" * 100)
    print(f"  {'stat_t':>7}  {'FULL':>4}  {'pct':>6}  "
          f"{'G1':>7} {'G5':>7} {'G7':>7} {'G8':>7} {'G10':>7}")

    best = (0, None)
    for st in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98]:
        pg = run(st)
        full = sum(v["full"] for v in pg.values())
        total = sum(sum(v.values()) for v in pg.values())
        g1 = pg.get(1, {}).get("full", 0)
        g5 = pg.get(5, {}).get("full", 0)
        g7 = pg.get(7, {}).get("full", 0)
        g8 = pg.get(8, {}).get("full", 0)
        g10 = pg.get(10, {}).get("full", 0)
        marker = ""
        if full > best[0]:
            best = (full, st)
            marker = " ***"
        print(f"  {st:>7.2f}  {full:>4}  {100*full/total:>5.2f}%  "
              f"{g1:>3}/10  {g5:>3}/10  {g7:>3}/10  {g8:>3}/10  {g10:>3}/10{marker}")

    print(f"\n  BALANCED BEST: stat_t={best[1]} FULL={best[0]}/390 ({100*best[0]/390:.2f}%)")

    # PER-GROUP detayli sweep
    print("\n\nPER-GROUP RESULT at different stat_t (10 sample each, max 10):")
    for st in [0.50, 0.70, 0.85, 0.92]:
        pg = run(st)
        print(f"\n  stat_t={st:.2f}:")
        for gid in sorted(pg.keys()):
            r = pg[gid]
            tot = r["full"] + r["partial"] + r["none"]
            print(f"    {gid:>2} {GROUP_NAMES[gid][:25]:<25}: {r['full']}/{tot}")


if __name__ == "__main__":
    main()
