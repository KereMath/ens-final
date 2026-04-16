"""
Balanced cache uzerinde comprehensive threshold + blend sweep.
"""
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np

from config import ANOM_LABELS, BASE_LABELS, GROUP_EXPECTED, GROUP_NAMES, META_MODELS_DIR, PROCESSED_DIR
from fast_grid import match_type

warnings.filterwarnings("ignore")


def main():
    cache = dict(np.load(PROCESSED_DIR / "balanced_eval_cache.npz"))
    print(f"  Cache: {len(cache['gid']):,} sample")
    unique, counts = np.unique(cache["gid"], return_counts=True)
    print(f"  Grup dagilimi: min={counts.min()}, max={counts.max()}, eq={(counts == counts[0]).all()}")

    base_meta = joblib.load(META_MODELS_DIR / "base_meta.pkl")
    anom_metas = {a: joblib.load(META_MODELS_DIR / f"anom_{a}.pkl") for a in ANOM_LABELS
                   if (META_MODELS_DIR / f"anom_{a}.pkl").exists()}
    router = joblib.load(META_MODELS_DIR / "router.pkl")

    stat_probs = cache["stat_v2"]
    meta_X = cache["meta_X"]
    gids = cache["gid"]

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

    def run(stat_t, rt=0.30, ct=0.0, blend_params=None):
        if blend_params is None:
            blend_params = blend
        pg = defaultdict(lambda: {"full": 0, "partial": 0, "none": 0})
        for i in range(len(gids)):
            gid = int(gids[i])
            expected = GROUP_EXPECTED[gid]
            base_type = BASE_LABELS[base_pred[i]]
            if stat_probs[i] >= stat_t:
                pb, pa = "stationary", []
            elif router_p[i] < rt:
                pb, pa = base_type, []
            else:
                anomalies = []
                for j, anom in enumerate(ANOM_LABELS):
                    p = blend_params.get(anom, {"alpha": 1.0, "threshold": 0.5})
                    meta_pp = anom_probs[anom][i]
                    new_pp = float(raw_new_anom[i, j])
                    blended = p["alpha"] * meta_pp + (1 - p["alpha"]) * new_pp
                    eff_t = min(p["threshold"], ct) if base_type == "stationary" else p["threshold"]
                    if blended >= eff_t:
                        anomalies.append(anom)
                pb, pa = base_type, anomalies
            mt = match_type(pb, pa, expected)
            pg[gid][mt.lower()] += 1
        return pg

    total_samples = len(gids)
    n_groups = len(unique)

    print(f"\n{'='*90}")
    print(f"  BALANCED EVAL THRESHOLD SWEEP ({total_samples:,} sample, {n_groups} grup)")
    print(f"{'='*90}")
    print(f"  {'stat_t':>7}  {'FULL':>6}  {'pct':>6}  {'Macro%':>7}  "
          f"{'G1%':>5} {'G5%':>5} {'G7%':>5} {'G8%':>5} {'G10%':>5}")

    best_w = (0, None)
    best_m = (0, None)
    for st in np.arange(0.30, 1.01, 0.05):
        pg = run(float(st))
        full = sum(v["full"] for v in pg.values())
        total = sum(sum(v.values()) for v in pg.values())
        pcts = [100 * v["full"] / (v["full"] + v["partial"] + v["none"])
                for v in pg.values() if (v["full"] + v["partial"] + v["none"]) > 0]
        macro = sum(pcts) / len(pcts)

        def _pct(g):
            v = pg.get(g, {})
            n = v.get("full", 0) + v.get("partial", 0) + v.get("none", 0)
            return 100 * v.get("full", 0) / n if n else 0

        m_w = " W*" if full > best_w[0] else ""
        m_m = " M*" if macro > best_m[0] else ""
        if full > best_w[0]:
            best_w = (full, float(st))
        if macro > best_m[0]:
            best_m = (macro, float(st))

        print(f"  {st:>7.2f}  {full:>6}  {100*full/total:>5.2f}%  {macro:>6.2f}%  "
              f"{_pct(1):>4.1f} {_pct(5):>4.1f} {_pct(7):>4.1f} {_pct(8):>4.1f} {_pct(10):>4.1f}{m_w}{m_m}")

    print(f"\n  BEST WEIGHTED: stat_t={best_w[1]:.2f}  FULL={best_w[0]}")
    print(f"  BEST MACRO:    stat_t={best_m[1]:.2f}  MACRO={best_m[0]:.2f}%")


if __name__ == "__main__":
    main()
