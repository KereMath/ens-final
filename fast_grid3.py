"""
Fast Grid Search v3 — Per-anomaly alpha + threshold joint tuning
"""
import warnings

import joblib
import numpy as np

from config import ANOM_LABELS, META_MODELS_DIR, PROCESSED_DIR
from fast_grid import evaluate_strategy, load_cache

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    cache = load_cache()
    blend_params = joblib.load(META_MODELS_DIR / "blend_weights.pkl")

    # Starting from best so far
    base = {
        "stat_version": "v2",
        "stat_threshold": 0.92,
        "router_theta": 0.32,
        "context_thresh": 0.0,
        "context_base_types": ["stationary"],
        "blend_params": {
            "collective_anomaly":  {"alpha": 1.00, "threshold": 0.45},
            "contextual_anomaly":  {"alpha": 0.85, "threshold": 0.55},
            "mean_shift":          {"alpha": 0.95, "threshold": 0.48},
            "point_anomaly":       {"alpha": 1.00, "threshold": 0.65},
            "trend_shift":         {"alpha": 0.90, "threshold": 0.65},
            "variance_shift":      {"alpha": 1.00, "threshold": 0.58},
        },
    }

    base_r = evaluate_strategy(cache, base)
    print(f"Starting: FULL={base_r['full']} ({100*base_r['full']/base_r['total']:.2f}%)")

    # Per-anomaly alpha x threshold joint grid
    print("\n[1] PER-ANOMALY ALPHA x THRESHOLD JOINT")
    cumulative = dict(base)
    cum_blend = dict(base["blend_params"])

    for anom_name in ANOM_LABELS:
        best_r = None
        for alpha in [0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
            for t in np.arange(0.25, 0.75, 0.02):
                new_blend = dict(cum_blend)
                new_blend[anom_name] = {"alpha": alpha, "threshold": float(t)}
                s = dict(cumulative)
                s["blend_params"] = new_blend
                r = evaluate_strategy(cache, s)
                if best_r is None or r["full"] > best_r["full"]:
                    best_r = r
                    best_r["alpha"] = alpha
                    best_r["threshold"] = float(t)

        cum_blend[anom_name] = {"alpha": best_r["alpha"], "threshold": best_r["threshold"]}
        cumulative["blend_params"] = cum_blend
        print(f"  {anom_name:<25}  alpha={best_r['alpha']:.2f}  t={best_r['threshold']:.2f}  "
              f"FULL={best_r['full']} ({100*best_r['full']/best_r['total']:.2f}%)")

    final_r = evaluate_strategy(cache, cumulative)
    print(f"\n  FINAL: FULL={final_r['full']} ({100*final_r['full']/final_r['total']:.2f}%)")
    print(f"\n  Final blend_params:")
    for k, v in cum_blend.items():
        print(f"    {k}: {v}")

    # ===== Ultra-fine stat threshold with new blend =====
    print("\n[2] ULTRA-FINE STAT THRESHOLD")
    stat_results = []
    for v in ["v1", "v2", "v3"]:
        for t in np.arange(0.85, 1.01, 0.01):
            s = dict(cumulative)
            s["stat_version"] = v
            s["stat_threshold"] = float(t)
            r = evaluate_strategy(cache, s)
            r["strategy"] = f"stat={v}@{t:.2f}"
            stat_results.append(r)

    stat_results.sort(key=lambda x: -x["full"])
    print("  Top 10:")
    for i, r in enumerate(stat_results[:10]):
        print(f"  #{i+1}  {r['strategy']:<30}  FULL={r['full']} ({100*r['full']/r['total']:.2f}%)")

    # Apply best stat
    best_stat_strategy = stat_results[0]["strategy"]
    bsv = best_stat_strategy.split("=")[1].split("@")[0]
    bst = float(best_stat_strategy.split("@")[1])

    # ===== Ultra-fine router theta =====
    print("\n[3] ULTRA-FINE ROUTER THETA (with best stat + blend)")
    rt_results = []
    for rt in np.arange(0.25, 0.50, 0.01):
        s = dict(cumulative)
        s["stat_version"] = bsv
        s["stat_threshold"] = bst
        s["router_theta"] = float(rt)
        r = evaluate_strategy(cache, s)
        r["strategy"] = f"RT={rt:.2f}"
        rt_results.append(r)

    rt_results.sort(key=lambda x: -x["full"])
    print("  Top 10:")
    for i, r in enumerate(rt_results[:10]):
        print(f"  #{i+1}  {r['strategy']:<30}  FULL={r['full']} ({100*r['full']/r['total']:.2f}%)")

    print("\n" + "="*70)
    overall_best = rt_results[0]
    print(f"  BEST OVERALL: stat={bsv}@{bst}  {overall_best['strategy']}")
    print(f"  FULL={overall_best['full']}  ({100*overall_best['full']/overall_best['total']:.2f}%)")
    print(f"  PARTIAL={overall_best['partial']}  NONE={overall_best['none']}")

    # Save final config
    final_config = {
        "stat_version": bsv,
        "stat_threshold": bst,
        "router_theta": float(overall_best['strategy'].split("=")[1]),
        "context_thresh": 0.0,
        "blend_params": cum_blend,
    }
    import json
    with open(PROCESSED_DIR / "best_config.json", "w") as f:
        json.dump({k: v if not isinstance(v, dict) else v for k, v in final_config.items()}, f, indent=2, default=str)
    print(f"\n  Saved: processed_data/best_config.json")
