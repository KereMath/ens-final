"""
Cached evaluation data ile super hizli grid search.
Tum stratejileri saniyeler icinde test eder.
"""
import json
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from config import (
    ANOM_LABELS, BASE_LABELS, GROUP_EXPECTED, GROUP_NAMES,
    META_MODELS_DIR, PROCESSED_DIR, RESULTS_DIR,
)

warnings.filterwarnings("ignore")


def load_cache():
    cache_path = PROCESSED_DIR / "eval_cache.npz"
    cache = dict(np.load(cache_path))
    print(f"  Cache yuklendi: {len(cache['gid'])} ornek")
    return cache


def match_type(pred_base, pred_anomalies, expected):
    true_base = expected["base"]
    true_anoms = expected["anomalies"]
    base_ok = (pred_base == true_base)
    if not true_anoms:
        if base_ok and len(pred_anomalies) == 0:
            return "FULL"
        elif base_ok:
            return "PARTIAL"
        else:
            return "NONE"
    else:
        all_anom = all(a in pred_anomalies for a in true_anoms)
        if base_ok and all_anom:
            return "FULL"
        elif base_ok or all_anom:
            return "PARTIAL"
        else:
            return "NONE"


def evaluate_strategy(cache, strategy):
    """
    strategy: dict with keys:
      - stat_version: v1-v6 or None
      - stat_threshold: float
      - router_theta: float
      - context_thresh: float (applied when combo + base=stationary)
      - context_base_types: list of base types to apply context to
      - blend_params: per-anomaly {alpha, threshold}
    """
    n = len(cache["gid"])
    results_per_gid = defaultdict(lambda: {"full": 0, "partial": 0, "none": 0})

    stat_v = strategy.get("stat_version")
    stat_t = strategy.get("stat_threshold", 1.01)
    rt = strategy.get("router_theta", 0.40)
    ct = strategy.get("context_thresh", 0.0)
    ct_bases = set(strategy.get("context_base_types", ["stationary"]))
    blend = strategy.get("blend_params", {})

    stat_probs = cache.get(f"stat_{stat_v}") if stat_v else None

    for i in range(n):
        gid = int(cache["gid"][i])
        expected = GROUP_EXPECTED[gid]

        # Base type
        base_idx = int(np.argmax(cache["base_proba"][i]))
        base_type = BASE_LABELS[base_idx]

        # Stat detector override
        if stat_probs is not None and stat_probs[i] >= stat_t:
            pred_base, pred_anoms = "stationary", []
        else:
            # Router
            p_combo = cache["router_proba"][i]
            if p_combo < rt:
                pred_base, pred_anoms = base_type, []
            else:
                # Combo branch
                new_anom_p = cache["raw_new"][i, 4:]
                context_lower = (base_type in ct_bases)
                anomalies = []
                for ai, anom_name in enumerate(ANOM_LABELS):
                    params = blend.get(anom_name, {"alpha": 1.0, "threshold": 0.5})
                    alpha = params["alpha"]
                    threshold = params["threshold"]
                    meta_p = float(cache[f"anom_{anom_name}"][i])
                    new_p = float(new_anom_p[ai])
                    blended = alpha * meta_p + (1 - alpha) * new_p
                    effective_threshold = threshold
                    if context_lower:
                        effective_threshold = min(threshold, ct)
                    if blended >= effective_threshold:
                        anomalies.append(anom_name)
                pred_base, pred_anoms = base_type, anomalies

        mt = match_type(pred_base, pred_anoms, expected)
        results_per_gid[gid][mt.lower()] += 1

    # Aggregate
    total = sum(sum(v.values()) for v in results_per_gid.values())
    full = sum(v["full"] for v in results_per_gid.values())
    partial = sum(v["partial"] for v in results_per_gid.values())
    none = sum(v["none"] for v in results_per_gid.values())

    return {"full": full, "partial": partial, "none": none, "total": total, "per_gid": dict(results_per_gid)}


def load_blend_params():
    bp = joblib.load(META_MODELS_DIR / "blend_weights.pkl")
    return bp


def print_top(results, title, n=10):
    results.sort(key=lambda x: -x["full"])
    print(f"\n=== {title} ===")
    for i, r in enumerate(results[:n]):
        print(f"  #{i+1}  {r['strategy']:<45}  FULL={r['full']:4d} ({100*r['full']/r['total']:.2f}%)")


if __name__ == "__main__":
    cache = load_cache()
    blend_params = joblib.load(META_MODELS_DIR / "blend_weights.pkl")

    baseline = {
        "stat_version": "v2",
        "stat_threshold": 0.95,
        "router_theta": 0.40,
        "context_thresh": 0.0,
        "context_base_types": ["stationary"],
        "blend_params": blend_params,
    }

    print("\n" + "=" * 70)
    print("  FAST GRID SEARCH (cached)")
    print("=" * 70)

    # ---- 1. Baseline ----
    r0 = evaluate_strategy(cache, baseline)
    print(f"\n  Baseline: FULL={r0['full']}  ({100*r0['full']/r0['total']:.2f}%)")

    # ---- 2. Stat version x threshold ----
    all_results = []
    print("\n[1] STAT VERSION x THRESHOLD")
    for version in ["v1", "v2", "v3", "v4", "v5", "v6"]:
        for thresh in [0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 1.01]:
            s = dict(baseline)
            s["stat_version"] = version
            s["stat_threshold"] = thresh
            r = evaluate_strategy(cache, s)
            r["strategy"] = f"stat={version}@{thresh}"
            all_results.append(r)
    print_top(all_results, "TOP 15 (stat x threshold)", 15)

    # ---- 3. Router theta sweep (best stat from above) ----
    best = max(all_results, key=lambda x: x["full"])
    best_stat_v = best["strategy"].split("=")[1].split("@")[0]
    best_stat_t = float(best["strategy"].split("@")[1])
    print(f"\n[2] ROUTER THETA SWEEP (stat={best_stat_v}@{best_stat_t})")
    rt_results = []
    for rt in [0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]:
        s = dict(baseline)
        s["stat_version"] = best_stat_v
        s["stat_threshold"] = best_stat_t
        s["router_theta"] = rt
        r = evaluate_strategy(cache, s)
        r["strategy"] = f"stat={best_stat_v}@{best_stat_t} RT={rt}"
        rt_results.append(r)
    print_top(rt_results, "TOP 10 (router theta)", 10)

    # ---- 4. Context threshold sweep ----
    best_rt = max(rt_results, key=lambda x: x["full"])
    best_router_t = float(best_rt["strategy"].split("RT=")[1])
    print(f"\n[3] CONTEXT THRESHOLD SWEEP")
    ct_results = []
    for ct in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.44, 0.48]:
        s = dict(baseline)
        s["stat_version"] = best_stat_v
        s["stat_threshold"] = best_stat_t
        s["router_theta"] = best_router_t
        s["context_thresh"] = ct
        r = evaluate_strategy(cache, s)
        r["strategy"] = f"CT={ct}"
        ct_results.append(r)
    print_top(ct_results, "TOP 10 (context threshold)", 10)

    # ---- 5. Blend alpha sweep ----
    best_ct = max(ct_results, key=lambda x: x["full"])
    best_ct_val = float(best_ct["strategy"].split("CT=")[1])
    print(f"\n[4] BLEND ALPHA SWEEP (fixed)")
    blend_results = []
    for alpha in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
        custom_blend = {k: {"alpha": alpha, "threshold": 0.5} for k in ANOM_LABELS}
        s = dict(baseline)
        s["stat_version"] = best_stat_v
        s["stat_threshold"] = best_stat_t
        s["router_theta"] = best_router_t
        s["context_thresh"] = best_ct_val
        s["blend_params"] = custom_blend
        r = evaluate_strategy(cache, s)
        r["strategy"] = f"blend_alpha={alpha}"
        blend_results.append(r)
    print_top(blend_results, "TOP (fixed alpha)", 8)

    # ---- 6. Combined best ----
    combined = [
        ("BASELINE", baseline),
        ("best stat+rt+ct", {**baseline, "stat_version": best_stat_v, "stat_threshold": best_stat_t,
                              "router_theta": best_router_t, "context_thresh": best_ct_val}),
    ]
    print("\n[5] FINAL COMPARISON")
    for name, s in combined:
        r = evaluate_strategy(cache, s)
        print(f"  {name:<30}  FULL={r['full']}  ({100*r['full']/r['total']:.2f}%)")

    # Overall best
    all_combined = all_results + rt_results + ct_results + blend_results
    best_overall = max(all_combined, key=lambda x: x["full"])
    print(f"\n  OVERALL BEST: {best_overall['strategy']}")
    print(f"  FULL={best_overall['full']} ({100*best_overall['full']/best_overall['total']:.2f}%)")
