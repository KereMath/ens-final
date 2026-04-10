"""
Fast Grid Search v2 — gelismis stratejiler
"""
import warnings
from collections import defaultdict

import joblib
import numpy as np

from config import (
    ANOM_LABELS, BASE_LABELS, GROUP_EXPECTED,
    META_MODELS_DIR, PROCESSED_DIR,
)
from fast_grid import evaluate_strategy, load_cache, match_type

warnings.filterwarnings("ignore")


def print_top(results, title, n=10):
    results.sort(key=lambda x: -x["full"])
    print(f"\n=== {title} ===")
    for i, r in enumerate(results[:n]):
        print(f"  #{i+1}  {r['strategy']:<55}  FULL={r['full']:4d} ({100*r['full']/r['total']:.2f}%)")


def evaluate_stat_ensemble(cache, stat_versions, stat_weights, threshold, baseline):
    """Birden fazla stat version'u birlestir."""
    n = len(cache["gid"])

    # Ensemble stat prob
    ensemble_probs = np.zeros(n)
    total_w = sum(stat_weights)
    for v, w in zip(stat_versions, stat_weights):
        ensemble_probs += (w / total_w) * cache[f"stat_{v}"]

    # Tek seferlik eval
    results_per_gid = defaultdict(lambda: {"full": 0, "partial": 0, "none": 0})
    rt = baseline.get("router_theta", 0.35)
    ct = baseline.get("context_thresh", 0.0)
    ct_bases = set(baseline.get("context_base_types", ["stationary"]))
    blend = baseline.get("blend_params", {})

    for i in range(n):
        gid = int(cache["gid"][i])
        expected = GROUP_EXPECTED[gid]

        base_idx = int(np.argmax(cache["base_proba"][i]))
        base_type = BASE_LABELS[base_idx]

        # Stat ensemble override
        if ensemble_probs[i] >= threshold:
            pred_base, pred_anoms = "stationary", []
        else:
            p_combo = cache["router_proba"][i]
            if p_combo < rt:
                pred_base, pred_anoms = base_type, []
            else:
                new_anom_p = cache["raw_new"][i, 4:]
                context_lower = (base_type in ct_bases)
                anomalies = []
                for ai, anom_name in enumerate(ANOM_LABELS):
                    params = blend.get(anom_name, {"alpha": 1.0, "threshold": 0.5})
                    alpha = params["alpha"]
                    threshold_a = params["threshold"]
                    meta_p = float(cache[f"anom_{anom_name}"][i])
                    new_p = float(new_anom_p[ai])
                    blended = alpha * meta_p + (1 - alpha) * new_p
                    eff_t = threshold_a
                    if context_lower:
                        eff_t = min(threshold_a, ct)
                    if blended >= eff_t:
                        anomalies.append(anom_name)
                pred_base, pred_anoms = base_type, anomalies

        mt = match_type(pred_base, pred_anoms, expected)
        results_per_gid[gid][mt.lower()] += 1

    total = sum(sum(v.values()) for v in results_per_gid.values())
    full = sum(v["full"] for v in results_per_gid.values())
    partial = sum(v["partial"] for v in results_per_gid.values())
    none = sum(v["none"] for v in results_per_gid.values())
    return {"full": full, "partial": partial, "none": none, "total": total}


def per_anomaly_grid(cache, baseline, anom_name, thresh_range):
    """Tek bir anomali icin threshold arama."""
    results = []
    for t in thresh_range:
        blend = dict(baseline["blend_params"])
        blend[anom_name] = {**blend.get(anom_name, {"alpha": 1.0, "threshold": 0.5}), "threshold": t}
        s = dict(baseline)
        s["blend_params"] = blend
        r = evaluate_strategy(cache, s)
        r["strategy"] = f"{anom_name} t={t:.2f}"
        results.append(r)
    return results


if __name__ == "__main__":
    cache = load_cache()
    blend_params = joblib.load(META_MODELS_DIR / "blend_weights.pkl")

    best_baseline = {
        "stat_version": "v2",
        "stat_threshold": 0.95,
        "router_theta": 0.35,
        "context_thresh": 0.0,
        "context_base_types": ["stationary"],
        "blend_params": blend_params,
    }

    all_results = []

    # ===== 1. JOINT: stat version x threshold x router theta =====
    print("\n[1] JOINT stat_v x stat_t x router_theta")
    joint_results = []
    for version in ["v1", "v2", "v3", "v4", "v5", "v6"]:
        for stat_t in [0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 1.01]:
            for rt in [0.30, 0.32, 0.35, 0.38, 0.40, 0.42]:
                s = dict(best_baseline)
                s["stat_version"] = version
                s["stat_threshold"] = stat_t
                s["router_theta"] = rt
                r = evaluate_strategy(cache, s)
                r["strategy"] = f"stat={version}@{stat_t} RT={rt}"
                joint_results.append(r)
    print_top(joint_results, "TOP 15 joint", 15)
    all_results += joint_results

    # ===== 2. Stat ensemble (v1+v2 ortalama) =====
    print("\n[2] STAT ENSEMBLE (v1+v2, v1+v2+v4, vs.)")
    ensemble_configs = [
        (["v1", "v2"], [1, 1]),
        (["v1", "v2", "v4"], [1, 1, 1]),
        (["v2"], [1]),                          # baseline (single)
        (["v1", "v2"], [0.3, 0.7]),             # v2 aglikli
        (["v1", "v2", "v3", "v4"], [1, 1, 1, 1]),
        (["v1", "v2", "v3", "v4", "v5", "v6"], [1]*6),
    ]
    ens_results = []
    for versions, weights in ensemble_configs:
        for thresh in [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]:
            r = evaluate_stat_ensemble(cache, versions, weights, thresh,
                                        {**best_baseline, "router_theta": 0.35})
            r["strategy"] = f"ens({'+'.join(versions)}) w={weights} @{thresh}"
            ens_results.append(r)
    print_top(ens_results, "TOP 10 stat ensembles", 10)
    all_results += ens_results

    # ===== 3. Per-anomaly threshold fine-tuning =====
    print("\n[3] PER-ANOMALY THRESHOLD TUNING")
    # Her anomaliyi sirasiyla fine-tune et
    cumulative_baseline = dict(best_baseline)
    cumulative_blend = dict(blend_params)

    for anom_name in ANOM_LABELS:
        thresh_range = [0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]
        results = per_anomaly_grid(cache, cumulative_baseline, anom_name, thresh_range)
        best_r = max(results, key=lambda x: x["full"])
        best_t = float(best_r["strategy"].split("t=")[1])
        cumulative_blend[anom_name] = {**cumulative_blend.get(anom_name, {"alpha": 1.0}), "threshold": best_t}
        cumulative_baseline["blend_params"] = cumulative_blend
        print(f"  {anom_name:<25}  best_t={best_t:.2f}  FULL={best_r['full']}")
        all_results += results

    # Final cumulative
    final_r = evaluate_strategy(cache, cumulative_baseline)
    final_r["strategy"] = "cumulative per-anom tuned"
    print(f"\n  Cumulative: FULL={final_r['full']} ({100*final_r['full']/final_r['total']:.2f}%)")
    all_results.append(final_r)

    # ===== 4. Router theta ultra fine =====
    print("\n[4] ROUTER THETA ULTRA FINE")
    rt_results = []
    for rt in np.arange(0.20, 0.45, 0.01):
        s = dict(best_baseline)
        s["router_theta"] = float(rt)
        r = evaluate_strategy(cache, s)
        r["strategy"] = f"RT={rt:.2f}"
        rt_results.append(r)
    print_top(rt_results, "TOP 10 router theta", 10)
    all_results += rt_results

    # ===== 5. COMBINED: best joint + per-anomaly tuning =====
    print("\n[5] COMBINED BEST")
    best_joint = max(joint_results, key=lambda x: x["full"])
    print(f"  Best joint: {best_joint['strategy']}")
    print(f"  Best cumulative: {final_r['strategy']}")

    # Try combining: joint best settings + per-anom thresholds
    best_joint_v = best_joint["strategy"].split("=")[1].split("@")[0]
    best_joint_t = float(best_joint["strategy"].split("@")[1].split(" ")[0])
    best_joint_rt = float(best_joint["strategy"].split("RT=")[1])

    combined_s = {
        "stat_version": best_joint_v,
        "stat_threshold": best_joint_t,
        "router_theta": best_joint_rt,
        "context_thresh": 0.0,
        "context_base_types": ["stationary"],
        "blend_params": cumulative_blend,
    }
    combined_r = evaluate_strategy(cache, combined_s)
    combined_r["strategy"] = "joint_best + per_anom_tuned"
    print(f"  Combined: FULL={combined_r['full']} ({100*combined_r['full']/combined_r['total']:.2f}%)")
    all_results.append(combined_r)

    # ===== OVERALL BEST =====
    print("\n" + "="*70)
    all_results.sort(key=lambda x: -x["full"])
    print("  TOP 5 OVERALL")
    for i, r in enumerate(all_results[:5]):
        print(f"  #{i+1}  {r['strategy']:<55}  FULL={r['full']} ({100*r['full']/r['total']:.2f}%)")

    best = all_results[0]
    print(f"\n  *** OVERALL BEST: {best['strategy']} ***")
    print(f"  FULL={best['full']}  PARTIAL={best['partial']}  NONE={best['none']}  Total={best['total']}")
    print(f"  FULL% = {100*best['full']/best['total']:.2f}%")
