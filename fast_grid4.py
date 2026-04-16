"""
Fast Grid v4 — Group 1 spurious anomaly suppression.
Eger base=stationary VE max(blended_anom) < SUPPRESS_THRESH ise → anomali yok.
"""
import warnings
from collections import defaultdict

import joblib
import numpy as np

from config import (
    ANOM_LABELS, BASE_LABELS, GROUP_EXPECTED, GROUP_NAMES,
    META_MODELS_DIR, PROCESSED_DIR,
)
from fast_grid import load_cache, match_type

warnings.filterwarnings("ignore")


def evaluate_with_suppression(cache, baseline, suppress_thresh, suppress_bases):
    """
    v12 baseline pipeline (context_thresh=0 stationary base'de) + spurious anomali suppression.

    Once context_thresh=0 ile anomalileri belirle, sonra:
    SUPPRESSION: base in suppress_bases VE max(blended) < suppress_thresh ise anomali listesi BOSALT
    """
    n = len(cache["gid"])
    rt = baseline["router_theta"]
    ct = baseline.get("context_thresh", 0.0)
    ct_bases = set(baseline.get("context_base_types", ["stationary"]))
    blend = baseline["blend_params"]
    stat_t = baseline["stat_threshold"]
    stat_v = baseline["stat_version"]
    stat_probs = cache[f"stat_{stat_v}"]

    results_per_gid = defaultdict(lambda: {"full": 0, "partial": 0, "none": 0})

    for i in range(n):
        gid = int(cache["gid"][i])
        expected = GROUP_EXPECTED[gid]

        base_idx = int(np.argmax(cache["base_proba"][i]))
        base_type = BASE_LABELS[base_idx]

        # Stat gate
        if stat_probs[i] >= stat_t:
            pred_base, pred_anoms = "stationary", []
        else:
            p_combo = cache["router_proba"][i]
            if p_combo < rt:
                pred_base, pred_anoms = base_type, []
            else:
                # Combo branch — v12 style
                new_anom_p = cache["raw_new"][i, 4:]
                context_lower = (base_type in ct_bases)
                blended_probs = {}
                anomalies = []
                for ai, anom_name in enumerate(ANOM_LABELS):
                    params = blend.get(anom_name, {"alpha": 1.0, "threshold": 0.5})
                    alpha = params["alpha"]
                    threshold = params["threshold"]
                    meta_p = float(cache[f"anom_{anom_name}"][i])
                    new_p = float(new_anom_p[ai])
                    blended = alpha * meta_p + (1 - alpha) * new_p
                    blended_probs[anom_name] = blended

                    eff_t = min(threshold, ct) if context_lower else threshold
                    if blended >= eff_t:
                        anomalies.append(anom_name)

                # SUPPRESSION RULE (yeni eklendi):
                # base in suppress_bases AND max(blended) cok dusukse → anomalileri bastir
                if base_type in suppress_bases and anomalies:
                    max_blended = max(blended_probs.values())
                    if max_blended < suppress_thresh:
                        anomalies = []

                pred_base, pred_anoms = base_type, anomalies

        mt = match_type(pred_base, pred_anoms, expected)
        results_per_gid[gid][mt.lower()] += 1

    total = sum(sum(v.values()) for v in results_per_gid.values())
    full = sum(v["full"] for v in results_per_gid.values())
    partial = sum(v["partial"] for v in results_per_gid.values())
    none = sum(v["none"] for v in results_per_gid.values())
    return {"full": full, "partial": partial, "none": none, "total": total,
            "per_gid": dict(results_per_gid)}


if __name__ == "__main__":
    cache = load_cache()

    baseline = {
        "stat_version": "v2",
        "stat_threshold": 0.92,
        "router_theta": 0.30,
        "context_thresh": 0.0,
        "context_base_types": ["stationary"],
        "blend_params": {
            "collective_anomaly":  {"alpha": 0.85, "threshold": 0.73},
            "contextual_anomaly":  {"alpha": 0.70, "threshold": 0.69},
            "mean_shift":          {"alpha": 0.90, "threshold": 0.49},
            "point_anomaly":       {"alpha": 0.70, "threshold": 0.69},
            "trend_shift":         {"alpha": 0.90, "threshold": 0.73},
            "variance_shift":      {"alpha": 0.70, "threshold": 0.69},
        },
    }

    # Baseline
    r0 = evaluate_with_suppression(cache, baseline, 0.0, [])
    print(f"Baseline (no suppression): FULL={r0['full']} ({100*r0['full']/r0['total']:.2f}%)")
    print(f"  G1: {r0['per_gid'].get(1,{}).get('full',0)}/120")

    print("\n[1] Naif suppression (sadece blended esigi):")
    for thresh in [0.20, 0.30, 0.40]:
        r = evaluate_with_suppression(cache, baseline, thresh, ["stationary"])
        g1 = r['per_gid'].get(1, {}).get('full', 0)
        print(f"  thresh={thresh} FULL={r['full']} G1={g1}/120")

    print("\n[2] AKILLI SUPPRESSION (stat_prob koşullu):")
    print(f"  base=stat AND stat_prob>=stat_low AND max(blended)<sup_thresh -> sil")
    print(f"  {'sup_t':>6} {'stat_low':>9} {'FULL':>5} {'G1':>7} "
          f"{'G5':>7} {'G7':>7} {'G8':>7} {'G10':>7}")

    def evaluate_smart(cache, baseline, sup_thresh, stat_low):
        n = len(cache["gid"])
        rt = baseline["router_theta"]
        ct = baseline["context_thresh"]
        ct_bases = set(baseline["context_base_types"])
        blend = baseline["blend_params"]
        stat_t = baseline["stat_threshold"]
        stat_v = baseline["stat_version"]
        stat_probs = cache[f"stat_{stat_v}"]
        results_per_gid = defaultdict(lambda: {"full": 0, "partial": 0, "none": 0})

        for i in range(n):
            gid = int(cache["gid"][i])
            expected = GROUP_EXPECTED[gid]
            base_idx = int(np.argmax(cache["base_proba"][i]))
            base_type = BASE_LABELS[base_idx]

            if stat_probs[i] >= stat_t:
                pred_base, pred_anoms = "stationary", []
            else:
                p_combo = cache["router_proba"][i]
                if p_combo < rt:
                    pred_base, pred_anoms = base_type, []
                else:
                    new_anom_p = cache["raw_new"][i, 4:]
                    context_lower = (base_type in ct_bases)
                    blended_probs = {}
                    anomalies = []
                    for ai, anom_name in enumerate(ANOM_LABELS):
                        params = blend.get(anom_name, {"alpha": 1.0, "threshold": 0.5})
                        alpha = params["alpha"]
                        threshold = params["threshold"]
                        meta_p = float(cache[f"anom_{anom_name}"][i])
                        new_p = float(new_anom_p[ai])
                        blended = alpha * meta_p + (1 - alpha) * new_p
                        blended_probs[anom_name] = blended
                        eff_t = min(threshold, ct) if context_lower else threshold
                        if blended >= eff_t:
                            anomalies.append(anom_name)

                    # AKILLI SUPPRESSION:
                    # base==stationary VE stat_prob>=stat_low (yani gercekten stat sinyali)
                    # VE max(blended) < sup_thresh ise → spurious, sil
                    if (base_type == "stationary"
                            and stat_probs[i] >= stat_low
                            and anomalies):
                        max_blended = max(blended_probs.values())
                        if max_blended < sup_thresh:
                            anomalies = []

                    pred_base, pred_anoms = base_type, anomalies

            mt = match_type(pred_base, pred_anoms, expected)
            results_per_gid[gid][mt.lower()] += 1

        total = sum(sum(v.values()) for v in results_per_gid.values())
        full = sum(v["full"] for v in results_per_gid.values())
        return {"full": full, "total": total, "per_gid": dict(results_per_gid)}

    best = (3919, None, None)
    for stat_low in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85]:
        for sup_t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
            r = evaluate_smart(cache, baseline, sup_t, stat_low)
            g1 = r['per_gid'].get(1, {}).get('full', 0)
            g5 = r['per_gid'].get(5, {}).get('full', 0)
            g7 = r['per_gid'].get(7, {}).get('full', 0)
            g8 = r['per_gid'].get(8, {}).get('full', 0)
            g10 = r['per_gid'].get(10, {}).get('full', 0)
            marker = ""
            if r['full'] > best[0]:
                best = (r['full'], stat_low, sup_t)
                marker = " ***"
            if r['full'] >= 3919:
                print(f"  {sup_t:6.2f} {stat_low:9.2f} {r['full']:5d} "
                      f"{g1:>3d}/120 {g5:>3d}/480 {g7:>3d}/480 {g8:>3d}/480 {g10:>3d}/480{marker}")

    print(f"\n  BEST SMART: stat_low={best[1]} sup_t={best[2]} FULL={best[0]}")
