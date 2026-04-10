"""
Mevcut en iyi konfigurasyonun 39 grup detay tablosu.
"""
from collections import defaultdict

import numpy as np

from config import ANOM_LABELS, BASE_LABELS, GROUP_EXPECTED, GROUP_NAMES
from fast_grid import load_cache, match_type

cache = load_cache()

# Best config
base = {
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

stat_probs = cache[f"stat_{base['stat_version']}"]
rt = base["router_theta"]
ct = base["context_thresh"]
ct_bases = set(base["context_base_types"])
blend = base["blend_params"]
stat_t = base["stat_threshold"]

n = len(cache["gid"])
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
            anomalies = []
            for ai, anom_name in enumerate(ANOM_LABELS):
                params = blend.get(anom_name, {"alpha": 1.0, "threshold": 0.5})
                alpha = params["alpha"]
                threshold = params["threshold"]
                meta_p = float(cache[f"anom_{anom_name}"][i])
                new_p = float(new_anom_p[ai])
                blended = alpha * meta_p + (1 - alpha) * new_p
                eff_t = threshold
                if context_lower:
                    eff_t = min(threshold, ct)
                if blended >= eff_t:
                    anomalies.append(anom_name)
            pred_base, pred_anoms = base_type, anomalies

    mt = match_type(pred_base, pred_anoms, expected)
    results_per_gid[gid][mt.lower()] += 1

# Print table
print("\n# Best Config v12 — %89.07 FULL (3919/4400)\n")
print("| # | Grup | n | FULL | PART | NONE | FULL% |")
print("|---|---|---|---|---|---|---|")

total_f = total_p = total_n = 0
for gid in sorted(results_per_gid.keys()):
    r = results_per_gid[gid]
    total = r["full"] + r["partial"] + r["none"]
    pct = 100 * r["full"] / total if total else 0
    name = GROUP_NAMES[gid]
    print(f"| {gid} | {name[:30]:<30} | {total} | {r['full']} | {r['partial']} | {r['none']} | {pct:.1f} |")
    total_f += r["full"]
    total_p += r["partial"]
    total_n += r["none"]

print(f"\n**TOTAL: {total_f}/{total_f+total_p+total_n} FULL ({100*total_f/(total_f+total_p+total_n):.2f}%)**")
print(f"PARTIAL: {total_p} | NONE: {total_n}")
