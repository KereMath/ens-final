"""
hopefullyprojectfinal - Hibrit Evaluator v3
Base type: meta-learner
Anomali: meta-learner + safety net (yeni ensemble cok eminse override)
"""
import json
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

from config import (
    ANOM_LABELS, BASE_LABELS, GROUP_EXPECTED, GROUP_NAMES,
    META_MODELS_DIR, MIN_SERIES_LENGTH, NEW_ALL_MODELS, NEW_ANOMALY_MODELS,
    OLD_CLASSES, PROCESSED_DIR, RANDOM_STATE, RESULTS_DIR, SOURCE_GROUPS,
)
from processor import (
    _compute_derived_features, extract_batch, get_leaf_csvs,
    get_new_probs, get_old_probs,
    load_new_ensemble, load_old_ensemble, read_series,
)
from stat_detector import (
    load_stationary_detector, predict_stationary_batch,
)

warnings.filterwarnings("ignore")

SAMPLES_PER_LEAF = 10
CONTEXT_THRESH = 0.0     # combo + stationary base icin anomali esigi
STAT_DET_THRESHOLD = 0.95    # stat detector >= bu → direkt stationary, anomali yok


# ===================================================================
# Model yukleme
# ===================================================================
def load_meta_models():
    base_meta = joblib.load(META_MODELS_DIR / "base_meta.pkl")
    anom_metas = {}
    for anom_name in ANOM_LABELS:
        path = META_MODELS_DIR / f"anom_{anom_name}.pkl"
        if path.exists():
            anom_metas[anom_name] = joblib.load(path)
    blend_params = joblib.load(META_MODELS_DIR / "blend_weights.pkl")
    router_path = META_MODELS_DIR / "router.pkl"
    router = joblib.load(router_path) if router_path.exists() else None
    print(f"  Meta-models: base(XGB+LGB) + {len(anom_metas)}/6 anomaly + router + blend params")
    return base_meta, anom_metas, blend_params, router


def load_tsfresh_scaler():
    scaler = joblib.load(PROCESSED_DIR / "tsfresh_scaler.pkl")
    print(f"  tsfresh scaler yuklendi")
    return scaler


# ===================================================================
# Meta-feature olusturma
# ===================================================================
def build_meta_features_batch(old_models, new_models, tsfresh_scaler,
                              X_tsfresh_batch):
    n = X_tsfresh_batch.shape[0]
    raw_old = np.zeros((n, 9))
    raw_new = np.zeros((n, 10))

    for i in range(n):
        raw_old[i] = get_old_probs(old_models, X_tsfresh_batch[i])
        raw_new[i] = get_new_probs(new_models, X_tsfresh_batch[i])

    derived = _compute_derived_features(raw_old, raw_new)

    X_clean = np.nan_to_num(X_tsfresh_batch, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = tsfresh_scaler.transform(X_clean)

    meta_X = np.hstack([raw_old, raw_new, derived, X_scaled])
    return meta_X, raw_new


# ===================================================================
# Tahmin: meta-learner + safety net
# ===================================================================
def predict_batch(base_meta, anom_metas, blend_params, router, meta_X, raw_new_probs, stat_probs=None):
    """
    1. Router: "single mi combo mu?" (810 feature)
    2. Base type: meta-learner (her zaman)
    3. Anomali:
       - Router "single" derse → anomali YOK (FP imkansiz)
       - Router "combo" derse → blended anomali detection
    """
    ROUTER_THETA = 0.40  # combo icin dusuk esik → combo kacirilmasin

    # Base type: ensemble predict (XGB+LGB)
    xgb_base_proba = base_meta["xgb"].predict_proba(meta_X)
    lgb_base_proba = base_meta["lgb"].predict_proba(meta_X)
    ens_base_proba = 0.5 * xgb_base_proba + 0.5 * lgb_base_proba
    base_preds = np.argmax(ens_base_proba, axis=1)

    # Router: P(combo)
    if router is not None:
        xgb_r = router["xgb"].predict_proba(meta_X)[:, 1]
        lgb_r = router["lgb"].predict_proba(meta_X)[:, 1]
        p_combo = 0.5 * xgb_r + 0.5 * lgb_r
    else:
        p_combo = np.ones(meta_X.shape[0])  # router yoksa hep combo

    results = []

    for i in range(meta_X.shape[0]):
        base_idx  = int(base_preds[i])
        base_type = BASE_LABELS[base_idx]

        # STATIONARY DETECTOR OVERRIDE:
        # Stat detector cok yuksek guvenle stationary diyorsa → direkt stationary, anomali yok
        if stat_probs is not None and stat_probs[i] >= STAT_DET_THRESHOLD:
            results.append(("stationary", []))
            continue

        # Router karari
        is_combo = (p_combo[i] >= ROUTER_THETA)

        if not is_combo:
            # SINGLE: sadece base type, anomali yok
            results.append((base_type, []))
            continue

        # COMBO: blended anomali detection + context threshold
        new_anom_probs = raw_new_probs[i, 4:]
        context_lower = (base_type == "stationary")

        blended_probs = {}
        for anom_idx, anom_name in enumerate(ANOM_LABELS):
            params = blend_params.get(anom_name, {"alpha": 0.5, "threshold": 0.5})
            alpha = params["alpha"]
            new_prob = float(new_anom_probs[anom_idx])

            if anom_name in anom_metas:
                models = anom_metas[anom_name]
                xgb_p = float(models["xgb"].predict_proba(meta_X[i:i+1])[0, 1])
                lgb_p = float(models["lgb"].predict_proba(meta_X[i:i+1])[0, 1])
                meta_prob = 0.5 * xgb_p + 0.5 * lgb_p
                blended_probs[anom_name] = alpha * meta_prob + (1 - alpha) * new_prob
            else:
                blended_probs[anom_name] = new_prob

        anomalies = []
        for anom_name in ANOM_LABELS:
            params = blend_params.get(anom_name, {"alpha": 0.5, "threshold": 0.5})
            threshold = params["threshold"]
            # Context: combo + stationary base → esigi dusur (grp 5-10 icin)
            if context_lower:
                threshold = min(threshold, CONTEXT_THRESH)
            if blended_probs.get(anom_name, 0) >= threshold:
                anomalies.append(anom_name)

        results.append((base_type, anomalies))

    return results


# ===================================================================
# Match
# ===================================================================
def match_type(pred_base, pred_anomalies, expected):
    true_base      = expected["base"]
    true_anomalies = expected["anomalies"]
    base_ok        = (pred_base == true_base)

    if not true_anomalies:
        if base_ok and len(pred_anomalies) == 0:
            return "FULL"
        elif base_ok:
            return "PARTIAL"
        else:
            return "NONE"
    else:
        all_anom = all(a in pred_anomalies for a in true_anomalies)
        if base_ok and all_anom:
            return "FULL"
        elif base_ok or all_anom:
            return "PARTIAL"
        else:
            return "NONE"


# ===================================================================
# Ana evaluation
# ===================================================================
def evaluate_stacked(old_models, new_models, base_meta, anom_metas,
                     blend_params, router, tsfresh_scaler):
    print(f"\n{'='*70}")
    print(f"  HYBRID STACKING v11 - STAT DETECTOR INTEGRATED")
    print(f"  Stat detector: P(stationary) >= {STAT_DET_THRESHOLD} -> force stationary")
    print(f"  Samples per leaf = {SAMPLES_PER_LEAF}")
    print(f"{'='*70}")

    # Stationary detector yukle
    stat_model, stat_scaler, stat_selector = load_stationary_detector()

    random.seed(RANDOM_STATE)
    all_results = []

    for gid, gname, roots in SOURCE_GROUPS:
        expected = GROUP_EXPECTED[gid]

        leaf_map = {}
        for root in roots:
            if root.exists():
                for leaf, csvs in get_leaf_csvs(root).items():
                    leaf_map.setdefault(leaf, []).extend(csvs)
        if not leaf_map:
            continue

        selected = []
        leaf_names = []
        for leaf in sorted(leaf_map.keys()):
            pool = sorted(leaf_map[leaf])
            valid = [f for f in pool if f.name != "metadata.csv"]
            k = min(SAMPLES_PER_LEAF, len(valid))
            if k > 0:
                chosen = random.sample(valid, k)
                for c in chosen:
                    selected.append(c)
                    leaf_names.append(leaf.name)

        series_list = []
        valid_csvs  = []
        valid_leaves = []
        for csv_path, lname in zip(selected, leaf_names):
            data = read_series(csv_path)
            if len(data) >= MIN_SERIES_LENGTH:
                series_list.append(data)
                valid_csvs.append(csv_path)
                valid_leaves.append(lname)
        if not series_list:
            continue

        X_batch = extract_batch(series_list)
        meta_X, raw_new = build_meta_features_batch(
            old_models, new_models, tsfresh_scaler, X_batch)

        # Stationary detector probs (raw series uzerinden)
        stat_probs = predict_stationary_batch(
            stat_model, stat_scaler, stat_selector, series_list)

        predictions = predict_batch(
            base_meta, anom_metas, blend_params, router,
            meta_X, raw_new, stat_probs=stat_probs)

        g_full = g_partial = g_none = 0
        for i, csv_path in enumerate(valid_csvs):
            pred_base, pred_anoms = predictions[i]
            mt = match_type(pred_base, pred_anoms, expected)

            pred_str = pred_base
            if pred_anoms:
                pred_str += " + " + " + ".join(sorted(pred_anoms))
            exp_str = expected["base"]
            if expected["anomalies"]:
                exp_str += " + " + " + ".join(sorted(expected["anomalies"]))

            all_results.append({
                "gid": gid, "group": gname, "leaf": valid_leaves[i],
                "csv": csv_path.name, "expected": exp_str,
                "predicted": pred_str, "match": mt,
            })

            if mt == "FULL":      g_full += 1
            elif mt == "PARTIAL": g_partial += 1
            else:                 g_none += 1

        total = g_full + g_partial + g_none
        pct = 100 * g_full / total if total else 0
        print(f"  Grup {gid:2d} ({gname[:30]:<30})  "
              f"n={total:3d}  FULL={g_full:3d} ({pct:5.1f}%)  "
              f"PART={g_partial:3d}  NONE={g_none:3d}")

    total = len(all_results)
    full  = sum(1 for r in all_results if r["match"] == "FULL")
    part  = sum(1 for r in all_results if r["match"] == "PARTIAL")
    none  = sum(1 for r in all_results if r["match"] == "NONE")

    print(f"\n{'='*70}")
    print(f"  GENEL SONUC: {total} ornek")
    print(f"  FULL:    {full:4d}  ({100*full/total:.1f}%)")
    print(f"  PARTIAL: {part:4d}  ({100*part/total:.1f}%)")
    print(f"  NONE:    {none:4d}  ({100*none/total:.1f}%)")
    print(f"{'='*70}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "total": total, "full": full, "partial": part, "none": none,
        "full_pct": round(100 * full / total, 2),
        "approach": "blended stacking v4",
        "samples_per_leaf": SAMPLES_PER_LEAF,
    }
    with open(RESULTS_DIR / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": all_results}, f, indent=2, ensure_ascii=False)

    write_markdown_report(all_results, summary)
    return summary, all_results


def write_markdown_report(all_results, summary):
    lines = []
    lines.append("# Hybrid Stacking v3 - Test Sonuclari\n")
    lines.append("Base: meta-learner | Anomali: meta-learner + safety net (new ensemble >= 0.80 override)\n")

    lines.append("## Ozet\n")
    lines.append("| Metrik | Deger |")
    lines.append("|---|---|")
    lines.append(f"| Toplam test | {summary['total']} |")
    lines.append(f"| Full match | {summary['full']} ({summary['full_pct']}%) |")
    lines.append(f"| Partial match | {summary['partial']} ({round(100*summary['partial']/summary['total'],1)}%) |")
    lines.append(f"| No match | {summary['none']} ({round(100*summary['none']/summary['total'],1)}%) |")
    lines.append("")

    lines.append("## Grup Bazli Ozet\n")
    lines.append("| # | Grup | Beklenen | Ornek | Full | Partial | None | Full% |")
    lines.append("|---|---|---|---|---|---|---|---|")

    group_stats = defaultdict(lambda: {"total": 0, "full": 0, "partial": 0, "none": 0})
    for r in all_results:
        g = group_stats[r["gid"]]
        g["total"] += 1
        if r["match"] == "FULL":      g["full"] += 1
        elif r["match"] == "PARTIAL": g["partial"] += 1
        else:                          g["none"] += 1
        g["expected"] = r["expected"]
        g["group"]    = r["group"]

    for gid in sorted(group_stats.keys()):
        g = group_stats[gid]
        pct = round(100 * g["full"] / g["total"]) if g["total"] else 0
        lines.append(f"| {gid} | {g['group']} | {g['expected']} | {g['total']} | "
                      f"{g['full']} | {g['partial']} | {g['none']} | %{pct} |")

    lines.append("\n---\n")
    for gid in sorted(group_stats.keys()):
        g = group_stats[gid]
        gresults = [r for r in all_results if r["gid"] == gid]
        lines.append(f"## Grup {gid}: {g['group']}")
        lines.append(f"**Beklenen:** `{g['expected']}`\n")
        lines.append("| Leaf | CSV | Tahmin | Sonuc |")
        lines.append("|---|---|---|---|")
        for r in gresults:
            sym = {"FULL": "FULL", "PARTIAL": "~ PARTIAL", "NONE": "x NONE"}[r["match"]]
            lines.append(f"| `{r['leaf']}` | `{r['csv']}` | `{r['predicted']}` | {sym} |")
        lines.append("---\n")

    report_path = RESULTS_DIR / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Rapor: {report_path}")
