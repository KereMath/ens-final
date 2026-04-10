"""
Standalone Stationary Detector Wrapper
Kullanir: "stationary detector ml/trained_models v6/" modellerini
Kendi feature extraction'i ile (tsfresh DEGIL, ~30 boyut custom features).
"""
import pickle
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np

from config import STATIONARY_DETECTOR_DIR


# ===================================================================
# Feature Extraction (stationary detector'dan kopyalandi)
# ===================================================================
def _skewness(data):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    if std == 0 or n < 3:
        return 0
    return (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)


def _kurtosis(data):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    if std == 0 or n < 4:
        return 0
    return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * \
           np.sum(((data - mean) / std) ** 4) - \
           (3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def _rolling_stat(data, window, func):
    return np.array([func(data[i:i + window]) for i in range(len(data) - window + 1)])


def _autocorr(data, lag):
    if lag >= len(data) or lag < 1:
        return 0
    n = len(data)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / n
    if c0 == 0:
        return 0
    ck = np.sum((data[:-lag] - mean) * (data[lag:] - mean)) / n
    return ck / c0


def _count_peaks(data):
    if len(data) < 3:
        return 0
    return sum(1 for i in range(1, len(data) - 1)
               if data[i] > data[i - 1] and data[i] > data[i + 1])


def _zero_cross_rate(data):
    if len(data) < 2:
        return 0
    return np.sum(np.diff(np.sign(data)) != 0) / (len(data) - 1)


def extract_stat_features(data: np.ndarray) -> Optional[Dict]:
    """Stationary detector'un kullandigi ~30 feature'i cikarir."""
    if len(data) < 2:
        return None

    features = {}
    try:
        features['mean'] = np.mean(data)
        features['std'] = np.std(data)
        features['var'] = np.var(data)
        features['min'] = np.min(data)
        features['max'] = np.max(data)
        features['range'] = features['max'] - features['min']
        features['q25'] = np.percentile(data, 25)
        features['median'] = np.median(data)
        features['q75'] = np.percentile(data, 75)
        features['iqr'] = features['q75'] - features['q25']
        features['skewness'] = _skewness(data)
        features['kurtosis'] = _kurtosis(data)
        features['cv'] = features['std'] / (features['mean'] + 1e-10)

        diff1 = np.diff(data)
        features['diff1_mean'] = np.mean(diff1)
        features['diff1_std'] = np.std(diff1)
        features['diff1_var'] = np.var(diff1)

        if len(diff1) > 1:
            diff2 = np.diff(diff1)
            features['diff2_mean'] = np.mean(diff2)
            features['diff2_std'] = np.std(diff2)
        else:
            features['diff2_mean'] = 0
            features['diff2_std'] = 0

        window = max(2, len(data) // 10)
        if window < len(data):
            rm = _rolling_stat(data, window, np.mean)
            rs = _rolling_stat(data, window, np.std)
            features['rolling_mean_std'] = np.std(rm)
            features['rolling_std_mean'] = np.mean(rs)
            features['rolling_std_std'] = np.std(rs)
        else:
            features['rolling_mean_std'] = 0
            features['rolling_std_mean'] = features['std']
            features['rolling_std_std'] = 0

        features['autocorr_lag1'] = _autocorr(data, 1)
        features['autocorr_lag10'] = _autocorr(data, min(10, len(data) - 1))
        features['num_peaks'] = _count_peaks(data)
        features['zero_crossing_rate'] = _zero_cross_rate(data - np.mean(data))
    except Exception:
        return None

    return features


def extract_feature_vector(data: np.ndarray) -> Optional[np.ndarray]:
    """Tek zaman serisi icin feature vektoru dondurur (single chunk)."""
    feats = extract_stat_features(data)
    if feats is None:
        return None
    return np.array(list(feats.values()))


# ===================================================================
# Model Yukleme
# ===================================================================
def load_stationary_detector():
    """Stationary detector v6'yi yukle."""
    model_path = STATIONARY_DETECTOR_DIR / "xgboost_fast.joblib"
    scalers_path = STATIONARY_DETECTOR_DIR / "scalers.pkl"

    model = joblib.load(model_path)
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    main_scaler = scalers.get("main")
    selector = scalers.get("selector")

    print(f"  Stationary detector yuklendi (XGBoost, {main_scaler.n_features_in_} feature)")
    return model, main_scaler, selector


def predict_stationary(model, main_scaler, selector, series: np.ndarray) -> float:
    """
    Seri icin P(stationary) dondurur.
    LABEL_MAP: 0=stationary, 1=non_stationary
    P(stationary) = predict_proba[:, 0]
    """
    feat_vec = extract_feature_vector(series)
    if feat_vec is None:
        return 0.0

    expected = main_scaler.n_features_in_
    if len(feat_vec) != expected:
        if len(feat_vec) < expected:
            feat_vec = np.pad(feat_vec, (0, expected - len(feat_vec)), 'constant')
        else:
            feat_vec = feat_vec[:expected]

    feat_vec = feat_vec.reshape(1, -1)
    scaled = main_scaler.transform(feat_vec)
    if selector is not None:
        scaled = selector.transform(scaled)

    probs = model.predict_proba(scaled)[0]
    # Index 0 = stationary
    return float(probs[0])


def predict_stationary_batch(model, main_scaler, selector, series_list) -> np.ndarray:
    """Batch: P(stationary) dondurur."""
    results = np.zeros(len(series_list))
    expected = main_scaler.n_features_in_

    feat_matrix = np.zeros((len(series_list), expected))
    for i, s in enumerate(series_list):
        fv = extract_feature_vector(s)
        if fv is None:
            continue
        if len(fv) != expected:
            if len(fv) < expected:
                fv = np.pad(fv, (0, expected - len(fv)), 'constant')
            else:
                fv = fv[:expected]
        feat_matrix[i] = fv

    scaled = main_scaler.transform(feat_matrix)
    if selector is not None:
        scaled = selector.transform(scaled)

    probs = model.predict_proba(scaled)
    return probs[:, 0]  # stationary prob
