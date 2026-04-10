# EN IYI SONUCLAR — BACKUP

## En Iyi Genel Skor: %88.6 FULL (3900/4400)

**Konfigurasyon:** v11 — Stat Detector v2 @ 0.95

### evaluator.py Ayarlari
```python
SAMPLES_PER_LEAF = 10
CONTEXT_THRESH = 0.0          # combo + stationary base icin
STAT_DET_THRESHOLD = 0.95     # stat detector >= bu -> direkt stationary
ROUTER_THETA = 0.40           # predict_batch icinde
```

### config.py Ayarlari
```python
STATIONARY_DETECTOR_DIR = ROOT_DIR / "stationary detector ml" / "trained_models v2"
META_N_PER_GROUP = 500
```

---

## Tum 39 Class Tablosu (v11 Best)

| # | Grup | Beklenen | n | FULL | PART | NONE | FULL% |
|---|---|---|---|---|---|---|---|
| 1 | stationary | stationary | 120 | 64 | 52 | 4 | 53.3 |
| 2 | deterministic_trend | det_trend | 720 | 668 | 16 | 36 | 92.8 |
| 3 | stochastic_trend | stoch_trend | 150 | 124 | 20 | 6 | 82.7 |
| 4 | volatility | volatility | 120 | 99 | 14 | 7 | 82.5 |
| 5 | collective_anomaly | stat+collective | 480 | 432 | 18 | 30 | 90.0 |
| 6 | contextual_anomaly | stat+contextual | 480 | 478 | 2 | 0 | 99.6 |
| 7 | mean_shift | stat+mean_shift | 480 | 421 | 27 | 32 | 87.7 |
| 8 | point_anomaly | stat+point | 480 | 407 | 26 | 47 | 84.8 |
| 9 | trend_shift | stat+trend_shift | 480 | 440 | 13 | 27 | 91.7 |
| 10 | variance_shift | stat+var_shift | 480 | 384 | 50 | 46 | 80.0 |
| 11 | cubic+collective | det+collective | 10 | 10 | 0 | 0 | 100.0 |
| 12 | cubic+mean_shift | det+mean | 20 | 19 | 1 | 0 | 95.0 |
| 13 | cubic+point_anomaly | det+point | 10 | 10 | 0 | 0 | 100.0 |
| 14 | cubic+variance_shift | det+var | 10 | 10 | 0 | 0 | 100.0 |
| 15 | damped+collective | det+collective | 10 | 10 | 0 | 0 | 100.0 |
| 16 | damped+mean_shift | det+mean | 20 | 20 | 0 | 0 | 100.0 |
| 17 | damped+point_anomaly | det+point | 10 | 10 | 0 | 0 | 100.0 |
| 18 | damped+variance_shift | det+var | 10 | 10 | 0 | 0 | 100.0 |
| 19 | exp+collective | det+collective | 10 | 10 | 0 | 0 | 100.0 |
| 20 | exp+mean_shift | det+mean | 20 | 20 | 0 | 0 | 100.0 |
| 21 | exp+point_anomaly | det+point | 10 | 10 | 0 | 0 | 100.0 |
| 22 | exp+variance_shift | det+var | 10 | 10 | 0 | 0 | 100.0 |
| 23 | linear+collective | det+collective | 10 | 10 | 0 | 0 | 100.0 |
| 24 | linear+mean_shift | det+mean | 20 | 19 | 1 | 0 | 95.0 |
| 25 | linear+point_anomaly | det+point | 10 | 10 | 0 | 0 | 100.0 |
| 26 | linear+trend_shift | det+trend_shift | 30 | 30 | 0 | 0 | 100.0 |
| 27 | linear+variance_shift | det+var | 10 | 10 | 0 | 0 | 100.0 |
| 28 | quad+collective | det+collective | 10 | 10 | 0 | 0 | 100.0 |
| 29 | quad+mean_shift | det+mean | 20 | 20 | 0 | 0 | 100.0 |
| 30 | quad+point_anomaly | det+point | 20 | 20 | 0 | 0 | 100.0 |
| 31 | quad+variance_shift | det+var | 10 | 9 | 1 | 0 | 90.0 |
| 32 | stoch+collective | stoch+collective | 10 | 7 | 3 | 0 | 70.0 |
| 33 | stoch+mean_shift | stoch+mean | 10 | 6 | 4 | 0 | 60.0 |
| 34 | stoch+point_anomaly | stoch+point | 10 | 10 | 0 | 0 | 100.0 |
| 35 | stoch+variance_shift | stoch+var | 50 | 45 | 5 | 0 | 90.0 |
| 36 | vol+collective | vol+collective | 10 | 5 | 5 | 0 | 50.0 |
| 37 | vol+mean_shift | vol+mean | 10 | 10 | 0 | 0 | 100.0 |
| 38 | vol+point_anomaly | vol+point | 10 | 6 | 4 | 0 | 60.0 |
| 39 | vol+variance_shift | vol+var | 10 | 7 | 3 | 0 | 70.0 |

**TOPLAM: 3900/4400 FULL = %88.6**
PARTIAL: 269 (6.1%) | NONE: 235 (5.3%)

---

## Tum Versiyon Gecmisi

| Version | Aciklama | FULL% |
|---|---|---|
| Orijinal (ensemble-alldata) | Tek ensemble | 59.8% |
| v1-v2 (stacking 50/grp) | Ilk stacking, PCA | 67.8% |
| v3-v4 (150/grp + threshold) | Per-anom threshold | 74.6% |
| v5 (oversample) | Grup 5-10 oversample | 77.0% |
| v6 (stat ctx) | Context threshold only stationary | 77.5% |
| v7 (XGB+LGB) | Dual meta-learner | 77.5% |
| v8 (router) | 810-feature router | 77.9% |
| v9 (CT=0.42) | Aggressive context | 78.3% |
| v10 (CT=0.0) | All context | 88.5% |
| **v11 (stat v2@0.95)** | **Stat detector v2** | **88.6% (EN IYI)** |

---

## Anahtar Dosyalar (Backup)

### Egitilmis Modeller
- `meta_models/base_meta.pkl` (XGB+LGB ensemble)
- `meta_models/anom_*.pkl` (6 anomaly meta-learners, XGB+LGB)
- `meta_models/router.pkl` (single/combo router, XGB+LGB)
- `meta_models/blend_weights.pkl` (blend params)

### Cached Data
- `processed_data/meta_X.npy` (19500 x 810 meta features)
- `processed_data/meta_y_base.npy`
- `processed_data/meta_y_anom.npy`
- `processed_data/tsfresh_scaler.pkl`

### External (Kullanilan)
- `../stationary detector ml/trained_models v2/` - stat detector v2
- `../ensemble-alldata/trained_models/` - yeni ensemble (10 model)
- `../tsfresh ensemble/trained_models/` - eski ensemble (9 detector)

### Kod
- `config.py` - 39 grup tanimi, path'ler
- `processor.py` - tsfresh + ensemble inference
- `trainer.py` - meta-learner egitimi
- `evaluator.py` - stacking evaluation (v11)
- `stat_detector.py` - stationary detector wrapper
- `main.py` - pipeline orchestration
