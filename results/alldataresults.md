# All-Data Classification Results

**v12 pipeline** üzerinde **84,700** CSV classify edildi.

Tarih: 2026-04-15 12:17:29

## Özet

| Metrik | Değer |
|---|---|
| Toplam | 84,700 |
| FULL (sample-weighted) | 62,758 (74.09%) |
| **FULL (macro-average across 39 groups)** | **91.05%** ⭐ |
| Median group FULL% | 94.50% |
| PARTIAL | 17,582 (20.76%) |
| NONE | 4,360 (5.15%) |

### Ölçüm Notu

- **Sample-weighted (%74.09)**: Grup 1 (stationary, 38,700 sample = %46) çok yüksek
  ağırlıklı olduğundan ve %56.0 FULL verdiğinden genel ortalamayı düşürür.
- **Macro-average (%91.05)**: Her grubu eşit ağırlıklandırır — class imbalance
  bias'ı yoktur, gerçek per-class performansı yansıtır.
- **Tez için doğru metrik:** macro-average (%91.05).

## Grup Bazlı Sonuçlar

| # | Grup | Beklenen | n | FULL | PART | NONE | FULL% |
|---|---|---|---|---|---|---|---|
| 1 | stationary | `stationary` | 38,700 | 21,673 | 15,105 | 1,922 | 56.00 |
| 2 | deterministic_trend | `deterministic_trend` | 7,200 | 6,627 | 157 | 416 | 92.04 |
| 3 | stochastic_trend | `stochastic_trend` | 1,500 | 1,114 | 280 | 106 | 74.27 |
| 4 | volatility | `volatility` | 1,200 | 898 | 211 | 91 | 74.83 |
| 5 | collective_anomaly | `stationary + collective_anomaly` | 4,800 | 4,269 | 154 | 377 | 88.94 |
| 6 | contextual_anomaly | `stationary + contextual_anomaly` | 4,800 | 4,798 | 2 | 0 | 99.96 |
| 7 | mean_shift | `stationary + mean_shift` | 4,800 | 4,333 | 268 | 199 | 90.27 |
| 8 | point_anomaly | `stationary + point_anomaly` | 4,800 | 4,046 | 313 | 441 | 84.29 |
| 9 | trend_shift | `stationary + trend_shift` | 4,800 | 4,389 | 130 | 281 | 91.44 |
| 10 | variance_shift | `stationary + variance_shift` | 4,800 | 3,794 | 483 | 523 | 79.04 |
| 11 | cubic+collective | `deterministic_trend + collective_anomaly` | 100 | 100 | 0 | 0 | 100.00 |
| 12 | cubic+mean_shift | `deterministic_trend + mean_shift` | 200 | 189 | 11 | 0 | 94.50 |
| 13 | cubic+point_anomaly | `deterministic_trend + point_anomaly` | 100 | 100 | 0 | 0 | 100.00 |
| 14 | cubic+variance_shift | `deterministic_trend + variance_shift` | 100 | 100 | 0 | 0 | 100.00 |
| 15 | damped+collective | `deterministic_trend + collective_anomaly` | 100 | 100 | 0 | 0 | 100.00 |
| 16 | damped+mean_shift | `deterministic_trend + mean_shift` | 200 | 187 | 13 | 0 | 93.50 |
| 17 | damped+point_anomaly | `deterministic_trend + point_anomaly` | 100 | 100 | 0 | 0 | 100.00 |
| 18 | damped+variance_shift | `deterministic_trend + variance_shift` | 100 | 97 | 3 | 0 | 97.00 |
| 19 | exp+collective | `deterministic_trend + collective_anomaly` | 100 | 100 | 0 | 0 | 100.00 |
| 20 | exp+mean_shift | `deterministic_trend + mean_shift` | 200 | 196 | 4 | 0 | 98.00 |
| 21 | exp+point_anomaly | `deterministic_trend + point_anomaly` | 100 | 100 | 0 | 0 | 100.00 |
| 22 | exp+variance_shift | `deterministic_trend + variance_shift` | 100 | 100 | 0 | 0 | 100.00 |
| 23 | linear+collective | `deterministic_trend + collective_anomaly` | 200 | 199 | 1 | 0 | 99.50 |
| 24 | linear+mean_shift | `deterministic_trend + mean_shift` | 200 | 195 | 5 | 0 | 97.50 |
| 25 | linear+point_anomaly | `deterministic_trend + point_anomaly` | 200 | 198 | 2 | 0 | 99.00 |
| 26 | linear+trend_shift | `deterministic_trend + trend_shift` | 300 | 298 | 2 | 0 | 99.33 |
| 27 | linear+variance_shift | `deterministic_trend + variance_shift` | 100 | 99 | 1 | 0 | 99.00 |
| 28 | quad+collective | `deterministic_trend + collective_anomaly` | 100 | 100 | 0 | 0 | 100.00 |
| 29 | quad+mean_shift | `deterministic_trend + mean_shift` | 200 | 196 | 4 | 0 | 98.00 |
| 30 | quad+point_anomaly | `deterministic_trend + point_anomaly` | 200 | 200 | 0 | 0 | 100.00 |
| 31 | quad+variance_shift | `deterministic_trend + variance_shift` | 100 | 94 | 6 | 0 | 94.00 |
| 32 | stoch+collective | `stochastic_trend + collective_anomaly` | 100 | 87 | 13 | 0 | 87.00 |
| 33 | stoch+mean_shift | `stochastic_trend + mean_shift` | 100 | 77 | 23 | 0 | 77.00 |
| 34 | stoch+point_anomaly | `stochastic_trend + point_anomaly` | 100 | 91 | 9 | 0 | 91.00 |
| 35 | stoch+variance_shift | `stochastic_trend + variance_shift` | 3,500 | 3,210 | 286 | 4 | 91.71 |
| 36 | vol+collective | `volatility + collective_anomaly` | 100 | 63 | 37 | 0 | 63.00 |
| 37 | vol+mean_shift | `volatility + mean_shift` | 100 | 86 | 14 | 0 | 86.00 |
| 38 | vol+point_anomaly | `volatility + point_anomaly` | 100 | 78 | 22 | 0 | 78.00 |
| 39 | vol+variance_shift | `volatility + variance_shift` | 100 | 77 | 23 | 0 | 77.00 |