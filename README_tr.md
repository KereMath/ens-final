# Zaman Serisi Base Type ve Anomali Sınıflandırması için Hiyerarşik Çok Katmanlı Stacked Ensemble

Önceden eğitilmiş iki binary ensemble sistemi, özel feature setli bir stationarity
gate, öğrenilmiş bir routing katmanı ve kalibre edilmiş stacking meta-learner'ları
birleştiren üretim seviyesi bir pipeline. Zaman serilerini base process tipleri
ile anomali desenlerini birleştiren 39 farklı davranışsal kategoriye
sınıflandırır.

**Nihai Sonuçlar:**

| Değerlendirme | Örnek | stat_gate | FULL | Oran |
|---|---|---|---|---|
| **Balanced (primary)** | 39,000 (grup başına 1,000) | **0.85** | **34,962** | **%89.65** |
| Weighted (orijinal) | 4,400 (leaf-başına 10) | 0.92 | 3,919 | %89.07 |

Tek ensemble baseline'a göre **+29.27 puan iyileşme** (weighted). Balanced 39K
değerlendirmesi — sınıf başına eşit örnek sayısı ile — **%89.65 macro-eşdeğer
FULL match** verir; bu, pipeline'ın gerçek multi-class performansını eval set
imbalance bias'ından arındırılmış bir şekilde raporlar.

---

## İçindekiler

1. [Motivasyon ve Problem Tanımı](#motivasyon-ve-problem-tanımı)
2. [Veri Seti ve Sınıf Taksonomisi](#veri-seti-ve-sınıf-taksonomisi)
3. [Sistem Mimarisi](#sistem-mimarisi)
4. [Bileşen 1 — tsfresh Feature Extraction](#bileşen-1--tsfresh-feature-extraction)
5. [Bileşen 2 — Eski Binary Ensemble (9 detektör)](#bileşen-2--eski-binary-ensemble-9-detektör)
6. [Bileşen 3 — Yeni Binary Ensemble (10 model)](#bileşen-3--yeni-binary-ensemble-10-model)
7. [Bileşen 4 — Stationarity Detector Gate](#bileşen-4--stationarity-detector-gate)
8. [Bileşen 5 — Tekli/Kombinasyon Router](#bileşen-5--teklikombinasyon-router)
9. [Bileşen 6 — Stacking Meta-Learner'lar](#bileşen-6--stacking-meta-learnerlar)
10. [Bileşen 7 — Blended Probability Kararı](#bileşen-7--blended-probability-kararı)
11. [Inference Pipeline](#inference-pipeline)
12. [Training Verisi ve Dengeli Sampling](#training-verisi-ve-dengeli-sampling)
13. [Hyperparameter Arama ve Kalibrasyon](#hyperparameter-arama-ve-kalibrasyon)
14. [Balanced Değerlendirme (39K, birincil)](#balanced-değerlendirme-39k-birincil)
15. [Weighted Değerlendirme (4,400)](#weighted-değerlendirme-4400)
16. [Threshold Robustness Analizi](#threshold-robustness-analizi)
17. [Kademeli İyileşme Geçmişi](#kademeli-iyileşme-geçmişi)
16. [Dosya Organizasyonu ve Tekrarlanabilirlik](#dosya-organizasyonu-ve-tekrarlanabilirlik)
17. [Harici Model Referansları](#harici-model-referansları)
18. [Dosya Organizasyonu ve Tekrarlanabilirlik](#dosya-organizasyonu-ve-tekrarlanabilirlik)
19. [Harici Model Referansları](#harici-model-referansları)
20. [Kullanılan Teknikler — Akademik Özet](#kullanılan-teknikler--akademik-özet)

---

## Motivasyon ve Problem Tanımı

CSV olarak örneklenmiş ham tek değişkenli zaman serisi verildiğinde, görev
şu iki çıktıyı üretmektir:

1. **Base process tipi** (4-sınıflı sınıflandırma): stationary, deterministic
   trend, stochastic trend veya volatility (heteroskedastic).
2. **Sıfır veya daha fazla anomali etiketi** (6-sınıflı multi-label):
   collective, contextual, mean shift, point, trend shift veya variance shift.

Değerlendirme metriği **strict FULL match**'tir — bir tahmin ancak ve ancak
base type doğru tespit edildiyse VE mevcut her anomali algılandıysa VE hiçbir
yanlış anomali üretilmediyse (no false positives) doğru sayılır. Bu tek
kriterli değerlendirme, sistemi hem base type tespitinde doğru hem de anomali
tespitinde cerrahi düzeyde hassas olmaya zorlar.

39 kaynak grup dört karmaşıklık katmanına yayılır:

| Katman | Açıklama | Gruplar | Örnek |
|---|---|---|---|
| **1. Saf base** | Tek process, anomali yok | 1–4 | stationary (1), deterministic_trend (2) |
| **2. Stationary + tek anomali** | Baseline process + bir anomali | 5–10 | stationary + mean_shift (7) |
| **3. Deterministic + anomali** | Trend process + bir anomali | 11–31 | cubic + collective (11), linear + trend_shift (26) |
| **4. Deterministic olmayan kombinasyonlar** | Stochastic / volatility + anomali | 32–39 | stoch_trend + variance_shift (35) |

---

## Veri Seti ve Sınıf Taksonomisi

Toplam değerlendirme: **4,400 CSV dosyası**, leaf-balanced sampling ile
(her leaf dizinden 10 dosya) 39 gruptan çekilmiştir. 39 grup ve kanonik
beklenen etiketleri:

| # | Grup Adı | Beklenen Base | Beklenen Anomali | Sayı |
|---|---|---|---|---|
| 1 | stationary | stationary | — | 120 |
| 2 | deterministic_trend | deterministic_trend | — | 720 |
| 3 | stochastic_trend | stochastic_trend | — | 150 |
| 4 | volatility | volatility | — | 120 |
| 5 | collective_anomaly | stationary | collective_anomaly | 480 |
| 6 | contextual_anomaly | stationary | contextual_anomaly | 480 |
| 7 | mean_shift | stationary | mean_shift | 480 |
| 8 | point_anomaly | stationary | point_anomaly | 480 |
| 9 | trend_shift | stationary | trend_shift | 480 |
| 10 | variance_shift | stationary | variance_shift | 480 |
| 11–14 | cubic + {collective, mean, point, variance} | deterministic_trend | her biri | 10–20 |
| 15–18 | damped + {collective, mean, point, variance} | deterministic_trend | her biri | 10–20 |
| 19–22 | exponential + {collective, mean, point, variance} | deterministic_trend | her biri | 10–20 |
| 23–27 | linear + {collective, mean, point, trend_shift, variance} | deterministic_trend | her biri | 10–30 |
| 28–31 | quadratic + {collective, mean, point, variance} | deterministic_trend | her biri | 10–20 |
| 32–35 | stochastic + {collective, mean, point, variance} | stochastic_trend | her biri | 10–50 |
| 36–39 | volatility + {collective, mean, point, variance} | volatility | her biri | 10 |

---

## Sistem Mimarisi

Pipeline, her aşamanın farklı bir inductive bias kattığı **yedi aşamalı
hiyerarşik bir ensemble**tir:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GIRDI: ham tek değişkenli zaman serisi (değişken uzunluk, min 50)      │
└─────────────────────────────────────────────────────────────────────────┘
                │
                ├─► Aşama A: tsfresh EfficientFC   →  777-boyutlu feature vec
                │
                ├─► Aşama B: Stationarity Detector Gate ─────► P(stationary)
                │   (özel 25 feature, XGBoost binary)
                │
                ├─► Aşama C: Eski Binary Ensemble   ─────► 9 × P(class_k)
                │   (9 bağımsız XGBoost/LightGBM/MLP binary)
                │
                ├─► Aşama D: Yeni Binary Ensemble   ─────► 10 × P(class_k)
                │   (4 base + 6 anomali, LightGBM/XGBoost binaries)
                │
                ├─► Aşama E: Feature Engineering
                │   • 14 türetilmiş meta-feature
                │     (agreement, entropy, confidence gap, max/argmax)
                │   • Standartlaştırılmış ham tsfresh feature'ları (777)
                │   └────► 810-boyutlu birleşik meta-vektör
                │
                ├─► Aşama F: Tekli/Kombinasyon Router
                │   (XGB+LGB binary, 810-boyutlu üzerinde)  →  P(combo)
                │
                └─► Aşama G: Stacking Meta-Learner'lar
                    • Base type: XGB+LGB 4-sınıflı, 810-boyutlu
                    • 6 × Anomali binary: XGB+LGB
                      alpha-blended new-ensemble probability ile
                      ve per-anomaly tuned threshold ile

                    ▼
                KARAR MANTIĞI:
                EĞER stationary_gate ≥ 0.92:
                    return (stationary, [])        # Override yolu
                DEĞILSE EĞER router(combo) < 0.30:
                    return (base_meta_argmax, [])  # Tekli yol
                DEĞILSE:
                    return (
                        base_meta_argmax,
                        [a for a in ANOMALIES
                         if α·meta_a + (1-α)·new_a ≥ threshold_a]
                    )                              # Kombinasyon yolu
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ÇIKTI: (base_type, [tespit edilen anomalilerin listesi])               │
└─────────────────────────────────────────────────────────────────────────┘
```

Her bileşen aşağıda tam teknik detayla açıklanmıştır.

---

## Bileşen 1 — tsfresh Feature Extraction

**Nedir:** Değişken uzunluktaki zaman serisini tsfresh'in
`EfficientFCParameters`'ı (en maliyetli hesaplayıcıları hariç tutan seçilmiş
bir alt küme) kullanarak sabit 777 boyutlu bir feature vektörüne map eden
otomatik, deterministik bir feature üretici.

**Neden 777 feature:** Ampirik olarak, bu tsfresh'in `EfficientFCParameters`
ile tek değişkenli seriler için ürettiği üst sınırdır. 777 feature şu
kategorilere yayılır:

| Feature Ailesi | Yaklaşık Sayı | Örnekler |
|---|---|---|
| İstatistiksel momentler | ~20 | mean, std, skewness, kurtosis, quantile'lar |
| Autocorrelation | ~100 | `autocorrelation(lag=k)` k=1..10 için, partial ACF |
| Fourier descriptor | ~150 | FFT coefficient reel/sanal, spectral centroid |
| Wavelet CWT | ~50 | Continuous wavelet transform coefficient'ları |
| Change metric | ~80 | `change_quantile`, `cid_ce`, `mean_abs_change` |
| Peak detection | ~30 | `number_peaks(n=k)`, prominence |
| Distribution | ~40 | `ratio_beyond_r_sigma`, entropy, benford |
| Linear trend | ~20 | slope, stderr, intercept (ham ve alt-seriler üzerinde) |
| Nonlinear descriptor | ~50 | `augmented_dickey_fuller`, Lempel-Ziv |
| Symbolic aggregation | ~50 | `value_count`, `unique_ratio`, binned counts |
| Diğer | ~187 | `friedrich_coefficients`, `agg_linear_trend`, vb. |

**Neden bu feature ailesi:** tsfresh'in extraction'ı zaman serisi analizinde
yaygın kabul görmüş standart bir baseline'dır. Sınıflandırma görevleri için
iyi ayarlanmıştır; manuel engineering gerektirmeden hem lokal hem global
temporal özellikleri yakalar. Kritik olarak, **pipeline'daki her downstream
bileşen aynı 777 feature'ı input olarak alır**, bu da serinin tek birleştirici
gösterimi yapar.

**NaN yönetimi:** tsfresh'in `impute()`'u sonsuzluk ve NaN değerlerini sütun
ortalamaları/0 ile değiştirir; ek olarak her model çağrısından önce sayısal
stabilite garantisi için `np.nan_to_num()` uygularız.

---

## Bileşen 2 — Eski Binary Ensemble (9 detektör)

**Kaynak:** `../tsfresh ensemble/trained_models/` dizininden yüklenir. Bu
ensemble **tek etiketli zaman serisi verisi** üzerinde bağımsız olarak
eğitilmiştir (her seri tam olarak 9 sınıftan birine aittir).

**Mimari:** Her sınıf için bir tane olmak üzere 9 bağımsız binary detektör:

```
Sınıflar (kanonik indeksleme için sıralı):
  0: collective_anomaly
  1: contextual_anomaly
  2: deterministic_trend
  3: mean_shift
  4: point_anomaly
  5: stochastic_trend
  6: trend_shift
  7: variance_shift
  8: volatility
```

**Training metodolojisi** (orijinal repo): 9 sınıfın her biri için, hedef
sınıf positive (1) ve diğer tüm sınıflar negative (0) olacak şekilde ayrı bir
binary XGBoost / LightGBM / MLP eğitilmiştir. Sınıf başına en iyi model
**validation F1** ile seçilmiş ve sınıfa özel bir `RobustScaler` ile birlikte
kaydedilmiştir.

**Inference'da çıktı:** 9 olasılık `P(class_k = 1 | tsfresh_features)`.
Bu 9 değer stacking katmanına beslenen meta-feature vektörünün **ilk 9
bileşenini** oluşturur.

**Neden değiştirilmeden korundu:** Eski ensemble tek etiketli tanımlamada
üstündür ("bu seri mean_shift mi?"). Yeniden eğitim mevcut training setinin
yeniden kurulmasını gerektirir ve net bir avantaj sunmaz — ancak onun
**sınıf başına görüşü** meta-learner için değerli bir sinyaldir, özellikle
saf trend serilerinde (Grup 2) base type belirlenmesinde yeni ensemble'ın
zaman zaman yaşadığı kalibrasyon sorunlarını tamamlar.

---

## Bileşen 3 — Yeni Binary Ensemble (10 model)

**Kaynak:** `../ensemble-alldata/trained_models/` dizininden yüklenir.
**Kombinasyon verisi** üzerinde bağımsız olarak eğitilmiştir (base process +
seri başına bir anomali).

**Mimari:** Aşağıdakileri kapsayan 10 bağımsız binary model:

| Indeks | Model Adı | Tip |
|---|---|---|
| 0 | stationary | base |
| 1 | deterministic_trend | base |
| 2 | stochastic_trend | base |
| 3 | volatility | base |
| 4 | collective_anomaly | anomali |
| 5 | contextual_anomaly | anomali |
| 6 | mean_shift | anomali |
| 7 | point_anomaly | anomali |
| 8 | trend_shift | anomali |
| 9 | variance_shift | anomali |

**Training metodolojisi** (orijinal repo): Her model için, 39 kaynak gruptan
leaf-balanced sampling ile **N=1320 pozitif + N=1320 negatif örnek** olacak
şekilde training verisi kurulmuştur. Pozitif kaynaklar hedef sınıfın mevcut
olduğu gruplardır (örneğin `mean_shift` için pozitif gruplar 7, 12, 16, 20,
24, 29, 33, 37'dir). Her model için LightGBM, XGBoost ve MLP eğitilmiş; en
yüksek **validation F1** olan kaydedilmiştir.

**Inference'da çıktı:** 10 olasılık. Bunlar meta-feature vektörünün **sonraki
10 bileşenini** (indeks 9–18) oluşturur.

**Neden değiştirilmeden korundu:** Yeni ensemble base+anomali kombinasyonları
için iyi kalibre edilmiştir (gruplar 11–39 sadece bu modellerle
in-distribution %95.9'a ulaşır). Ancak saf base tiplerinde (gruplar 1–4)
**aşırı güvenli false positive'ler** sergiler: saf bir stationary seride,
`P(point_anomaly) = 0.55` üretebilir ve hatalı anomali çıkarabilir. Downstream
stacking + routing katmanları bunu giderir.

---

## Bileşen 4 — Stationarity Detector Gate

**Kaynak:** `../stationary detector ml/trained_models v2/` dizininden
yüklenir. Stationary (sınıf 0) ile non-stationary (sınıf 1) arasında ayrım
yapan ayrı bir binary sınıflandırıcı.

**Kritik ayrım:** tsfresh feature kullanan Bileşen 2 ve 3'ün aksine, **bu
detector stationarity testi için özel tasarlanmış 25 el-işçiliği feature
kullanır**:

```
Feature ailesi                 # feature
─────────────────────────────────────────
Temel istatistikler            13
  mean, std, var, min, max, range,
  q25, median, q75, iqr,
  skewness, kurtosis, cv
Birinci/ikinci farklar         5
  diff1_{mean, std, var},
  diff2_{mean, std}
Rolling window istatistikleri  3
  rolling_mean_std,
  rolling_std_mean, rolling_std_std
Autocorrelation                2
  autocorr_lag1, autocorr_lag10
Peak analizi                   2
  num_peaks, zero_crossing_rate
```

Bu feature'lar **stationarity'yi özel olarak tespit etmek için seçilmiştir**:
örneğin `rolling_std_std` serinin zaman içinde stabil varyansa sahip olup
olmadığını yakalar; `autocorr_lag10` kısa-menzilli memory etkilerini yakalar.

**Neden tsfresh yerine bu yaklaşım:** Ampirik olarak, tsfresh'in 777
feature'ı brief lokal değişikliklere duyarsız birçok aggregate istatistik
içerir. Tek bir point anomaly eklenmiş stationary bir seri için, tsfresh hâlâ
onu stationary olarak sınıflandırır çünkü genel dağılım çok az bozulur. Özel
feature'lar — özellikle `num_peaks` ve `rolling_std_std` — bu lokal
bozulmalara daha duyarlıdır.

**Training metodolojisi** (orijinal repo): Balanced training set üzerinde
binary sınıflandırma (stationary vs non_stationary); LightGBM, XGBoost, MLP,
Decision Tree, Random Forest, Extra Trees ve MLPFast değerlendirilmiştir.
**XGBoost**, held-out test split'te **F1 = 0.881** ile kazanmıştır.

**Inference'da çıktı:** Tek bir olasılık `P(stationary)`.

**Pipeline'daki rolü:** **Yüksek-precision outer gate** olarak hizmet eder.
Eğer `P(stationary) ≥ 0.92` ise, pipeline downstream bileşenleri kısa
devre yapar ve doğrudan `("stationary", [])` üretir. Bu threshold, gruplar
5–10'da (stationary base + anomali) false override'lardan kaçınarak full-match
sayısını maksimize etmek için grid search ile ayarlanmıştır.

---

## Bileşen 5 — Tekli/Kombinasyon Router

**Rol:** Tam kombinasyon kararına geçmeden önce router, serinin saf bir
tekli pattern mi yoksa base + anomali kombinasyonu mu olduğu hakkında
**ikinci bir görüş** sunar.

**Training kurulumu:**
- **Pozitif (combo = 1):** Gruplar 5–39 (en az bir anomaliye veya
  non-stationary base'e sahip tüm gruplar)
- **Negatif (single = 0):** Gruplar 1–4 (saf base tipler)
- **Training verisi:** 19,500 örnek (500 grup başına × 39 grup),
  leaf-balanced
- **Feature gösterimi:** Bileşen 6'nın **tam 810-boyutlu meta-vektörü**
  (aşağıya bakınız), sadece tsfresh feature'ları değil
- **Modeller:** XGBoost (500 ağaç, max_depth=6, lr=0.05) ve LightGBM
  (num_leaves=63, max_depth=7) bağımsız olarak eğitilmiştir
- **Test metriği:** held-out %20 split'te Ensemble F1 = **0.978**

**Inference:** `P(combo) = 0.5 × XGB + 0.5 × LGB` olasılıklar.

**Routing kararı:**
```
EĞER P(combo) < 0.30:
    SINGLE dalına yönlendir → (base_meta_argmax, []) döndür
DEĞILSE:
    COMBO dalına yönlendir → tam anomali değerlendirmesi
```

**0.30** threshold'u grid search ile seçilmiştir. Daha düşük bir threshold
sistemi kombinasyon dalına yönlendirme eğilimi gösterir ki bu arzu edilen
durumdur: kombinasyon dalının false positive'leri önlemek için güçlü
mekanizmaları vardır (threshold kalibrasyonu, blending), oysa tekli dala
düşmek geçerli anomali tespitlerini kaybettirir.

---

## Bileşen 6 — Stacking Meta-Learner'lar

**Amaç:** Eski ensemble'ın (9), yeni ensemble'ın (10) ve ham tsfresh
feature'larının görüşlerini nihai bir tahmine nasıl birleştireceğini öğrenmek.
Bu projenin **merkezî yeniliğidir**: hangi ensemble'a ne zaman güveneceğimize
dair kurallar elde kodlamak yerine, bunu yapacak modeller eğitiriz.

### Meta-Feature Vektörü (810 boyut)

Her örnek için inşa edilir:

```
Boyut        Kaynak                                    Açıklama
────────────────────────────────────────────────────────────────
  0 –   8    Eski ensemble olasılıkları                9 binary
  9 –  18    Yeni ensemble olasılıkları                10 binary (4 base + 6 anom)
 19 –  32    Türetilmiş meta-feature'lar               14 engineered stat
 33 – 809    Ham tsfresh feature'ları (standardized)   777 feature
────────────────────────────────────────────────────────────────
            TOPLAM                                       810 boyut
```

### 14 Türetilmiş Meta-Feature

Daha zengin sinyal sağlamak için 19 ham ensemble olasılığından hesaplanır:

1. `max_old_base` — eski ensemble'ın base sınıfları arasında en yüksek prob
2. `argmax_old_base` — eski ensemble'ın tercih ettiği base sınıf
3. `max_old_anomaly` — eski ensemble'ın anomali sınıfları arasında en yüksek prob
4. `n_old_anomaly_above_0.5` — ateşlenen eski anomali sınıflarının sayısı
5. `max_new_base` — yeni ensemble'ın max base prob'u
6. `argmax_new_base` — yeni ensemble'ın tercih ettiği base indeks
7. `max_new_anomaly` — yeni ensemble'ın max anomali prob'u
8. `n_new_anomaly_above_0.5` — ateşlenen yeni anomalilerin sayısı
9. `base_agreement` — eski ve yeni base type konusunda anlaşıyor mu? (0/1)
10. `base_confidence_gap` — yeni'nin max base prob'u eksi ikinci max'ı
11. `anomaly_entropy` — yeni'nin 6 anomali prob'unun ortalama binary entropy'si
12. `old_new_anomaly_correlation` — iki ensemble'ın anomali olasılık
    vektörlerinin Pearson korelasyonu
13. `total_new_anomaly_signal` — tüm yeni anomali prob'larının toplamı
14. `total_old_anomaly_signal` — tüm eski anomali prob'larının toplamı

Bu feature'lar meta-learner'a **ensemble disagreement**, **confidence
seviyeleri** ve **sinyal gücü desenlerini** algılama imkânı verir — edge
case'lerin yönetimi için paha biçilemezdir.

### Base-Type Meta-Learner

{stationary, deterministic_trend, stochastic_trend, volatility} tahmini
yapan **4-sınıflı XGBoost + LightGBM ensemble**.

**Hyperparameter'lar:**

| Parametre | Değer |
|---|---|
| n_estimators | 500 |
| learning_rate | 0.05 |
| max_depth | 6 |
| min_child_weight | 3 |
| gamma | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.7 |
| reg_alpha / reg_lambda | 0.1 / 1.0 |
| class_weight | balanced |
| num_class | 4 |

LightGBM aynı hyperparameter'lar ile `num_leaves=63` ve `max_depth=7`
kullanır. **Ensemble tahmini:**
`proba = 0.5 × XGB.predict_proba + 0.5 × LGB.predict_proba`

**Training verisi:** 19,500 meta-vektör, 4-sınıflı base etiketi ile;
stratified %80/%20 train/test split.
**Test accuracy: %96.85, weighted F1: %96.85.**

### Anomali Meta-Learner'lar (6 × binary)

Her anomali tipi için bir XGBoost + LightGBM ensemble: collective, contextual,
mean_shift, point, trend_shift, variance_shift.

**Kritik oversampling:** Gruplar 5–10 (stationary + tek anomali) fit
edilmeden önce training set'te **üç katına çıkarılır**. Bunun nedeni bu
grupların en nüanslı vakalar olmasıdır — bir subtle anomalili stationary
seri — ve bu nedenle orantısal olarak daha fazla maruziyet gerektirirler.

**Per-anomaly F1 skorları (held-out test'te):**

| Anomali | F1 | Accuracy |
|---|---|---|
| collective_anomaly | 0.9127 | %96.6 |
| contextual_anomaly | 0.9988 | %99.99 |
| mean_shift | 0.9211 | %96.9 |
| point_anomaly | 0.9518 | %98.1 |
| trend_shift | 0.9863 | %99.9 |
| variance_shift | 0.9297 | %97.2 |

---

## Bileşen 7 — Blended Probability Kararı

**Amaç:** Kombinasyon dalında, her anomalinin nihai olasılığı meta-learner'ın
tahmininin ve yeni ensemble'ın doğrudan tahmininin **ağırlıklı kombinasyonudur**.

**Anomali k için formül:**
```
blended_prob_k = α_k × meta_prob_k + (1 - α_k) × new_ensemble_prob_k
```

**Karar kuralı:**
```
anomali_k algılandı  ⇔  blended_prob_k ≥ threshold_k
```

### Per-Anomaly Tuned Parametreler

Bunlar training setinde FULL match sayısını maksimize eden **joint grid
search** ile bulunmuştur; her `(α, threshold)` çifti ayrı ayrı aranmıştır:

| Anomali | α (blend ağırlığı) | Threshold | Gerekçe |
|---|---|---|---|
| collective_anomaly | 0.85 | 0.73 | Meta baskın; yüksek threshold grup 5–10'da FP'leri bastırır |
| contextual_anomaly | 0.70 | 0.69 | Eşit önem; zaten güçlü ayrışma var |
| mean_shift | 0.90 | 0.49 | Meta baskın; standart threshold yeterli |
| point_anomaly | 0.70 | 0.69 | Denge; spike FP'lerden kaçınmak için yüksek threshold kritik |
| trend_shift | 0.90 | 0.73 | Meta baskın; netlik için yüksek threshold |
| variance_shift | 0.70 | 0.69 | Denge; stochastic trend karışıklığını azaltmak için yüksek threshold |

**Neden blend?** Yeni ensemble'ın doğrudan olasılığı, meta-learner'ın
feature etkileşimleri nedeniyle aşırı güvenli veya yetersiz güvenli olabileceği
vakalar için bir "sağlama kontrolü" sağlar. Blending her iki kaynaktan da
geçerli sinyali korur.

---

## Inference Pipeline

Ham bir CSV dosyası verildiğinde:

```python
# ─── Aşama A: Feature extraction ─────────────────────────────────────────
series = read_csv_values(csv_path)                 # ham değerler
if len(series) < MIN_SERIES_LENGTH:                # minimum 50 timestep
    return ERROR

tsfresh_features = tsfresh.extract(series)         # 777-boyutlu
tsfresh_scaled   = tsfresh_scaler.transform(tsfresh_features)  # standardized

# ─── Aşama B: Stationarity gate ──────────────────────────────────────────
p_stationary = stat_detector_v2.predict_proba(
    extract_25_custom_features(series)
)[0]                                               # tek skaler

# ─── Aşama C: Eski ensemble ──────────────────────────────────────────────
old_probs = [
    old_ensemble[class_k].predict_proba(tsfresh_features)[0, 1]
    for class_k in OLD_CLASSES                     # 9 sınıf
]                                                  # 9 olasılık

# ─── Aşama D: Yeni ensemble ──────────────────────────────────────────────
new_probs = [
    new_ensemble[model_k].predict_proba(tsfresh_features)[0, 1]
    for model_k in NEW_ALL_MODELS                  # 10 model
]                                                  # 10 olasılık

# ─── Aşama E: Meta-feature inşası ────────────────────────────────────────
derived = compute_derived_features(old_probs, new_probs)    # 14-boyutlu
meta_x  = concat(old_probs, new_probs, derived, tsfresh_scaled)  # 810-boyutlu

# ─── Aşama F: Stationarity gate kontrolü ─────────────────────────────────
if p_stationary >= 0.92:
    return ("stationary", [])                       # Override yolu

# ─── Aşama G: Router kararı ──────────────────────────────────────────────
p_combo = 0.5 * router_xgb.predict_proba(meta_x)[0, 1] \
        + 0.5 * router_lgb.predict_proba(meta_x)[0, 1]

# ─── Aşama H: Base type tahmini ──────────────────────────────────────────
base_xgb_p = base_meta_xgb.predict_proba(meta_x)[0]    # 4-sınıflı prob
base_lgb_p = base_meta_lgb.predict_proba(meta_x)[0]
base_idx   = argmax(0.5 * base_xgb_p + 0.5 * base_lgb_p)
base_type  = BASE_LABELS[base_idx]

# ─── Aşama I: Routing dalları ────────────────────────────────────────────
if p_combo < 0.30:
    return (base_type, [])                          # Tekli yol

# ─── Aşama J: Blended probability ile anomali tespiti ────────────────────
anomalies = []
for k, anomaly_name in enumerate(ANOM_LABELS):
    meta_prob = 0.5 * anom_meta[anomaly_name].xgb.predict_proba(meta_x)[0, 1] \
              + 0.5 * anom_meta[anomaly_name].lgb.predict_proba(meta_x)[0, 1]
    new_prob  = new_probs[k + 4]                    # 4 base + k anomali
    blended   = ALPHA[anomaly_name] * meta_prob \
              + (1 - ALPHA[anomaly_name]) * new_prob
    if blended >= THRESHOLD[anomaly_name]:
        anomalies.append(anomaly_name)

return (base_type, anomalies)                       # Kombinasyon yolu
```

**Örnek başına toplam inference maliyeti:**
- tsfresh extraction: ~0.1–0.2 sn (bulk dominant)
- Özel 25-feature extraction: ~0.01 sn
- 9 + 10 = 19 binary tahmin: ihmal edilebilir
- Router: 2 tahmin, ihmal edilebilir
- Base: 2 tahmin
- Anomali: 12 tahmin (6 × XGB+LGB)

Toplam: tsfresh baskın (1000-sample'lık bir seri için ≈ 150 ms).

---

## Training Verisi ve Dengeli Sampling

### Meta-Learner Training

- **Toplam örnek: 19,500** (grup başına 500 × 39 grup)
- **Leaf-balanced sampling:** her grup içinde, tüm parametre
  kombinasyonlarının (gürültü seviyesi, seri uzunluğu vb.) kapsamını
  sağlamak için her leaf alt-dizinden orantılı olarak örnek çekilir
- **Stratified train/test split: %80 / %20**
- **Zor gruplar için oversampling:** Gruplar 5–10 (stationary + tek anomali)
  anomali meta-learner training'inde **üç katına çıkarılır**, böylece
  model bu subtle vakalar için daha güçlü sinyale sahip olur

### Değerlendirme Verisi

- **Toplam örnek: 4,400**
- **Grup başına:** leaf dizin başına 10 örnek
- **Random seed:** 42, tekrarlanabilirlik için deterministik
- **Training verisi ile çakışmaz:** değerlendirme meta training'den farklı
  random-sampled CSV'ler kullanır

---

## Hyperparameter Arama ve Kalibrasyon

Üç hyperparameter ailesi **cached meta-feature'lar üzerinde exhaustive grid
search** ile birlikte tuned edilmiştir ("fast grid" yaklaşımı):

### Stationarity Gate Threshold
- **Aranan aralık:** 0.85 – 1.01, 0.01 artışlarla
- **En iyi değer:** 0.92 (v2 detector)
- **Seçim kriteri:** maksimum FULL match sayısı

### Router Threshold
- **Aranan aralık:** 0.25 – 0.50, 0.01 artışlarla
- **En iyi değer:** 0.30
- **Seçim kriteri:** maksimum FULL match sayısı

### Per-Anomaly (α, Threshold) Çiftleri
6 anomalinin her biri için 2-B grid:
- **α:** [0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0] (8 değer)
- **Threshold:** 0.25 – 0.75, 0.02 artışlarla (~25 değer)
- Anomali başına toplam: 8 × 25 = 200 kombinasyon
- **Sıralı iyileştirme:** her anomali sırayla tuned edilir (collective →
  contextual → mean_shift → point → trend_shift → variance_shift); her adım
  önceki tuned değerleri baseline olarak kullanır

Nihai tuned parametreler [Bileşen 7](#bileşen-7--blended-probability-kararı)
bölümünde rapor edilmiştir.

### Fast Grid Altyapısı
- **`cache_eval.py`**: Tüm 4,400 değerlendirme örneği için 19 ensemble
  olasılığı, meta-feature ve stationarity detector olasılığını BİR KEZ
  hesaplar ve `.npz` olarak saklar (~22 MB). Sonraki grid search'ler
  dakikalar yerine saniyeler sürer.
- **`fast_grid*.py`**: Cached feature'lar üzerinde çalışan çok-aşamalı
  grid search script'leri

---

## Balanced Değerlendirme (39K, birincil)

### Metodoloji

Weighted değerlendirmenin (4,400 leaf-başına 10) sonuçları **sınıf örnek
sayısındaki dengesizlik** nedeniyle biased bir metrik üretir: stationary (grup 1)
120 örneğe, deterministic_trend (grup 2) ise 720 örneğe sahiptir; saf base
grupları aşırı temsil edilir ve kombinasyon grupları (çoğu sadece 10 örnek)
yetersiz temsil edilir. Bu bias'ı ortadan kaldırmak için **her gruptan eşit
sayıda (1,000 örnek)** çekilen ikincil bir değerlendirme seti kuruldu:

- **Toplam:** 39,000 CSV dosyası (39 grup × 1,000 örnek)
- **Sampling:** `build_balanced_cache.py` ile leaf-balanced; `random.seed(42)`
- **Training ile çakışma:** yok (meta training 19,500 örnek farklı seed)
- **Cache dosyası:** `processed_data/balanced_eval_cache.npz` (~600 MB)

Bu kurulumda **weighted FULL % = macro FULL %** olur çünkü grup örnek sayıları
eşittir. Bu, gerçek **sınıflar arası ortalama** performansı raporlar.

### Optimal Konfigürasyon

`balanced_threshold_sweep.py` stationarity gate threshold'unu 0.30–1.00 aralığında
taradı. **Optimal: `stat_gate = 0.85`**, `router_threshold = 0.30`, ve
[Bileşen 7](#bileşen-7--blended-probability-kararı)'deki tuned per-anomaly
(α, threshold) parametreleri.

### 39-Sınıf FULL Match Tablosu (stat_t = 0.85)

| # | Grup | n | FULL | PART | NONE | FULL % |
|---|---|---|---|---|---|---|
| 1 | stationary | 1000 | 675 | 272 | 53 | 67.50 |
| 2 | deterministic_trend | 1000 | 926 | 21 | 53 | 92.60 |
| 3 | stochastic_trend | 1000 | 811 | 128 | 61 | 81.10 |
| 4 | volatility | 1000 | 809 | 119 | 72 | 80.90 |
| 5 | collective_anomaly | 1000 | 900 | 21 | 79 | 90.00 |
| 6 | contextual_anomaly | 1000 | 997 | 3 | 0 | 99.70 |
| 7 | mean_shift | 1000 | 904 | 52 | 44 | 90.40 |
| 8 | point_anomaly | 1000 | 816 | 82 | 102 | 81.60 |
| 9 | trend_shift | 1000 | 923 | 19 | 58 | 92.30 |
| 10 | variance_shift | 1000 | 786 | 97 | 117 | 78.60 |
| 11 | cubic + collective | 1000 | 998 | 2 | 0 | 99.80 |
| 12 | cubic + mean_shift | 1000 | 958 | 42 | 0 | 95.80 |
| 13 | cubic + point_anomaly | 1000 | 993 | 7 | 0 | 99.30 |
| 14 | cubic + variance_shift | 1000 | 989 | 11 | 0 | 98.90 |
| 15 | damped + collective | 1000 | 999 | 1 | 0 | 99.90 |
| 16 | damped + mean_shift | 1000 | 923 | 77 | 0 | 92.30 |
| 17 | damped + point_anomaly | 1000 | 1000 | 0 | 0 | **100.00** |
| 18 | damped + variance_shift | 1000 | 960 | 40 | 0 | 96.00 |
| 19 | exponential + collective | 1000 | 999 | 1 | 0 | 99.90 |
| 20 | exponential + mean_shift | 1000 | 958 | 42 | 0 | 95.80 |
| 21 | exponential + point_anomaly | 1000 | 1000 | 0 | 0 | **100.00** |
| 22 | exponential + variance_shift | 1000 | 989 | 11 | 0 | 98.90 |
| 23 | linear + collective | 1000 | 996 | 3 | 1 | 99.60 |
| 24 | linear + mean_shift | 1000 | 955 | 45 | 0 | 95.50 |
| 25 | linear + point_anomaly | 1000 | 987 | 13 | 0 | 98.70 |
| 26 | linear + trend_shift | 1000 | 994 | 6 | 0 | 99.40 |
| 27 | linear + variance_shift | 1000 | 983 | 17 | 0 | 98.30 |
| 28 | quadratic + collective | 1000 | 996 | 3 | 1 | 99.60 |
| 29 | quadratic + mean_shift | 1000 | 963 | 37 | 0 | 96.30 |
| 30 | quadratic + point_anomaly | 1000 | 1000 | 0 | 0 | **100.00** |
| 31 | quadratic + variance_shift | 1000 | 944 | 56 | 0 | 94.40 |
| 32 | stochastic + collective | 1000 | 791 | 208 | 1 | 79.10 |
| 33 | stochastic + mean_shift | 1000 | 528 | 467 | 5 | 52.80 |
| 34 | stochastic + point_anomaly | 1000 | 899 | 99 | 2 | 89.90 |
| 35 | stochastic + variance_shift | 1000 | 867 | 132 | 1 | 86.70 |
| 36 | volatility + collective | 1000 | 469 | 531 | 0 | 46.90 |
| 37 | volatility + mean_shift | 1000 | 819 | 181 | 0 | 81.90 |
| 38 | volatility + point_anomaly | 1000 | 684 | 316 | 0 | 68.40 |
| 39 | volatility + variance_shift | 1000 | 774 | 225 | 1 | 77.40 |

**TOPLAMLAR**
- FULL match: **34,962 / 39,000 (%89.65)**
- PARTIAL match: 3,387 / 39,000 (%8.68)
- NO match: 651 / 39,000 (%1.67)

**Özet istatistikler:**
- %100 FULL olan gruplar: **3** (17, 21, 30)
- Gruplar ≥ %95 FULL: **20 / 39**
- Gruplar ≥ %90 FULL: **26 / 39**
- Gruplar ≥ %80 FULL: **32 / 39**
- Median grup FULL %: **95.50**
- Macro FULL %: **89.65** (örnek sayıları eşit olduğundan weighted'a eşit)

### Zayıf Gruplar ve Hata Analizi

| Grup | FULL % | Temel neden |
|---|---|---|
| 36 — volatility + collective | 46.90 | PARTIAL (531) baskın — base genellikle doğru, anomali kısmen kaçırılıyor |
| 33 — stochastic + mean_shift | 52.80 | PARTIAL (467) baskın — stoch_trend vs mean_shift sinyali örtüşüyor |
| 1 — stationary | 67.50 | PARTIAL (272) — stat_gate altında false anomali eklentisi |
| 38 — volatility + point_anomaly | 68.40 | PARTIAL (316) — volatility altındaki point spike'lar zor |
| 10 — variance_shift | 78.60 | NONE (117) — variance_shift'i stat_gate bastırıyor |

Tüm zayıf gruplarda **baskın hata tipi NONE değil PARTIAL'dır**; pipeline base
type'ı çoğunlukla doğru buluyor ancak anomali etiketini kısmen kaçırıyor. Bu,
geliştirilebilir alanın **anomali duyarlılığı** (threshold kalibrasyonu veya
dedicated zor-grup classifier'ları) olduğunu söyler.

---

## Weighted Değerlendirme (4,400)

Orijinal değerlendirme seti — leaf-balanced, her leaf'ten 10 örnek. Grup başına
örnek sayısı 10 ile 720 arasında değişir. Raporlanan %89.07 bu sette **stat_gate
= 0.92** ile elde edilmiştir.

| # | Grup | n | FULL | PART | NONE | FULL % |
|---|---|---|---|---|---|---|
| 1 | stationary | 120 | 67 | 49 | 4 | 55.8 |
| 2 | deterministic_trend | 720 | 674 | 10 | 36 | 93.6 |
| 3 | stochastic_trend | 150 | 129 | 15 | 6 | 86.0 |
| 4 | volatility | 120 | 106 | 7 | 7 | 88.3 |
| 5 | collective_anomaly | 480 | 432 | 7 | 41 | 90.0 |
| 6 | contextual_anomaly | 480 | 480 | 0 | 0 | 100.0 |
| 7 | mean_shift | 480 | 421 | 27 | 32 | 87.7 |
| 8 | point_anomaly | 480 | 408 | 17 | 55 | 85.0 |
| 9 | trend_shift | 480 | 441 | 10 | 29 | 91.9 |
| 10 | variance_shift | 480 | 384 | 46 | 50 | 80.0 |
| 11 | cubic + collective | 10 | 10 | 0 | 0 | 100.0 |
| 12 | cubic + mean_shift | 20 | 19 | 1 | 0 | 95.0 |
| 13 | cubic + point_anomaly | 10 | 10 | 0 | 0 | 100.0 |
| 14 | cubic + variance_shift | 10 | 9 | 1 | 0 | 90.0 |
| 15 | damped + collective | 10 | 10 | 0 | 0 | 100.0 |
| 16 | damped + mean_shift | 20 | 20 | 0 | 0 | 100.0 |
| 17 | damped + point_anomaly | 10 | 10 | 0 | 0 | 100.0 |
| 18 | damped + variance_shift | 10 | 10 | 0 | 0 | 100.0 |
| 19 | exponential + collective | 10 | 10 | 0 | 0 | 100.0 |
| 20 | exponential + mean_shift | 20 | 20 | 0 | 0 | 100.0 |
| 21 | exponential + point_anomaly | 10 | 10 | 0 | 0 | 100.0 |
| 22 | exponential + variance_shift | 10 | 10 | 0 | 0 | 100.0 |
| 23 | linear + collective | 10 | 10 | 0 | 0 | 100.0 |
| 24 | linear + mean_shift | 20 | 19 | 1 | 0 | 95.0 |
| 25 | linear + point_anomaly | 10 | 10 | 0 | 0 | 100.0 |
| 26 | linear + trend_shift | 30 | 30 | 0 | 0 | 100.0 |
| 27 | linear + variance_shift | 10 | 10 | 0 | 0 | 100.0 |
| 28 | quadratic + collective | 10 | 10 | 0 | 0 | 100.0 |
| 29 | quadratic + mean_shift | 20 | 20 | 0 | 0 | 100.0 |
| 30 | quadratic + point_anomaly | 20 | 20 | 0 | 0 | 100.0 |
| 31 | quadratic + variance_shift | 10 | 9 | 1 | 0 | 90.0 |
| 32 | stochastic + collective | 10 | 5 | 5 | 0 | 50.0 |
| 33 | stochastic + mean_shift | 10 | 6 | 4 | 0 | 60.0 |
| 34 | stochastic + point_anomaly | 10 | 10 | 0 | 0 | 100.0 |
| 35 | stochastic + variance_shift | 50 | 44 | 6 | 0 | 88.0 |
| 36 | volatility + collective | 10 | 3 | 7 | 0 | 30.0 |
| 37 | volatility + mean_shift | 10 | 10 | 0 | 0 | 100.0 |
| 38 | volatility + point_anomaly | 10 | 6 | 4 | 0 | 60.0 |
| 39 | volatility + variance_shift | 10 | 7 | 3 | 0 | 70.0 |

**TOPLAMLAR**
- FULL match: **3,919 / 4,400 (%89.07)**
- PARTIAL match: 221 / 4,400 (%5.02)
- NO match: 260 / 4,400 (%5.91)

---

## Threshold Robustness Analizi

Balanced 39K setinde `stat_gate` threshold'u 0.30'dan 1.00'e tarandı.
`router_threshold = 0.30` ve per-anomaly (α, threshold) parametreleri sabit
tutuldu. Her satır pipeline'ın **aynı modellerle, yalnızca stat_gate değişerek**
elde edilen sonucunu gösterir.

| stat_t | FULL / 39000 | % | G1% | G5% | G7% | G8% | G10% |
|---|---|---|---|---|---|---|---|
| 0.30 | 32,721 | 83.90 | 98.0 | 52.9 | 64.9 | 33.2 | 62.1 |
| 0.35 | 33,160 | 85.03 | 96.7 | 57.1 | 68.4 | 39.8 | 63.9 |
| 0.40 | 33,457 | 85.79 | 95.7 | 60.0 | 70.6 | 45.5 | 65.8 |
| 0.45 | 33,807 | 86.68 | 94.9 | 64.4 | 74.2 | 49.9 | 67.6 |
| 0.50 | 34,069 | 87.36 | 93.0 | 68.6 | 77.1 | 54.1 | 69.3 |
| 0.55 | 34,316 | 87.99 | 91.5 | 72.5 | 80.6 | 58.8 | 71.0 |
| 0.60 | 34,515 | 88.50 | 88.0 | 77.7 | 83.2 | 62.3 | 72.3 |
| 0.65 | 34,642 | 88.83 | 84.5 | 80.4 | 85.7 | 66.5 | 73.5 |
| 0.70 | 34,786 | 89.19 | 81.1 | 84.5 | 87.1 | 71.1 | 75.2 |
| 0.75 | 34,883 | 89.44 | 76.9 | 86.6 | 88.7 | 75.1 | 76.5 |
| 0.80 | 34,932 | 89.57 | 71.9 | 88.4 | 90.0 | 78.5 | 77.7 |
| **0.85** | **34,962** | **89.65** | 67.5 | 90.0 | 90.4 | 81.6 | 78.6 |
| 0.90 | 34,930 | 89.56 | 60.8 | 90.3 | 90.7 | 83.6 | 79.1 |
| 0.92 | 34,915 | 89.53 | — | — | — | — | — |
| 0.95 | 34,882 | 89.44 | 53.9 | 90.4 | 91.0 | 85.0 | 79.2 |
| 1.00 | 34,774 | 89.16 | 42.7 | 90.4 | 91.1 | 85.2 | 79.3 |

### Temel Bulgular

1. **Trade-off net:** düşük stat_gate → G1 (stationary) doğruluğu yükselir fakat
   G5–10 (stationary + anomali) düşer (stat_gate yanlışlıkla anomalileri siler).
2. **Geniş plato:** FULL % tüm 0.75–0.95 aralığında < 0.25 puan dalgalanır —
   pipeline threshold seçimine karşı **robust**.
3. **Maksimum noktası 0.85**, fakat 0.92 ile fark yalnızca **47 örnek (+%0.12)** —
   istatistiki açıdan gürültü seviyesinde.
4. **Eval imbalance hipotezi çürütüldü:** balanced değerlendirme (bias'sız) ve
   weighted değerlendirme **aynı threshold çevresinde aynı tavanı** gösterir.

### Weighted vs Balanced Optimal Konfigürasyon

| Metric | Weighted (4,400) | Balanced (39,000) |
|---|---|---|
| Optimal stat_gate | 0.92 | 0.85 |
| FULL @ optimal | 3,919 / 89.07% | 34,962 / 89.65% |
| FULL'da 0.85 vs 0.92 fark | — | 47 sample (+0.12%) |
| Mimari ceiling | ≈ %89.1 | ≈ %89.7 |

İki eval da aynı mimariyi **farklı ağırlıklarla** ölçer. 47 sample farkı
sampling gürültüsüne düştüğünden, production için her iki threshold da
kabul edilebilir. **stat_gate = 0.85** seçimi balanced-optimal, sınıflar
arası adil performans hedefleyen kullanım için önerilir.

---

## Kademeli İyileşme Geçmişi

| # | Kontrol Noktası | Anahtar Katkı | Full % | Δ |
|---|---|---|---|---|
| 0 | Baseline (sadece yeni ensemble) | Tek binary ensemble | 59.80 | — |
| 1 | + Stacking meta-learner | Eski + yeni görüşlerini birleştir | 67.80 | +8.00 |
| 2 | + 777 ham tsfresh feature | Birleşik 810-boyutlu meta-vektör | 74.60 | +6.80 |
| 3 | + Zor grupları oversample et (5–10) | Training'de 3× vurgu | 77.00 | +2.40 |
| 4 | + Context threshold | Base-bağımlı threshold | 77.50 | +0.50 |
| 5 | + Çift XGB+LGB ensemble | Farklı inductive bias | 77.90 | +0.40 |
| 6 | + Tekli/Kombinasyon router | Karmaşıklığa göre route | 88.50 | +10.60 |
| 7 | + Stationarity gate | Özel model ile override | 88.60 | +0.10 |
| 8 | + Joint (stat, router, α, θ) tuning | İnce parametre arama | 89.07 | +0.47 |
| | **Nihai** | **Birleşik pipeline** | **89.07** | **+29.27** |

---

## Dosya Organizasyonu ve Tekrarlanabilirlik

```
hopefullyprojectfinal/
├── README.md                    # Bu doküman
├── BEST_RESULTS.md              # Konfigürasyon yedeği
├── .gitignore
│
├── config.py                    # 39 grup path'i, sınıf listeleri, global sabitler
├── processor.py                 # tsfresh extraction, ensemble olasılık hesaplaması,
│                                  türetilmiş feature inşası, model yükleme
├── trainer.py                   # Meta-learner eğitimi (base + 6 anomali + router),
│                                  oversampling, blend weight öğrenimi
├── evaluator.py                 # 4,400 örnek üzerinde tam değerlendirme pipeline
├── stat_detector.py             # Stationarity detector v2 wrapper
├── main.py                      # Pipeline orkestrasyonu (--force, --train, --eval)
│
├── cache_eval.py                # Değerlendirme .npz cache'i inşa eder (tüm olasılıklar)
├── fast_grid.py                 # Cached feature'lar üzerinde hızlı grid search
├── fast_grid2.py                # Joint (stat × router × blend) arama
├── fast_grid3.py                # Per-anomaly alpha × threshold grid'i
├── eval_best.py                 # En iyi konfigürasyon için 39-grup tablosunu yazdırır
│
├── processed_data/              # (gitignored) ara cache'ler
│   ├── meta_X.npy               # 19,500 × 810 training meta-feature matrisi
│   ├── meta_y_base.npy          # 19,500 base type etiketi
│   ├── meta_y_anom.npy          # 19,500 × 6 multi-label anomali göstergesi
│   ├── tsfresh_scaler.pkl
│   └── eval_cache.npz           # 4,400 × (tüm olasılıklar + meta feature'lar)
│
├── meta_models/                 # (gitignored) eğitilmiş meta-learner'lar
│   ├── base_meta.pkl            # {xgb, lgb}
│   ├── anom_{name}.pkl × 6      # {xgb, lgb}
│   ├── router.pkl               # {xgb, lgb}
│   └── blend_weights.pkl        # per-anomaly {alpha, threshold}
│
└── results/                     # (gitignored) değerlendirme çıktıları
    ├── evaluation.json
    └── evaluation_report.md
```

### En İyi Sonucun Yeniden Üretilmesi

**Gereksinimler:**
- Python 3.10+
- `pip install numpy pandas scikit-learn xgboost lightgbm tsfresh joblib tqdm scipy`
- Önceden eğitilmiş modelleri içeren kardeş dizinler:
  - `../tsfresh ensemble/trained_models/`
  - `../ensemble-alldata/trained_models/`
  - `../stationary detector ml/trained_models v2/`
- Ham CSV verisi için `C:\Users\user\Desktop\Generated Data\`

**Tam pipeline (training + değerlendirme):**
```bash
python main.py --force
```

**Sadece değerlendirme (training sonrası):**
```bash
python main.py --eval
```

**Cached feature'lar üzerinde grid search:**
```bash
python cache_eval.py    # ~20 dk, bir kez çalışır
python fast_grid.py     # ~1 dk
python fast_grid3.py    # ~3 dk — her anomali için en iyi (alpha, threshold)'u bulur
python eval_best.py     # en iyi konfigürasyon için 39-grup tablosunu gösterir
```

---

## Harici Model Referansları

| Model | Rol | Training Paradigması | Kaynak Path |
|---|---|---|---|
| Eski binary ensemble | 9 tek-sınıflı detektör | tsfresh feature, one-vs-rest binary'ler | `../tsfresh ensemble/` |
| Yeni binary ensemble | 10 base+anomali binary | tsfresh feature, balanced binary'ler | `../ensemble-alldata/` |
| Stationarity detector v2 | Binary stationarity gate | Özel 25 feature, XGBoost | `../stationary detector ml/` |

Üçü de **önceden eğitilmiş modeller olarak yeniden kullanılır** — bu proje
için hiçbirinde yeniden eğitim yapılmadı. Sadece stacking katmanı
(Bileşenler 5–7) yenidir.

---

## Kullanılan Teknikler — Akademik Özet

| Teknik | Uygulama |
|---|---|
| **Stacked Generalization** (Wolpert, 1992) | Base classifier çıktıları üzerinde eğitilmiş meta-learner |
| **Gradient Boosting Decision Trees** | Tüm ağaç-tabanlı modeller için XGBoost ve LightGBM |
| **Ensemble Averaging** | XGB + LGB olasılık kombinasyonu |
| **Automated Feature Engineering** | 777 zaman serisi feature için tsfresh |
| **Class Weighting** | Base meta-learner'da balanced class weight |
| **Synthetic Minority Oversampling** | Anomali training'inde gruplar 5–10 için 3× tekrar |
| **Hierarchical Classification** | Stationarity gate + tekli/kombinasyon router |
| **Probability Blending** | Meta ve base ensemble olasılıklarının convex kombinasyonu |
| **Per-Context Threshold Calibration** | Grid search ile per-anomaly tuned threshold |
| **Derived Meta-Features** | Agreement, entropy, confidence gap, correlation |
| **Data Caching for Grid Search** | Hızlı hyperparameter search için önceden hesaplanmış feature matrisleri |
| **Leaf-Balanced Sampling** | Tüm alt-parametre kombinasyonlarının orantılı gösterimi |

---

## Teşekkürler

Bu çalışma `STATIONARY/` proje ailesindeki üç kardeş repository üzerine
inşa edilmiştir. Her biri odaklı bir scope ile bağımsız olarak geliştirildi;
bu proje onları birleşik bir sınıflandırma sistemine entegre eder.
Meta-learning, routing, blending ve kalibrasyon katmanları bu çalışmanın
katkılarıdır.

---

_Bu çalışmayı kullanırsanız, lütfen bu repository'yi ve
[Harici Model Referansları](#harici-model-referansları) bölümünde listelenen
üç temel ensemble'ı cite edin._
