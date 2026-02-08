# 📊 Podsumowanie eksperymentów ML — Human Activity Recognition (HAR)

## Dane

| Zbiór danych | Rozmiar | Cechy (features) | Klasy |
|---|---|---|---|
| **Combined Dataset** (Kaggle + własne dane) | 7 969 próbek (6 375 train / 1 594 test) | 197 (180 wspólnych z Kaggle) | 6: LAYING, SITTING, STANDING, WALKING, WALKING_DOWNSTAIRS, WALKING_UPSTAIRS |
| **Kaggle (benchmark)** | 10 299 (7 352 train / 2 947 test) | 561 (180 wspólnych) | 6 j.w. |
| **Własne dane (z bazy)** | 631 próbek (504 train / 127 test) | 180 | 5: STANDING, WALKING, WALKING_DOWNSTAIRS, SITTING, CUTOUT |

---

## Porównanie modeli na Combined Dataset (80/20 split)

| Model | Accuracy | CV Mean ± Std (5-fold) | Uwagi |
|---|---|---|---|
| **Random Forest** (100 drzew) | **98%** | 97.28% ± 0.39% | Wysoka stabilność |
| **XGBoost** (domyślne parametry) | **98%** | 98.02% ± 0.28% | Nieco lepsza generalizacja, mniejsze odchylenie |

### Szczegóły per klasa (Combined Dataset)

| Aktywność | RF Precision | XGB Precision | RF Recall | XGB Recall |
|---|---|---|---|---|
| LAYING | 1.00 | 1.00 | 1.00 | 1.00 |
| SITTING | 0.94 | **0.97** | 0.96 | **0.98** |
| STANDING | 0.97 | **0.98** | 0.96 | **0.97** |
| WALKING | 0.97 | 0.97 | **0.99** | 0.98 |
| WALKING_DOWNSTAIRS | 0.99 | 0.98 | 0.97 | **0.99** |
| WALKING_UPSTAIRS | 0.99 | **1.00** | **1.00** | 0.99 |

---

## Test generalizacji — ewaluacja na zewnętrznym zbiorze Kaggle test

| Scenariusz | Accuracy | Features |
|---|---|---|
| **Kaggle-only (baseline, pełne 561 cech)** | **94%** | 561 |
| **Kaggle-only (ograniczone do 180 wspólnych cech)** | **91%** | 180 |
| **Combined Dataset → Kaggle test** | **91%** | 180 |

> ⚠️ Dodanie własnych danych do treningu nie poprawiło wyników na zbiorze Kaggle — accuracy wyniosło tyle samo (91%) co Kaggle z 180 cechami. Spadek z 94% na 91% wynika z redukcji cech z 561 do 180, a nie z dodania własnych danych.

---

## Model produkcyjny (tylko własne dane z bazy — 631 próbek)

| Parametr | Wartość |
|---|---|
| Model | **XGBoost** |
| `n_estimators` | 100 |
| `max_depth` | 6 |
| `learning_rate` | 0.3 |
| `eval_metric` | mlogloss |
| **Accuracy** | **85%** |
| Najlepsza klasa | CUTOUT (1.00 F1), STANDING (0.91 F1) |
| Najsłabsza klasa | SITTING (0.25 F1 — tylko 6 próbek testowych) |

> ⚠️ Niska jakość dla SITTING i WALKING_DOWNSTAIRS wynika z **bardzo małej liczby próbek** (29 i 34). Zbiór jest silnie niezbalansowany — STANDING stanowi 67% danych.

---

## 🔧 Infrastruktura MLOps

- **Tracking**: MLflow z backendem PostgreSQL
- **Artifact Store**: Azure Blob Storage (`wasbs://mlflow-artifacts@harmlstorage.blob.core.windows.net`)
- **Model Registry**: MLflow — model zarejestrowany jako `HAR_xgboost` (wersja 5) z aliasem `production`
- **URI modelu**: `models:/HAR_xgboost@production`

---

## 🎯 Wnioski i rekomendacje

1. **XGBoost vs Random Forest**: Oba modele osiągają 98% accuracy na dużym zbiorze — XGBoost ma minimalną przewagę w stabilności (CV: 98.02% ± 0.28% vs 97.28% ± 0.39%) i lepiej klasyfikuje SITTING/STANDING.

2. **Redukcja cech**: Ograniczenie do 180 wspólnych cech (z 561) kosztuje ~3 pp. accuracy (94% → 91%), ale jest konieczne do kompatybilności z pipeline'em własnych danych sensorycznych.

3. **Główny problem — ilość własnych danych**: Model produkcyjny (85% accuracy) jest znacząco słabszy niż ten na combined dataset (98%), co wynika z:
   - Tylko **631 próbek** własnych danych vs 7 969 w combined
   - **Silny brak balansu** klas (STANDING: 423, SITTING: 29, CUTOUT: 13)
   - Brak klasy LAYING i WALKING_UPSTAIRS w danych własnych

4. **Rekomendacja**: Zebrać więcej danych sensorycznych, szczególnie dla niedoreprezentowanych klas (SITTING, WALKING_DOWNSTAIRS), oraz dodać nagrania LAYING i WALKING_UPSTAIRS, aby poprawić model produkcyjny.

5. **Ryzyko łączenia zbiorów (Data Mismatch)**: Analiza wykazała, że łączenie danych Kaggle z własnymi wprowadza szum informacyjny. Dane Kaggle są statystycznie znormalizowane (np. ujemna energia), podczas gdy własne dane zachowują fizyczną charakterystykę sensora. Prowadzi to do "rozjazdu" modelu i spadku jakości predykcji w warunkach rzeczywistych.

6. **Specjalizacja modelu (Device Specific)**: Po odseparowaniu danych Kaggle i dotrenowaniu modelu wyłącznie na powiększonym zbiorze własnym, **accuracy wzrosło z 85% do 92%**. Model skutecznie wyspecjalizował się w charakterystyce konkretnego urządzenia (sensora), co znacząco poprawiło jakość predykcji w docelowym środowisku, eliminując błędy wynikające z różnic w dystrybucji danych.
