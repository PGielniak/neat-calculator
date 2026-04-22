# ML Microservices — Opis architektury

## Czym jest ten projekt?

System rozpoznawania aktywności ludzkiej (Human Activity Recognition, HAR) zbudowany jako zestaw mikroserwisów w Pythonie. Na podstawie danych z akcelerometru i żyroskopu (np. ze smartfona) system rozpoznaje, co robi użytkownik: chodzi, siedzi, stoi, leży, wchodzi lub schodzi po schodach.

---

## Stos technologiczny

| Kategoria | Technologia |
|---|---|
| Framework API | FastAPI |
| ML | scikit-learn (RandomForest), XGBoost |
| MLOps | MLflow (śledzenie eksperymentów, rejestr modeli) |
| Baza danych | PostgreSQL |
| Cloud Storage | Azure Blob Storage |
| Konteneryzacja | Docker + docker-compose |
| Język | Python 3.12 |
| Zarządzanie pakietami | uv (workspace) |

---

## Struktura projektu

```
ml_microservices/
├── pyproject.toml                  # Konfiguracja workspace (6 pakietów)
├── docker-compose.yaml             # Orkiestracja kontenerów
├── docker/                         # Dockerfile'y per serwis
└── src/ml_microservices/
    ├── prediction_api/             # Serwis predykcji        (port 8012)
    ├── data_pipeline/              # Serwis przetwarzania    (port 8010)
    ├── model_training/             # Serwis trenowania       (port 8011)
    ├── api_keys/                   # Serwis kluczy API       (port 8013)
    ├── shared/                     # Wspólne narzędzia (nie serwis)
    └── database/                   # Abstrakcja DB (nie serwis)
```

---

## Mikroserwisy

### 1. `prediction_api` — Serwis predykcji (port 8012)

Główny serwis, z którego korzystają aplikacje klienckie. Przyjmuje surowe próbki z czujników i zwraca przewidywaną aktywność.

**Endpointy:**

| Metoda | Ścieżka | Opis |
|---|---|---|
| `GET` | `/health` | Status serwisu i załadowanego modelu |
| `POST` | `/predict` | Predykcja aktywności (wymaga `X-Api-Key`) |

**Przykładowe żądanie `POST /predict`:**
```json
{
  "samples": [
    {
      "accelerometerX": 0.123,
      "accelerometerY": -0.456,
      "accelerometerZ": 9.812,
      "gyroscopeX": 0.01,
      "gyroscopeY": -0.02,
      "gyroscopeZ": 0.005,
      "timestamp": 1710000000000,
      "timestampNanos": 1710000000000000000,
      "label": "UNLABELED"
    }
  ]
}
```

**Przykładowa odpowiedź:**
```json
{
  "activity": "WALKING",
  "confidence": 0.87,
  "prediction_index": 3,
  "all_probabilities": {
    "LAYING": 0.01,
    "SITTING": 0.03,
    "STANDING": 0.05,
    "WALKING": 0.87,
    "WALKING_DOWNSTAIRS": 0.02,
    "WALKING_UPSTAIRS": 0.02
  }
}
```

**Rozpoznawane aktywności:**
- `LAYING` — leżenie
- `SITTING` — siedzenie
- `STANDING` — stanie
- `WALKING` — chodzenie
- `WALKING_DOWNSTAIRS` — schodzenie po schodach
- `WALKING_UPSTAIRS` — wchodzenie po schodach

**Potok przetwarzania danych:**
1. Walidacja — wymagane minimum **128 próbek**
2. Usunięcie duplikatów — próbki z tym samym timestamp uśredniają się
3. Resampling — normalizacja do **50 Hz**
4. Okna przesuwne — okna 128-próbkowe, krok 64
5. Ekstrakcja cech — zestaw UCI HAR (dziedzina czasu + częstotliwości)
6. Skalowanie — MinMaxScaler do zakresu `[-1, 1]`
7. Predykcja modelem RandomForest z MLflow
8. **"Stair tax"** — progi pewności dla schodów:
   - `WALKING_UPSTAIRS` wymaga ≥ 65% pewności
   - `WALKING_DOWNSTAIRS` wymaga ≥ 40% pewności
   - Poniżej progu → spada do `WALKING`

Jeśli dane są niewystarczające, zwracane jest `"activity": "BUFFERING"`.

---

### 2. `data_pipeline` — Serwis przetwarzania danych (port 8010)

Przetwarza surowe pliki JSON z czujników, ekstrahuje cechy i zapisuje je do bazy danych jako dane treningowe.

**Endpointy:**

| Metoda | Ścieżka | Opis |
|---|---|---|
| `POST` | `/data_pipeline_webhook` | Uruchamia pipeline v1 |
| `POST` | `/data_pipeline_webhook2` | Uruchamia pipeline v2 (nowe cechy) |
| `GET` | `/v2/data_pipeline_status/{pipeline_run_id}` | Status uruchomienia pipeline |

**Co robi pipeline:**
1. Pobiera pliki z lokalnego katalogu lub Azure Blob Storage
2. Tworzy manifest plików z checksumami MD5 (deduplication)
3. Rejestruje przetworzone pliki w tabeli `processed_files`
4. Przetwarza dane czujników (ten sam potok co `prediction_api`)
5. Etykietuje dane na podstawie zakresów czasowych z pliku CSV
6. Zapisuje cechy z etykietami do tabeli `training_data_labeled`
7. Aktualizuje status uruchomienia w tabeli `pipeline_runs`

---

### 3. `model_training` — Serwis trenowania modeli (port 8011)

Trenuje model RandomForest na danych z bazy i rejestruje go w MLflow.

**Endpointy:**

| Metoda | Ścieżka | Opis |
|---|---|---|
| `POST` | `/train_model_webhook` | Uruchamia trening asynchronicznie |

**Co robi trening:**
1. Ładuje dane z tabeli `training_data_labeled`
2. Balansuje klasy (4 strategie: `least_represented`, `cap_at_median`, `undersample_highest`, `cap_stairs`)
3. Skaluje do `[-1, 1]` i enkoduje etykiety
4. Dzieli na zbiór treningowy/testowy (80/20, stratyfikowany)
5. Szuka najlepszych hiperparametrów — **RandomizedSearchCV** (30 iteracji, 5-fold CV):
   - `n_estimators`: 100–500
   - `max_depth`: 10–30 lub None
   - `max_features`: sqrt, log2, 0.3
6. Loguje do MLflow: metryki, parametry, artefakty (scaler, label encoder, confusion matrix, feature importances)
7. Rejestruje model w MLflow Model Registry jako `har-randomforest01`

---

### 4. `api_keys` — Serwis kluczy API (port 8013)

Zarządza kluczami API używanymi do autoryzacji żądań do `prediction_api`.

**Endpointy:**

| Metoda | Ścieżka | Opis |
|---|---|---|
| `POST` | `/api-keys/generate` | Generuje nowy klucz API |
| `POST` | `/api-keys/validate` | Weryfikuje klucz API |
| `PATCH` | `/api-keys/disable` | Dezaktywuje klucz (po prefiksie) |
| `PATCH` | `/api-keys/enable` | Aktywuje klucz (po prefiksie) |

**Format klucza:** `{prefiks}_{sekret}` np. `ak7a9c2d_UZy-seCretTokenUrlSafe123`

W bazie przechowywany jest tylko **hash SHA256** części sekretnej (nie sam klucz).

---

### 5. `shared` — Wspólne narzędzia

Biblioteka współdzielona przez `data_pipeline` i `prediction_api`. Zawiera:
- Potok przetwarzania surowych danych (`process_raw_data.py`)
- Ekstrakcję cech HAR zgodną z UCI (`helper_functions.py`) — filtr Butterwortha, FFT, SMA, energia, entropia, korelacje, kąty
- Pomocniki Azure Blob Storage (`storage_account_helpers.py`)

---

### 6. `database` — Abstrakcja bazy danych

Warstwa dostępu do danych używana przez wszystkie serwisy:
- Wzorzec fabryki: SQLite (dev) lub PostgreSQL (prod)
- Wzorzec repozytorium: `save_dataframe()`, `save_record()`, `get_records()`, `update_record()`

**Zmienne środowiskowe:**
```
DATABASE_USER
DATABASE_PASSWORD
DATABASE_URL
DB_PORT
DATABASE_NAME
```

---

## Przepływ danych (end-to-end)

```
Czujniki (telefon/urządzenie)
         │ pliki JSON (akcelerometr + żyroskop)
         ▼
┌─────────────────────┐
│   data_pipeline     │  POST /data_pipeline_webhook
│   (port 8010)       │
└────────┬────────────┘
         │ ekstrakcja cech + etykietowanie
         ▼
┌─────────────────────┐
│    PostgreSQL DB    │  tabela: training_data_labeled
└────────┬────────────┘
         │ dane treningowe
         ▼
┌─────────────────────┐
│   model_training    │  POST /train_model_webhook
│   (port 8011)       │
└────────┬────────────┘
         │ model + artefakty
         ▼
┌─────────────────────┐
│   MLflow Registry   │  model: har-randomforest01
│   (port 8014)       │  alias: production
└────────┬────────────┘
         │ model załadowany przy starcie
         ▼
┌─────────────────────┐
│   prediction_api    │  POST /predict
│   (port 8012)       │◄── X-Api-Key (weryfikacja przez api_keys)
└────────┬────────────┘
         │
         ▼
   {activity, confidence, all_probabilities}

┌─────────────────────┐
│     api_keys        │  zarządza kluczami dostępu
│   (port 8013)       │  do prediction_api
└─────────────────────┘
```

---

## Uruchomienie

```bash
# Skopiuj i uzupełnij zmienne środowiskowe
cp .env.example .env

# Uruchom wszystkie serwisy
docker-compose up -d
```

**Wymagane zmienne środowiskowe (`.env`):**
```
# Baza danych
DATABASE_USER=...
DATABASE_PASSWORD=...
DATABASE_URL=...
DB_PORT=5432
DATABASE_NAME=...

# MLflow
MLFLOW_BACKEND_STORE_URI=postgresql://...
MLFLOW_ARTIFACT_URI=wasbs://...

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=...

# Model
LABELED_TRAINING_DATA_TABLE_NAME=training_data_labeled
```

---

## Tabele w bazie danych

| Tabela | Serwis | Opis |
|---|---|---|
| `training_data_labeled` | data_pipeline → model_training | Cechy z etykietami aktywności |
| `pipeline_runs` | data_pipeline | Historia uruchomień pipeline |
| `processed_files` | data_pipeline | Przetworzone pliki (checksuma MD5) |
| `api_keys` | api_keys | Klucze API (hash SHA256) |
| MLflow tables | model_training / MLflow | Eksperymenty, runs, modele |
