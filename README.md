# Bundesliga Player Valuation — ML Predictor

## Interactive edition

Live valuation controls recalculate as you adjust age, goals, assists and minutes. Switch between trained models in the sidebar, scout any covered club, compare up to four players, filter prospects by age/position/budget, export shortlists, and inspect historical valuations. The Interactive lab includes counterfactual performance scenarios and a separate next-season prediction model. Historical player photos, squad estimates, filtered data exports and model leaderboards are included.

**Fresh clone on Windows:** install Python 3.11 or 3.12, run `./setup.ps1`, then `./run.ps1`. Setup installs the pinned dependencies and trains both model collections from the included dated SQLite data. Large fitted artifacts, local Python runtimes and raw source archives are excluded from Git; training metrics and metadata are committed in `reports/`. Setup requires internet access for Python dependencies. Portraits already cached in `assets/players/` work offline.

### Current benchmark

Fourteen candidates include Random Forest, XGBoost, Extra Trees, histogram gradient boosting (MAE/RMSE), Huber gradient boosting, CatBoost (MAE/RMSE), LightGBM, log-target variants, a median baseline, and a mean blend of the three validation-leading individual models. The blend is selected by validation MAE; no model is selected by test results. Each selectable candidate has its own calibration interval.

The selected valuation blend scores **€6.39m test MAE**, **€11.94m RMSE**, **0.454 R²**, and **88.3% observed interval coverage**. Compared with the preceding selected XGBoost, MAE falls by approximately 1%, while R² falls from 0.465. This is a small MAE improvement, not an across-the-board accuracy gain. Bundesliga subset MAE is €6.44m (74 observations). This historical test set has already been used and reported in earlier experiments, so it is a reused benchmark rather than a newly untouched final test. Data still ends in 2025/26; no new 2026/27 ground truth has been acquired.

### Next-season forecasting

`prepare_forecast.py` pairs each primary player-season with the following season's recorded value, keeping preceding-season age, goals, assists, position and minutes as features. Previous market value is not an input. The dataset has **14,192 paired rows**; 5,982 are used for fitting, 265 for validation, 229 for calibration, and 256 for testing. Players remain disjoint across partitions, and fitting precedes validation/calibration and testing in time. Only players with consecutive observations are included, so survivorship bias applies. Source positions are taken from current profiles, not historical position records.

Validation selects the top-three blend for forecasting too. Historical forecast test MAE is **€9.43m**, RMSE **€19.43m**, R² **0.409**, and interval coverage **91.4%**. The target differs from same-season valuation, so these errors are not directly comparable. Forecasts beyond the 2025/26 target season are unvalidated; historical interval coverage may not hold. The what-if studio holds market-year context fixed and is a separate counterfactual tool, not a future-price forecast.

```powershell
.\.venv\Scripts\python train.py --database data/european_players.sqlite --output artifacts/european_model.joblib
.\.venv\Scripts\python prepare_forecast.py
.\.venv\Scripts\python train.py --database data/forecast_players.sqlite --task next-season --output artifacts/forecast_model.joblib
```

The sections below describe the earlier dataset expansion and baseline experiments; `reports/valuation-metrics.csv` and `reports/forecast-metrics.csv` hold the latest model comparison.

A working FC Bayern-focused sports analytics prototype: Random Forest and XGBoost estimate player market values from age, goals, assists, playing position, and minutes played. Includes an interactive Streamlit dashboard, CSV training import, individual and squad predictions, model comparison, uncertainty intervals, CSV exports, and a reproducible training CLI.

The SCOUT / BAYERN interface includes a player selector, photographic profile panels, prefilled valuation controls, and a club squad gallery. Run `python prepare_images.py` to cache featured-player portraits from image URLs in the downloaded source dataset. Cached portraits work locally without remote image requests; missing images display initials. `assets/player_images.json` records the original image URLs. Images may show a different age or kit than the historical training season.

## Prepared database and trained model

The default database is **`data/european_players.sqlite`**, and the expanded model is **`artifacts/european_model.joblib`**. The dashboard loads these when present, verifies the dataset fingerprint, and defaults prediction context to Bundesliga, season 2025/26.

The expanded database contains **24,604 player-season-league observations from 7,645 unique players**, across Bundesliga, Premier League, La Liga, Serie A, and Ligue 1, spanning 2016/17–2025/26. Source statistics cover 17,937 league matches. Every target is the earliest available valuation after that league-season's final match and no later than August 31. Missing targets are excluded. The pipeline adds goals/assists per 90 and total goal contributions, plus season and league context, without using previous market values as features.

Players are assigned to four disjoint groups before applying time restrictions: fitting uses 10,825 observations through 2023/24; validation uses 358 observations from 2024/25; calibration uses 382 observations from 2024/25; testing uses 377 observations from 2025/26. No player appears in more than one partition. The remaining 12,662 database rows are intentionally excluded because they violate the assigned player group or season restrictions. `.splits.csv` records the assignments. The artifact is fitted on 10,825 observations, not all 24,604 rows.

Six fixed candidates compare a median baseline, Random Forest, XGBoost, a deeper XGBoost trained with absolute-error loss, and log-target Random Forest/XGBoost variants. Validation MAE selected **XGBoost depth 5**. Its test MAE is **€6.46m**, RMSE **€11.82m**, and R² **0.465**. For the 74 held-out Bundesliga observations, MAE is **€6.40m** and R² **0.260**. Observed test interval coverage is **89.4%**. The previous and expanded model scores use different test sets and cannot establish a direct improvement claim.

```powershell
.\.python-runtime\python.exe expand_data.py
.\.python-runtime\python.exe train.py --database data/european_players.sqlite --output artifacts/european_model.joblib
.\.python-runtime\python.exe predict.py --model artifacts/european_model.joblib --data data/profiles.csv
```

`data/european_provenance.json` records league-season coverage and dates; `data/european_players.csv` is the exported training table. The expanded SQLite file contains `training_players`, latest-season `bayern_players`, and `dataset_metadata`; original compressed source tables remain cached under `data/raw/`. Historical values are in nominal euros, and profile positions are not historically versioned. The preceding-season calibration does not guarantee coverage after temporal distribution changes.

### Earlier single-season model

The original database **`data/bundesliga.sqlite`** and model **`artifacts/bundesliga_model.joblib`** are retained. The app uses them only if the expanded database/model are unavailable.

The prepared 2025/26 dataset contains 470 Bundesliga players, including 26 whose main club by league minutes was Bayern Munich. It aggregates all 306 league matches ending May 16, 2026, and uses each player's earliest available valuation after the season ended (observed targets: May 20–June 11, 2026). Each player appears once. Of 497 players with recorded appearances, 470 have eligible targets and features. Missing target values are excluded rather than estimated.

Source: [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets), published under CC0-1.0. The publisher reports paused updates, so this is a historical snapshot, not current 2026/27 pricing. See `data/provenance.json` for dates, counts, exclusions, and source file checksums. Position is taken from downloaded player profiles and is not guaranteed to be the historical season position.

Database tables: `training_players`, `bayern_players`, `bundesliga_games`, `bundesliga_appearances`, `player_profiles`, `player_valuations`, and `dataset_metadata`. The training CSV is `data/bundesliga_players.csv`. Compressed upstream tables are cached in `data/raw/`.

Rebuild and retrain with the project-local interpreter:

```powershell
.\.python-runtime\python.exe prepare_data.py --season 2025
.\.python-runtime\python.exe train.py --database data/bundesliga.sqlite --output artifacts/bundesliga_model.joblib
.\.python-runtime\python.exe predict.py --model artifacts/bundesliga_model.joblib --data data/profiles.csv
```

Both regressors have been trained and compared with a median baseline. Validation MAE selected XGBoost. Its held-out test MAE is approximately €8.47m, RMSE €13.10m, and R² 0.127. These results show limited predictive accuracy with the five requested features. Selection uses validation scores even though Random Forest performed better on the final test set. Exact scores and split sizes are saved beside the artifact in `.metrics.csv` and `.json` files. Only the selected model pipeline is persisted; all candidates' evaluation scores are saved.

## Run

Requires Python 3.11 or 3.12. From this directory on Windows:

`requirements-lock.txt` records the exact dependency versions verified locally; install it instead of `requirements.txt` to reproduce that environment.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501). On macOS/Linux, use `.venv/bin/python` instead.

If the project-local `.python-runtime` has already been installed, run `./run.ps1` directly, or use `.\.python-runtime\python.exe` in place of the virtual environment interpreter below.

```powershell
.\.venv\Scripts\python train.py
.\.venv\Scripts\python train.py --data data/players.csv --output artifacts/model.joblib
.\.venv\Scripts\python predict.py --data data/profiles.csv --output artifacts/predictions.csv
.\.venv\Scripts\python -m pytest -q
```

The CLI saves the fitted preprocessing/model pipeline, evaluation metrics, and training metadata. The dashboard loads the saved real-data artifact in database mode and trains independently in CSV/demo modes. Only load trusted joblib files, since deserialization executes code.

## Data contract

Upload UTF-8 CSV with one row per independent player observation. Use a consistent season and valuation date. All monetary values are euros, not millions. Optional `player_name` must be unique when present unless longitudinal identifiers are supplied. For longitudinal data, provide `player_id`, `season` (integer starting year), and `competition_id` (`L1`, `GB1`, `ES1`, `IT1`, `FR1`); repeated players are allowed across distinct season/league observations and split by identity. Squad prediction uploads do not require the target column. Expanded-model batch predictions accept optional season and competition columns; omitted context defaults to 2025 and L1.

| Column | Meaning |
| --- | --- |
| `player_name` | Optional display label |
| `age` | Numeric age, 15–45 |
| `goals` | Nonnegative season goal count |
| `assists` | Nonnegative season assist count |
| `position` | `Goalkeeper`, `Defender`, `Midfielder`, or `Forward` |
| `minutes_played` | Nonnegative season minutes |
| `market_value_eur` | Nonnegative numeric target; training only |

At least 60 training rows and varying target values are required, with enough players in every evaluation partition. The app provides a downloadable synthetic example. It rejects missing/nonfinite numbers, invalid positions, duplicate labeled players without longitudinal identifiers, and duplicate player/season/league keys. Unlabeled repeated players cannot be detected automatically; verify player independence before importing.

## Evaluation

For independent single-season observations, seeded random splits reserve approximately 55% for fitting, 15% for model selection, 15% for interval calibration, and 15% for final testing. Multi-season data uses the disjoint-player and chronological design described above. Category encoding is fitted inside each pipeline. The lowest validation MAE selects the model. Reports include test MAE, RMSE, R², observed interval coverage, actual/predicted scatter, and permutation importance. The selected model is not refitted on held-out observations.

Prediction intervals use split-conformal absolute residual calibration targeting 90% marginal coverage under exchangeability, with finite-sample quantile correction. They do not guarantee coverage for an individual player or out-of-distribution inputs. Small datasets produce unstable metrics and wide intervals. Temporal validation intentionally tests distribution changes, so exact exchangeability between calibration and test seasons should not be assumed.

## Scope and limitations

The optional synthetic demonstration is generated locally from the formula in `valuation/data.py`; every demo player and value is fictional. Demo scores measure recovery of that formula and do **not** establish real-world accuracy. The real-data mode uses the dated source snapshot described above. Market-value labels are Transfermarkt estimates rather than observed transfer fees. The expanded evaluation tests later seasons with unseen players; it does not establish accuracy for 2026/27. The complete league dataset includes Bayern players, so predictions for the entire Bayern squad are not an independent club-level test.

Contracts, injuries, wages, reputation, inflation, and negotiation dynamics are not included. Estimates support experimentation and are not transfer-price appraisals. This project is not affiliated with FC Bayern Munich or the Bundesliga.
