# Bayern player valuation

A football analytics project that estimates player market values from age, position, goals, assists and minutes played. It started with a Bayern-focused Bundesliga dataset and now uses ten seasons from five European leagues.

The Streamlit app lets you adjust a player's statistics, compare players and models, browse squads, and export results. A separate model estimates the following season's value.

## Run locally

Install Python 3.11 or 3.12, then run these commands from the project folder on Windows:

```powershell
./setup.ps1
./run.ps1
```

Setup installs the dependencies and trains both model collections from the included databases. Open the address printed by Streamlit, normally http://localhost:8501. Model files and the local Python environment are excluded from Git because of their size. Evaluation results are saved in `reports/`.

For manual setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
.\.venv\Scripts\python train.py --database data/european_players.sqlite --output artifacts/european_model.joblib
.\.venv\Scripts\python prepare_forecast.py
.\.venv\Scripts\python train.py --database data/forecast_players.sqlite --task next-season --output artifacts/forecast_model.joblib
.\.venv\Scripts\python -m streamlit run app.py
```

On macOS or Linux, replace `.venv\Scripts\python` with `.venv/bin/python`. If a project-local `.python-runtime` is already installed, its `python.exe` can be used instead.

## Data

The source is [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets), which publishes Transfermarkt-derived tables under CC0-1.0. Market values are estimates from that source, not actual transfer fees. The downloaded snapshot has no 2026/27 data.

The main database, `data/european_players.sqlite`, contains **24,604 player-season-league records from 7,645 players**, covering 2016/17 through 2025/26 in the Bundesliga, Premier League, La Liga, Serie A and Ligue 1. Statistics come from 17,937 league matches. Each target is the first available valuation after the league season ends, up to August 31. Records without a suitable target are left out.

`data/european_provenance.json` records dates and coverage. The SQLite tables are `training_players`, `bayern_players` and `dataset_metadata`; the training table is also exported as CSV. The earlier single-season database is retained as `data/bundesliga.sqlite`.

Player photos are cached in `assets/players/`, with source URLs in `assets/player_images.json`. A photo may show a different kit or age from the season being viewed. Missing photos are replaced with initials. To rebuild the data and refresh featured portraits:

```powershell
python expand_data.py
python prepare_images.py
```

## Models and evaluation

The benchmark compares 14 candidates: a median baseline, Random Forest, Extra Trees, XGBoost, histogram boosting, Huber gradient boosting, CatBoost and LightGBM, including loss-function and log-target variants. A final candidate averages the three individual models with the lowest validation MAE.

The multi-season pipeline adds league, season, goals and assists per 90, and total goal contributions to the five basic player statistics. Previous market value is not an input.

Players are assigned to separate groups before applying the season restrictions. This keeps the same player from appearing in training and evaluation.

| Partition | Seasons | Valuation records | Forecast records |
| --- | --- | ---: | ---: |
| Training | Through 2023/24 | 10,825 | 5,982 |
| Validation | 2024/25 | 358 | 265 |
| Interval calibration | 2024/25 | 382 | 229 |
| Test | 2025/26 | 377 | 256 |

Records outside a player's assigned group or season window are excluded. Preprocessing is fitted on training data, and validation MAE determines the selected model. The selected pipeline is not refitted on evaluation rows. Training writes the scores, metadata and split assignments beside the model file.

### Results

The validation-selected blend has these historical test results:

| Metric | Same-season valuation | Next-season forecast |
| --- | ---: | ---: |
| MAE | EUR 6.39m | EUR 9.43m |
| RMSE | EUR 11.94m | EUR 19.43m |
| R-squared | 0.454 | 0.409 |
| Observed 90% interval coverage | 88.3% | 91.4% |

For the 74 Bundesliga observations in the valuation test set, MAE is EUR 6.44m. The newer blend reduced overall valuation MAE by about 1% compared with the previous selected XGBoost, but R-squared fell from 0.465. That is a small improvement in average absolute error, rather than a clear improvement on every measure.

The historical test set has been reported in earlier experiments, so it is a reused benchmark. A new season of data would be needed for a fresh test. Exact results are in [valuation-metrics.csv](reports/valuation-metrics.csv) and [forecast-metrics.csv](reports/forecast-metrics.csv).

### Forecasts and scenarios

`prepare_forecast.py` pairs one season's statistics with the next season's recorded market value. It produces 14,192 paired records. Only players with consecutive observations can be included, so players who leave the covered leagues or disappear from the data are underrepresented.

The app's performance scenarios are a different tool: they change age, goals, assists and minutes while keeping the market-year context fixed. They show how the valuation model responds to those inputs. They are not forecasts of future transfer prices.

## Using your own CSV

Training uploads need these columns. Values must be in euros, not millions.

| Column | Expected value |
| --- | --- |
| `age` | Number from 15 to 45 |
| `goals`, `assists`, `minutes_played` | Nonnegative numbers |
| `position` | `Goalkeeper`, `Defender`, `Midfielder` or `Forward` |
| `market_value_eur` | Nonnegative training target |
| `player_name` | Optional display name |

For repeated players across seasons, also provide `player_id`, `season` as the starting year, and `competition_id`: `L1`, `GB1`, `ES1`, `IT1` or `FR1`. Each player/season/league combination must be unique. Without those identifiers, use one row per player. Training needs at least 60 records and enough observations in each evaluation partition.

Prediction CSVs do not need `market_value_eur`. The multi-season model defaults missing league and season context to Bundesliga and 2025/26.

```powershell
python train.py --data players.csv --output artifacts/custom_model.joblib
python predict.py --model artifacts/custom_model.joblib --data profiles.csv --output artifacts/predictions.csv
python -m pytest -q
```

Only load model files you trust: joblib files can execute code when loaded.

## Limitations and next steps

The model leaves out contracts, injuries, wages, reputation and transfer negotiations. Those omissions matter, especially for expensive players. Historical values are in nominal euros, and player positions come from downloaded profiles rather than a position history.

The intervals are calibrated from held-out residuals with a 90% coverage target. They do not guarantee coverage for an individual player, and changes between seasons can reduce coverage. Forecasts beyond the observed 2025/26 target season have not been validated. The optional synthetic demo is useful for checking the app, but its scores say nothing about real player valuations.

The most useful next steps are to add dated contract and injury information, evaluate errors separately by position and value range, and test against a new season before further tuning. Squad comparisons may include training players and should not be treated as independent evidence of accuracy.

This is an independent project, with no affiliation to FC Bayern Munich, the Bundesliga or Transfermarkt.
