# Bayern player valuation

I built this project to explore how well season statistics can explain a football player's market value. The initial focus was Bayern Munich and the Bundesliga. I later expanded the dataset to five European leagues and added a separate next-season prediction task.

The project combines data preparation, regression models and a Streamlit dashboard. The main inputs are age, playing position, goals, assists and minutes played.

## Data preparation

I used the public [Transfermarkt dataset maintained by dcaribou](https://github.com/dcaribou/transfermarkt-datasets). It contains player profiles, match appearances and historical market-value estimates. The published tables are marked CC0-1.0. The valuation labels are estimates from Transfermarkt, rather than completed transfer fees.

I joined appearances with match records, grouped the statistics by player and season, and matched each record to the first available valuation after the league season ended. I excluded records without a suitable target and stored the prepared data in SQLite, with CSV exports for inspection.

The expanded dataset contains **24,604 player-season-league records from 7,645 players**, covering 2016/17 through 2025/26. It includes the Bundesliga, Premier League, La Liga, Serie A and Ligue 1, with statistics from 17,937 league matches. The source snapshot does not include 2026/27.

I also kept the original Bundesliga-only dataset and recorded the source dates, coverage and preparation details in the provenance files.

## Model development

I started with Random Forest and XGBoost, then added Extra Trees, histogram boosting, Huber gradient boosting, CatBoost and LightGBM. The benchmark compares 14 candidates, including a median baseline, different loss functions, log-transformed targets and a blend of the three best individual models on the validation set.

Besides the five basic inputs, the multi-season pipeline uses league and season context, goals and assists per 90 minutes, and total goal contributions. Previous market value is not used as an input.

I separated players into training, validation, calibration and test groups before applying the season restrictions. This prevents records for the same player from appearing on both sides of the evaluation. Preprocessing is fitted on the training data, and validation MAE determines the selected model.

| Group | Seasons | Valuation records | Forecast records |
| --- | --- | ---: | ---: |
| Training | Through 2023/24 | 10,825 | 5,982 |
| Validation | 2024/25 | 358 | 265 |
| Interval calibration | 2024/25 | 382 | 229 |
| Test | 2025/26 | 377 | 256 |

Not all database rows are used for fitting. Records outside their assigned player group or season window are excluded. I saved model scores, split assignments and training metadata so the evaluation can be checked later.

## Results

Validation MAE selected the three-model blend for both tasks.

| Metric | Same-season valuation | Next-season forecast |
| --- | ---: | ---: |
| Mean absolute error | EUR 6.39m | EUR 9.43m |
| Root mean squared error | EUR 11.94m | EUR 19.43m |
| R-squared | 0.454 | 0.409 |
| Observed 90% interval coverage | 88.3% | 91.4% |

The valuation model's Bundesliga subset has an MAE of EUR 6.44m across 74 test observations. The blend reduced overall MAE by about 1% compared with the previous selected XGBoost, but R-squared fell from 0.465. The improvement is small and depends on which metric is used.

The historical test set has been reported in earlier experiments, so these results are a reused benchmark. They do not replace a fresh test on a new season. The detailed comparisons are saved in [valuation-metrics.csv](reports/valuation-metrics.csv) and [forecast-metrics.csv](reports/forecast-metrics.csv).

## Forecasting

For the next-season model, I paired a player's statistics with their recorded value in the following season. This produced 14,192 paired records. Only players with consecutive observations could be included, which means the forecast dataset underrepresents players who left the covered leagues or disappeared from the source.

I kept forecasting separate from the dashboard's what-if scenarios. Those scenarios change age and performance statistics while holding the market-year context fixed. They show how the valuation model responds to different inputs, rather than predicting actual future transfer prices.

## Dashboard

I built the dashboard in Streamlit to make the results easier to explore. It includes player photos, squad browsing, live valuation controls, model switching and comparisons between up to four players. Scouting filters cover age, position and recorded value, with CSV exports for shortlists and results.

The app also shows historical valuations, model errors, feature importance and calibrated prediction intervals. Recorded source values and model estimates are labeled separately. Photos are cached locally, with initials used when an image is missing.

## Testing and limitations

I added tests for invalid inputs, repeated players, chronological splits, saved-model reloads, scenario calculations, forecast pairing and the dashboard's prediction controls. The project includes a GitHub Actions workflow for the Python checks.

The models leave out contracts, injuries, wages, reputation and transfer negotiations. Historical values are in nominal euros, and positions come from downloaded player profiles rather than a historical position table. These gaps matter, especially for expensive players.

Prediction intervals use held-out residuals with a 90% coverage target. They do not guarantee coverage for a particular player, and changes between seasons can reduce coverage. Forecasts beyond the observed 2025/26 target season have not been validated. Squad comparisons may include players used during fitting and are not an independent accuracy test.

The next improvements would be dated contract and injury data, error analysis by position and value range, and evaluation against a new season before further tuning.

This is an independent project with no affiliation to FC Bayern Munich, the Bundesliga or Transfermarkt.
