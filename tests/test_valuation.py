from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from prepare_forecast import forecast_pairs
from valuation.data import FEATURES, TARGET, demo_data, load_database, validate_data
from valuation.interactive import scenario_profiles
from valuation.model import predict, split_observations, train


@pytest.fixture(scope="module")
def bundle():
    return train(demo_data(240))


def test_reproducible_data():
    pd.testing.assert_frame_equal(demo_data(), demo_data())


@pytest.mark.parametrize(
    "field,value",
    [
        ("age", 70),
        ("goals", -1),
        ("assists", np.nan),
        ("minutes_played", np.inf),
        ("position", "Striker"),
    ],
)
def test_invalid_inputs(field, value):
    frame = demo_data(100)
    if field != "position":
        frame[field] = frame[field].astype(float)
    frame.loc[0, field] = value
    with pytest.raises(ValueError):
        validate_data(frame)


def test_missing_columns_and_duplicate_players():
    with pytest.raises(ValueError, match="Missing columns"):
        validate_data(demo_data().drop(columns="age"))
    frame = demo_data(100)
    frame.loc[1, "player_name"] = frame.loc[0, "player_name"]
    with pytest.raises(ValueError, match="one row per player"):
        validate_data(frame)


def test_models_and_disjoint_split_sizes(bundle):
    assert set(bundle["metrics"].index) == {
        "Median baseline",
        "Random Forest",
        "XGBoost",
    }
    assert sum(bundle["metadata"]["split_sizes"].values()) == 240
    assert bundle["selected"] == bundle["metrics"].validation_mae.idxmin()
    assert np.isfinite(bundle["metrics"].to_numpy()).all()


def test_prediction_bounds_and_no_target_requirement(bundle):
    profiles = demo_data(8).drop(columns=TARGET)
    result = predict(bundle, profiles)
    assert len(result) == 8
    assert (result.lower_90_eur >= 0).all()
    assert (result.lower_90_eur <= result.estimated_value_eur).all()
    assert (result.upper_90_eur >= result.estimated_value_eur).all()
    assert set(FEATURES).issubset(result.columns)


def test_saved_pipeline_round_trip(bundle, tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    restored = joblib.load(path)
    profile = demo_data(3).drop(columns=TARGET)
    pd.testing.assert_frame_equal(predict(bundle, profile), predict(restored, profile))


def test_database_loading(tmp_path):
    import sqlite3

    frame = demo_data(100)
    path = tmp_path / "players.sqlite"
    # Real databases include a player ID used for deterministic row ordering.
    frame["player_id"] = range(len(frame))
    with sqlite3.connect(path) as connection:
        frame.to_sql("training_players", connection, index=False)
    pd.testing.assert_frame_equal(load_database(path), frame)


def longitudinal_data():
    base = demo_data(120)
    base["player_id"] = range(len(base))
    base["competition_id"] = "L1"
    return pd.concat(
        [base.assign(season=year) for year in range(2022, 2026)], ignore_index=True
    )


def test_longitudinal_splits_do_not_share_players_or_future_seasons():
    frame = validate_data(longitudinal_data())
    parts, grouped = split_observations(frame)
    assert grouped
    for name, part in parts.items():
        for other, other_part in parts.items():
            if other != name:
                assert set(part.player_id).isdisjoint(other_part.player_id)
    assert parts["fit"].season.max() < parts["validation"].season.min()
    assert parts["validation"].season.max() < parts["test"].season.min()
    assert parts["calibration"].season.max() < parts["test"].season.min()


def test_duplicate_season_observations_rejected():
    frame = longitudinal_data()
    with pytest.raises(ValueError, match="Duplicate player/season"):
        validate_data(pd.concat([frame, frame.iloc[:1]], ignore_index=True))


def test_single_season_cross_league_players_stay_together():
    frame = longitudinal_data().query("season == 2025")
    frame = pd.concat([frame, frame.assign(competition_id="GB1")], ignore_index=True)
    parts, _ = split_observations(validate_data(frame))
    groups = [set(part.player_id) for part in parts.values()]
    assert sum(map(len, groups)) == len(set.union(*groups))


def test_expanded_pipeline_with_default_and_explicit_context(tmp_path):
    bundle = train(longitudinal_data())
    assert {
        "CatBoost MAE",
        "LightGBM MAE",
        "Extra Trees",
        "Top 3 validation blend",
    }.issubset(bundle["metrics"].index)
    profiles = demo_data(3).drop(columns=TARGET)
    inferred = predict(bundle, profiles)
    explicit = predict(bundle, profiles.assign(season=2025, competition_id="L1"))
    pd.testing.assert_frame_equal(inferred, explicit)
    path = tmp_path / "expanded.joblib"
    joblib.dump(bundle, path)
    pd.testing.assert_frame_equal(predict(joblib.load(path), profiles), inferred)
    alternative = predict(bundle, profiles, "CatBoost MAE")
    assert np.isfinite(alternative.estimated_value_eur).all()
    assert (alternative.lower_90_eur <= alternative.estimated_value_eur).all()
    with pytest.raises(ValueError, match="Unknown model"):
        predict(bundle, profiles, "not-a-model")


def test_scenarios_keep_market_context_fixed_and_respect_age_limit():
    profile = demo_data(1).assign(age=44, season=2025, competition_id="L1")
    scenarios = scenario_profiles(profile, years=5, goals_change=10)
    assert scenarios.age.tolist() == [44, 45]
    assert scenarios.season.eq(2025).all()
    assert scenarios.goals.iloc[1] == round(profile.goals.iloc[0] * 1.1)


def test_forecast_pairs_use_previous_season_features():
    frame = longitudinal_data()
    frame["valuation_date"] = pd.to_datetime((frame.season + 1).astype(str) + "-06-01")
    frame["minutes_played"] = frame["minutes_played"] + (frame.season - 2022)
    paired = forecast_pairs(frame)
    assert (paired.season == paired.feature_season + 1).all()
    assert (paired.target_valuation_date > paired.valuation_date).all()
    assert len(paired) == 360


def test_streamlit_prediction_flow():
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=60
    ).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert any("Estimated market value" in metric.label for metric in app.metric)
    model_control = next(
        control for control in app.selectbox if control.label == "Valuation model"
    )
    model_control.set_value("Median baseline").run()
    assert not app.exception
    app.radio[0].set_value("Synthetic demonstration").run()
    assert not app.exception
