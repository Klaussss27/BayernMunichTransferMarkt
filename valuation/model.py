import hashlib
import platform
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from valuation.candidates import get_estimators
from valuation.data import FEATURES, TARGET, validate_data


class FittedBlend(RegressorMixin, BaseEstimator):
    """Average already fitted pipelines; constituent models never see validation targets."""

    def __init__(self, models):
        self.models = models

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.mean(
            [np.maximum(0, model.predict(X)) for model in self.models], axis=0
        )


def engineer_features(frame):
    result = frame.copy()
    matches = np.maximum(result["minutes_played"] / 90, 1)
    result["goals_per_90"] = result["goals"] / matches
    result["assists_per_90"] = result["assists"] / matches
    result["goal_contributions"] = result["goals"] + result["assists"]
    return result


def split_four_groups(items, seed):
    fit, remainder = train_test_split(items, test_size=0.45, random_state=seed)
    validation, remainder = train_test_split(
        remainder, test_size=2 / 3, random_state=seed
    )
    calibration, test = train_test_split(remainder, test_size=0.5, random_state=seed)
    return dict(fit=fit, validation=validation, calibration=calibration, test=test)


def split_observations(data, seed=42):
    identifiers = {"player_id", "season", "competition_id"}
    longitudinal = identifiers.issubset(data.columns) and data.season.nunique() > 1
    repeated_players = "player_id" in data and data.player_id.duplicated().any()

    if longitudinal or repeated_players:
        player_ids = np.sort(data.player_id.unique())
        player_groups = split_four_groups(player_ids, seed)
        parts = {}
        for name, group in player_groups.items():
            parts[name] = data[data.player_id.isin(group)]
    else:
        parts = split_four_groups(data, seed)

    if longitudinal:
        latest_season = int(data.season.max())
        # Keep later seasons out of fitting, even for players in the fit group.
        parts["fit"] = parts["fit"][parts["fit"].season <= latest_season - 2]
        for name in ["validation", "calibration"]:
            parts[name] = parts[name][parts[name].season == latest_season - 1]
        parts["test"] = parts["test"][parts["test"].season == latest_season]
    if any(len(part) < 8 for part in parts.values()):
        raise ValueError(
            "Insufficient observations for independent fitting, validation, calibration and testing."
        )
    return parts, longitudinal


def build_pipeline(estimator, input_columns, longitudinal):
    categorical = ["position"]
    if longitudinal:
        categorical.append("competition_id")

    numeric = [column for column in input_columns if column not in categorical]
    steps = []
    if longitudinal:
        numeric.extend(["goals_per_90", "assists_per_90", "goal_contributions"])
        steps.append(("derive", FunctionTransformer(engineer_features)))

    preprocessing = ColumnTransformer(
        [
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("numeric", "passthrough", numeric),
        ]
    )
    steps.extend([("prepare", preprocessing), ("regressor", estimator)])
    return Pipeline(steps)


def evaluate_model(name, model, validation, test, input_columns):
    validation_predictions = np.maximum(0, model.predict(validation[input_columns]))
    test_predictions = np.maximum(0, model.predict(test[input_columns]))
    return {
        "model": name,
        "validation_mae": mean_absolute_error(
            validation[TARGET], validation_predictions
        ),
        "test_mae": mean_absolute_error(test[TARGET], test_predictions),
        "test_rmse": np.sqrt(mean_squared_error(test[TARGET], test_predictions)),
        "test_r2": r2_score(test[TARGET], test_predictions),
    }


def calibrate_interval(model, features, targets):
    predictions = np.maximum(0, model.predict(features))
    errors = np.abs(targets - predictions)
    # The extra observation in the rank gives the finite-sample correction.
    level = min(1.0, np.ceil((len(errors) + 1) * 0.9) / len(errors))
    return float(np.quantile(errors, level, method="higher"))


def train(frame: pd.DataFrame, seed: int = 42) -> dict:
    data = validate_data(frame)
    parts, longitudinal = split_observations(data, seed)
    fit, validation, calibration, test = [
        parts[key] for key in ["fit", "validation", "calibration", "test"]
    ]
    input_columns = FEATURES + (["season", "competition_id"] if longitudinal else [])
    estimators = get_estimators(longitudinal, seed)
    models, scores = {}, []
    for name, estimator in estimators.items():
        model = build_pipeline(estimator, input_columns, longitudinal)
        model.fit(fit[input_columns], fit[TARGET])
        scores.append(evaluate_model(name, model, validation, test, input_columns))
        models[name] = model
    if longitudinal:
        leaders = sorted(scores, key=lambda row: row["validation_mae"])[:3]
        blend = FittedBlend([models[row["model"]] for row in leaders])
        name = "Top 3 validation blend"
        models[name] = blend
        scores.append(evaluate_model(name, blend, validation, test, input_columns))
    scores = pd.DataFrame(scores).set_index("model")
    selected = scores["validation_mae"].idxmin()
    model = models[selected]
    intervals = {}
    for name, candidate in models.items():
        intervals[name] = calibrate_interval(
            candidate, calibration[input_columns], calibration[TARGET]
        )
    radius = intervals[selected]
    predictions = np.maximum(0, model.predict(test[input_columns]))
    importance = permutation_importance(
        model,
        test[input_columns],
        test[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=seed,
    )
    bundesliga = (
        test.competition_id.eq("L1") if longitudinal else np.ones(len(test), dtype=bool)
    )
    league_metrics = (
        {
            "rows": int(bundesliga.sum()),
            "mae": float(
                mean_absolute_error(
                    test.loc[bundesliga, TARGET], predictions[bundesliga]
                )
            ),
            "r2": float(
                r2_score(test.loc[bundesliga, TARGET], predictions[bundesliga])
            ),
        }
        if bundesliga.sum() > 1
        else None
    )
    if longitudinal:
        latest_season = int(data.season.max())
        evaluation = (
            f"Disjoint player groups; fit through season {latest_season - 2}, "
            f"validation/calibration season {latest_season - 1}, "
            f"test season {latest_season} (starting years)"
        )
    elif "player_id" in data:
        evaluation = "Independent player groups"
    else:
        evaluation = "Independent random observations"
    return {
        "model": model,
        "models": models,
        "model_intervals": intervals,
        "selected": selected,
        "metrics": scores,
        "interval_radius": radius,
        "importance": pd.DataFrame(
            {"feature": input_columns, "mae_increase": importance.importances_mean}
        ).sort_values("mae_increase"),
        "test_predictions": pd.DataFrame(
            {"actual": test[TARGET].to_numpy(), "predicted": predictions}
        ),
        "input_columns": input_columns,
        "context_defaults": {"season": int(data.season.max()), "competition_id": "L1"}
        if longitudinal
        else {},
        "bundesliga_test": league_metrics,
        "split_assignments": pd.concat(
            [
                part.reindex(
                    columns=["player_id", "player_name", "season", "competition_id"]
                ).assign(split=name)
                for name, part in parts.items()
            ],
            ignore_index=True,
        ),
        "coverage": float(np.mean(np.abs(test[TARGET] - predictions) <= radius)),
        "ranges": {
            f: [float(data[f].min()), float(data[f].max())]
            for f in FEATURES
            if f != "position"
        },
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "selected_model": selected,
            "candidate_count": len(models),
            "test_caveat": "The historical 2025/26 test set has been reported in earlier experiments; this is a reused benchmark, not a newly untouched test set. Model selection uses validation MAE only.",
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in ["scikit-learn", "xgboost", "pandas", "numpy"]
            },
            "dataset_sha256": hashlib.sha256(
                data.to_csv(index=False).encode("utf-8")
            ).hexdigest(),
            "seed": seed,
            "rows": len(data),
            "unique_players": int(data.player_id.nunique())
            if "player_id" in data
            else len(data),
            "evaluation": evaluation,
            "unused_rows": len(data) - sum(len(part) for part in parts.values()),
            "split_seasons": {
                name: sorted(map(int, part.season.unique()))
                for name, part in parts.items()
            }
            if longitudinal
            else {},
            "bundesliga_test": league_metrics,
            "test_interval_coverage": float(
                np.mean(np.abs(test[TARGET] - predictions) <= radius)
            ),
            "split_sizes": {
                "fit": len(fit),
                "validation": len(validation),
                "calibration": len(calibration),
                "test": len(test),
            },
        },
    }


def predict(
    bundle: dict, frame: pd.DataFrame, model_name: str | None = None
) -> pd.DataFrame:
    data = validate_data(frame, require_target=False)
    for column, value in bundle.get("context_defaults", {}).items():
        if column not in data:
            data[column] = value
    if model_name is not None and model_name not in bundle.get("models", {}):
        raise ValueError(f"Unknown model: {model_name}")
    model = bundle["models"][model_name] if model_name else bundle["model"]
    radius = bundle.get("model_intervals", {}).get(
        model_name, bundle["interval_radius"]
    )
    point = np.maximum(0, model.predict(data[bundle.get("input_columns", FEATURES)]))
    result = data.copy()
    result["estimated_value_eur"] = point
    result["lower_90_eur"] = np.maximum(0, point - radius)
    result["upper_90_eur"] = point + radius
    return result
