import platform
import hashlib
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from xgboost import XGBRegressor

from valuation.data import FEATURES, TARGET, validate_data


class FittedBlend(RegressorMixin, BaseEstimator):
    """Average already fitted pipelines; constituent models never see validation targets."""
    def __init__(self, models):
        self.models = models

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.mean([np.maximum(0, model.predict(X)) for model in self.models], axis=0)


def engineer_features(frame):
    result = frame.copy()
    matches = np.maximum(result["minutes_played"] / 90, 1)
    result["goals_per_90"] = result["goals"] / matches
    result["assists_per_90"] = result["assists"] / matches
    result["goal_contributions"] = result["goals"] + result["assists"]
    return result


def split_observations(data, seed=42):
    longitudinal = {"player_id", "season", "competition_id"}.issubset(data.columns) and data.season.nunique() > 1
    if not longitudinal and "player_id" in data and data.player_id.duplicated().any():
        ids = np.sort(data.player_id.unique())
        fit_ids, remainder = train_test_split(ids, test_size=.45, random_state=seed)
        validation_ids, remainder = train_test_split(remainder, test_size=2 / 3, random_state=seed)
        calibration_ids, test_ids = train_test_split(remainder, test_size=.5, random_state=seed)
        fit, validation, calibration, test = [data[data.player_id.isin(group)] for group in [fit_ids, validation_ids, calibration_ids, test_ids]]
    elif not longitudinal:
        fit, remainder = train_test_split(data, test_size=.45, random_state=seed)
        validation, remainder = train_test_split(remainder, test_size=2 / 3, random_state=seed)
        calibration, test = train_test_split(remainder, test_size=.5, random_state=seed)
    else:
        ids = np.sort(data.player_id.unique())
        fit_ids, remainder = train_test_split(ids, test_size=.45, random_state=seed)
        validation_ids, remainder = train_test_split(remainder, test_size=2 / 3, random_state=seed)
        calibration_ids, test_ids = train_test_split(remainder, test_size=.5, random_state=seed)
        latest = int(data.season.max())
        fit = data[data.player_id.isin(fit_ids) & (data.season <= latest - 2)]
        validation = data[data.player_id.isin(validation_ids) & (data.season == latest - 1)]
        calibration = data[data.player_id.isin(calibration_ids) & (data.season == latest - 1)]
        test = data[data.player_id.isin(test_ids) & (data.season == latest)]
    parts = dict(fit=fit, validation=validation, calibration=calibration, test=test)
    if any(len(part) < 8 for part in parts.values()):
        raise ValueError("Insufficient observations for independent fitting, validation, calibration and testing.")
    return parts, longitudinal


def train(frame: pd.DataFrame, seed: int = 42) -> dict:
    data = validate_data(frame)
    parts, longitudinal = split_observations(data, seed)
    fit, validation, calibration, test = [parts[key] for key in ["fit", "validation", "calibration", "test"]]
    input_columns = FEATURES + (["season", "competition_id"] if longitudinal else [])
    estimators = {
        "Median baseline": DummyRegressor(strategy="median"),
        "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=3, max_features=.9, random_state=seed, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=.04, subsample=.85, colsample_bytree=.9, objective="reg:squarederror", random_state=seed, n_jobs=2),
    }
    if longitudinal:
        estimators.update({
            "Extra Trees": ExtraTreesRegressor(n_estimators=350, min_samples_leaf=3, max_features=.9, random_state=seed, n_jobs=2),
            "Histogram Boosting MAE": HistGradientBoostingRegressor(loss="absolute_error", max_iter=350, max_leaf_nodes=15, l2_regularization=10, early_stopping=False, learning_rate=.05, random_state=seed),
            "Histogram Boosting RMSE": HistGradientBoostingRegressor(max_iter=350, max_leaf_nodes=15, l2_regularization=10, early_stopping=False, learning_rate=.05, random_state=seed),
            "Gradient Boosting Huber": GradientBoostingRegressor(loss="huber", n_estimators=300, max_depth=3, learning_rate=.04, min_samples_leaf=8, random_state=seed),
            "CatBoost MAE": CatBoostRegressor(iterations=700, depth=5, learning_rate=.04, loss_function="MAE", l2_leaf_reg=8, random_seed=seed, verbose=False, allow_writing_files=False, thread_count=2),
            "CatBoost RMSE": CatBoostRegressor(iterations=700, depth=5, learning_rate=.04, loss_function="RMSE", l2_leaf_reg=8, random_seed=seed, verbose=False, allow_writing_files=False, thread_count=2),
            "LightGBM MAE": LGBMRegressor(n_estimators=600, num_leaves=15, learning_rate=.035, objective="regression_l1", reg_lambda=8, min_child_samples=30, random_state=seed, n_jobs=2, verbosity=-1),
            "XGBoost depth 5": XGBRegressor(n_estimators=600, max_depth=5, learning_rate=.035, subsample=.85, colsample_bytree=.9, reg_lambda=5, objective="reg:absoluteerror", random_state=seed, n_jobs=2),
            "Random Forest log target": TransformedTargetRegressor(regressor=RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1), func=np.log1p, inverse_func=np.expm1),
            "XGBoost log target": TransformedTargetRegressor(regressor=XGBRegressor(n_estimators=600, max_depth=5, learning_rate=.035, subsample=.85, colsample_bytree=.9, reg_lambda=5, objective="reg:squarederror", random_state=seed, n_jobs=2), func=np.log1p, inverse_func=np.expm1),
        })
    models, scores = {}, []
    for name, estimator in estimators.items():
        categorical = ["position"] + (["competition_id"] if longitudinal else [])
        numeric = [f for f in input_columns if f not in categorical]
        if longitudinal:
            numeric += ["goals_per_90", "assists_per_90", "goal_contributions"]
        preprocessing = ColumnTransformer([("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical), ("numeric", "passthrough", numeric)])
        steps = ([("derive", FunctionTransformer(engineer_features))] if longitudinal else []) + [("prepare", preprocessing), ("regressor", estimator)]
        model = Pipeline(steps)
        model.fit(fit[input_columns], fit[TARGET])
        val_pred = np.maximum(0, model.predict(validation[input_columns]))
        pred = np.maximum(0, model.predict(test[input_columns]))
        scores.append({"model": name, "validation_mae": mean_absolute_error(validation[TARGET], val_pred), "test_mae": mean_absolute_error(test[TARGET], pred), "test_rmse": np.sqrt(mean_squared_error(test[TARGET], pred)), "test_r2": r2_score(test[TARGET], pred)})
        models[name] = model
    if longitudinal:
        leaders = sorted(scores, key=lambda row: row["validation_mae"])[:3]
        blend = FittedBlend([models[row["model"]] for row in leaders])
        val_pred = blend.predict(validation[input_columns])
        pred = blend.predict(test[input_columns])
        name = "Top 3 validation blend"
        models[name] = blend
        scores.append({"model": name, "validation_mae": mean_absolute_error(validation[TARGET], val_pred), "test_mae": mean_absolute_error(test[TARGET], pred), "test_rmse": np.sqrt(mean_squared_error(test[TARGET], pred)), "test_r2": r2_score(test[TARGET], pred)})
    scores = pd.DataFrame(scores).set_index("model")
    selected = scores["validation_mae"].idxmin()
    model = models[selected]
    residuals = np.abs(calibration[TARGET] - np.maximum(0, model.predict(calibration[input_columns])))
    # Split-conformal 90% marginal coverage, under exchangeability assumptions.
    level = min(1., np.ceil((len(residuals) + 1) * .9) / len(residuals))
    radius = float(np.quantile(residuals, level, method="higher"))
    intervals = {name: float(np.quantile(np.abs(calibration[TARGET] - np.maximum(0, candidate.predict(calibration[input_columns]))), level, method="higher")) for name, candidate in models.items()}
    predictions = np.maximum(0, model.predict(test[input_columns]))
    importance = permutation_importance(model, test[input_columns], test[TARGET], scoring="neg_mean_absolute_error", n_repeats=5, random_state=seed)
    bundesliga = test.competition_id.eq("L1") if longitudinal else np.ones(len(test), dtype=bool)
    league_metrics = {"rows": int(bundesliga.sum()), "mae": float(mean_absolute_error(test.loc[bundesliga, TARGET], predictions[bundesliga])), "r2": float(r2_score(test.loc[bundesliga, TARGET], predictions[bundesliga]))} if bundesliga.sum() > 1 else None
    return {
        "model": model,
        "models": models,
        "model_intervals": intervals,
        "selected": selected,
        "metrics": scores,
        "interval_radius": radius,
        "importance": pd.DataFrame({"feature": input_columns, "mae_increase": importance.importances_mean}).sort_values("mae_increase"),
        "test_predictions": pd.DataFrame({"actual": test[TARGET].to_numpy(), "predicted": predictions}),
        "input_columns": input_columns,
        "context_defaults": {"season": int(data.season.max()), "competition_id": "L1"} if longitudinal else {},
        "bundesliga_test": league_metrics,
        "split_assignments": pd.concat([part.reindex(columns=["player_id", "player_name", "season", "competition_id"]).assign(split=name) for name, part in parts.items()], ignore_index=True),
        "coverage": float(np.mean(np.abs(test[TARGET] - predictions) <= radius)),
        "ranges": {f: [float(data[f].min()), float(data[f].max())] for f in FEATURES if f != "position"},
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "selected_model": selected,
            "candidate_count": len(models),
            "test_caveat": "The historical 2025/26 test set has been reported in earlier experiments; this is a reused benchmark, not a newly untouched test set. Model selection uses validation MAE only.",
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ["scikit-learn", "xgboost", "pandas", "numpy"]},
            "dataset_sha256": hashlib.sha256(data.to_csv(index=False).encode("utf-8")).hexdigest(),
            "seed": seed,
            "rows": len(data),
            "unique_players": int(data.player_id.nunique()) if "player_id" in data else len(data),
            "evaluation": f"Disjoint player groups; fit through season {int(data.season.max()) - 2}, validation/calibration season {int(data.season.max()) - 1}, test season {int(data.season.max())} (starting years)" if longitudinal else "Independent player groups" if "player_id" in data else "Independent random observations",
            "unused_rows": len(data) - sum(len(part) for part in parts.values()),
            "split_seasons": {name: sorted(map(int, part.season.unique())) for name, part in parts.items()} if longitudinal else {},
            "bundesliga_test": league_metrics,
            "test_interval_coverage": float(np.mean(np.abs(test[TARGET] - predictions) <= radius)),
            "split_sizes": {"fit": len(fit), "validation": len(validation), "calibration": len(calibration), "test": len(test)},
        },
    }


def predict(bundle: dict, frame: pd.DataFrame, model_name: str | None = None) -> pd.DataFrame:
    data = validate_data(frame, require_target=False)
    for column, value in bundle.get("context_defaults", {}).items():
        if column not in data:
            data[column] = value
    if model_name is not None and model_name not in bundle.get("models", {}):
        raise ValueError(f"Unknown model: {model_name}")
    model = bundle["models"][model_name] if model_name else bundle["model"]
    radius = bundle.get("model_intervals", {}).get(model_name, bundle["interval_radius"])
    point = np.maximum(0, model.predict(data[bundle.get("input_columns", FEATURES)]))
    result = data.copy()
    result["estimated_value_eur"] = point
    result["lower_90_eur"] = np.maximum(0, point - radius)
    result["upper_90_eur"] = point + radius
    return result
