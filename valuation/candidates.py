"""Model configurations used in the valuation and forecast comparisons."""

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor


def get_estimators(longitudinal, seed):
    """Use the larger comparison for multi-season data."""
    estimators = {
        "Median baseline": DummyRegressor(strategy="median"),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=3,
            max_features=0.9,
            random_state=seed,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=2,
        ),
    }
    if longitudinal:
        estimators.update(
            {
                "Extra Trees": ExtraTreesRegressor(
                    n_estimators=350,
                    min_samples_leaf=3,
                    max_features=0.9,
                    random_state=seed,
                    n_jobs=2,
                ),
                "Histogram Boosting MAE": HistGradientBoostingRegressor(
                    loss="absolute_error",
                    max_iter=350,
                    max_leaf_nodes=15,
                    l2_regularization=10,
                    early_stopping=False,
                    learning_rate=0.05,
                    random_state=seed,
                ),
                "Histogram Boosting RMSE": HistGradientBoostingRegressor(
                    max_iter=350,
                    max_leaf_nodes=15,
                    l2_regularization=10,
                    early_stopping=False,
                    learning_rate=0.05,
                    random_state=seed,
                ),
                "Gradient Boosting Huber": GradientBoostingRegressor(
                    loss="huber",
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.04,
                    min_samples_leaf=8,
                    random_state=seed,
                ),
                "CatBoost MAE": CatBoostRegressor(
                    iterations=700,
                    depth=5,
                    learning_rate=0.04,
                    loss_function="MAE",
                    l2_leaf_reg=8,
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=2,
                ),
                "CatBoost RMSE": CatBoostRegressor(
                    iterations=700,
                    depth=5,
                    learning_rate=0.04,
                    loss_function="RMSE",
                    l2_leaf_reg=8,
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=2,
                ),
                "LightGBM MAE": LGBMRegressor(
                    n_estimators=600,
                    num_leaves=15,
                    learning_rate=0.035,
                    objective="regression_l1",
                    reg_lambda=8,
                    min_child_samples=30,
                    random_state=seed,
                    n_jobs=2,
                    verbosity=-1,
                ),
                "XGBoost depth 5": XGBRegressor(
                    n_estimators=600,
                    max_depth=5,
                    learning_rate=0.035,
                    subsample=0.85,
                    colsample_bytree=0.9,
                    reg_lambda=5,
                    objective="reg:absoluteerror",
                    random_state=seed,
                    n_jobs=2,
                ),
                "Random Forest log target": TransformedTargetRegressor(
                    regressor=RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=3,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
                "XGBoost log target": TransformedTargetRegressor(
                    regressor=XGBRegressor(
                        n_estimators=600,
                        max_depth=5,
                        learning_rate=0.035,
                        subsample=0.85,
                        colsample_bytree=0.9,
                        reg_lambda=5,
                        objective="reg:squarederror",
                        random_state=seed,
                        n_jobs=2,
                    ),
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            }
        )
    return estimators
