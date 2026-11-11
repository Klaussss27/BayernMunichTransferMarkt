import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["age", "goals", "assists", "position", "minutes_played"]
TARGET = "market_value_eur"
POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]


def demo_data(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Generate fictional independent players; never represents actual valuations."""
    rng = np.random.default_rng(seed)
    position = rng.choice(POSITIONS, n, p=[0.10, 0.33, 0.34, 0.23])
    age = rng.integers(17, 38, n)
    minutes = rng.integers(0, 3061, n)
    attacking = np.select(
        [position == "Forward", position == "Midfielder", position == "Defender"],
        [0.65, 0.25, 0.07],
        default=0.002,
    )
    goals = rng.poisson(minutes / 90 * attacking)
    assists = rng.poisson(
        minutes / 90 * np.where(position == "Goalkeeper", 0.002, 0.16)
    )
    peak = np.exp(-(((age - 24) / 9) ** 2))
    value = (
        1_000_000
        + peak * 13_000_000
        + minutes * 5500
        + goals * 1_500_000
        + assists * 1_100_000
    )
    value *= rng.lognormal(0, 0.32, n)
    return pd.DataFrame(
        {
            "player_name": [f"Demo player {i + 1:03}" for i in range(n)],
            "age": age,
            "goals": goals,
            "assists": assists,
            "position": position,
            "minutes_played": minutes,
            TARGET: (value / 100_000).round() * 100_000,
        }
    )


def validate_data(frame: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    required = FEATURES + ([TARGET] if require_target else [])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    data = frame.copy()
    for col in [c for c in required if c != "position"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if not np.isfinite(data[col]).all():
            raise ValueError(
                f"{col} must contain finite numbers, with no missing values."
            )
        if (data[col] < 0).any():
            raise ValueError(f"{col} cannot be negative.")
    if not data["age"].between(15, 45).all():
        raise ValueError("Age must be between 15 and 45.")
    if not data["position"].isin(POSITIONS).all():
        raise ValueError(f"Position must be one of: {', '.join(POSITIONS)}.")
    if "season" in data:
        data["season"] = pd.to_numeric(data["season"], errors="coerce")
        if (
            not np.isfinite(data.season).all()
            or not data.season.between(2000, 2100).all()
            or (data.season % 1 != 0).any()
        ):
            raise ValueError(
                "Season must be an integer starting year between 2000 and 2100."
            )
    if (
        "competition_id" in data
        and not data.competition_id.isin(["L1", "GB1", "ES1", "IT1", "FR1"]).all()
    ):
        raise ValueError("Competition must be L1, GB1, ES1, IT1 or FR1.")
    if require_target:
        if len(data) < 60:
            raise ValueError("Provide at least 60 independent player observations.")
        if data[TARGET].nunique() < 2:
            raise ValueError("Market values must vary to train a model.")
        longitudinal = {"player_id", "season", "competition_id"}.issubset(data.columns)
        if longitudinal:
            if data[["player_id", "season", "competition_id"]].isna().any().any():
                raise ValueError(
                    "Player, season and competition identifiers cannot be missing."
                )
            if data.duplicated(["player_id", "season", "competition_id"]).any():
                raise ValueError("Duplicate player/season/competition observation.")
        elif "player_name" in data and data["player_name"].duplicated().any():
            raise ValueError(
                "Use one row per player; repeated players could leak across data splits."
            )
        if not longitudinal and data.duplicated(subset=FEATURES + [TARGET]).any():
            raise ValueError("Remove duplicate observations before training.")
    return data


def load_data(path: str | Path) -> pd.DataFrame:
    return validate_data(pd.read_csv(path))


def load_database(path: str | Path) -> pd.DataFrame:
    uri = f"file:{Path(path).resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return validate_data(
            pd.read_sql_query(
                "SELECT * FROM training_players ORDER BY player_id", connection
            )
        )
