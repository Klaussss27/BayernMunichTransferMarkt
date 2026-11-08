"""Pair observed season features with the following season's recorded valuation."""
import json
from pathlib import Path
import sqlite3

import pandas as pd

from valuation.data import load_database, validate_data


def forecast_pairs(data):
    # One primary league per player-season; avoid duplicate transfer observations.
    history = data.sort_values("minutes_played", ascending=False).drop_duplicates(["player_id", "season"])
    following = history[["player_id", "season", "market_value_eur", "valuation_date"]].copy()
    following["season"] -= 1
    following = following.rename(columns={"market_value_eur": "next_value", "valuation_date": "target_valuation_date"})
    paired = history.merge(following, on=["player_id", "season"], validate="one_to_one")
    paired["feature_season"] = paired["season"]
    paired["season"] += 1  # season context refers to the target year
    paired["market_value_eur"] = paired.pop("next_value")
    return validate_data(paired.sort_values(["player_id", "season"]).reset_index(drop=True))


if __name__ == "__main__":
    frame = forecast_pairs(load_database("data/european_players.sqlite"))
    with sqlite3.connect("data/forecast_players.sqlite") as connection:
        frame.to_sql("training_players", connection, if_exists="replace", index=False)
    frame.to_csv("data/forecast_players.csv", index=False)
    metadata = {"task": "Next-season recorded market value from preceding-season performance", "rows": len(frame), "unique_players": int(frame.player_id.nunique()), "target_seasons": sorted(map(int, frame.season.unique())), "limitation": "Only players with consecutive observations are included, creating survivorship bias. Targets are source estimates, not realized transfer fees. Current-profile historical positions are not versioned."}
    Path("data/forecast_provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
