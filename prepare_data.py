"""Download public source tables and build a dated Bundesliga SQLite database."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd
import requests

from valuation.data import validate_data

BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
SOURCE = "https://github.com/dcaribou/transfermarkt-datasets"


def download(table):
    path = Path("data/raw") / f"{table}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {table}…", flush=True)
        with requests.get(BASE + path.name, timeout=(20, 180), stream=True) as response:
            response.raise_for_status()
            temporary = path.with_suffix(".part")
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
            temporary.replace(path)
    return path


def prepare(season=2025):
    tables = ["players", "games", "appearances", "player_valuations", "clubs"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = dict(zip(tables, pool.map(download, tables)))
    players = pd.read_csv(paths["players"])
    games = pd.read_csv(paths["games"], low_memory=False)
    valuations = pd.read_csv(paths["player_valuations"])
    clubs = pd.read_csv(paths["clubs"])
    # All season features end before the summer valuation window opens.
    end = pd.Timestamp(f"{season + 1}-08-31")
    games["date"] = pd.to_datetime(games["date"])
    league_games = games[(games.competition_id == "L1") & (games.season == season)]
    if len(league_games) != 306:
        raise ValueError(f"Season {season} is incomplete: only {len(league_games)} Bundesliga matches.")
    cutoff = league_games.date.max() + pd.Timedelta(days=1)
    chunks = []
    for chunk in pd.read_csv(paths["appearances"], chunksize=200_000):
        chunks.append(chunk[chunk.game_id.isin(league_games.game_id)])
    appearances = pd.concat(chunks, ignore_index=True)
    stats = appearances.groupby("player_id", as_index=False)[["goals", "assists", "minutes_played"]].sum()
    main_club = appearances.groupby(["player_id", "player_club_id"], as_index=False).minutes_played.sum()
    main_club = main_club.sort_values("minutes_played", ascending=False).drop_duplicates("player_id")
    stats = stats.merge(main_club[["player_id", "player_club_id"]], on="player_id", validate="one_to_one")
    valuations["date"] = pd.to_datetime(valuations["date"])
    summer = valuations[valuations.date.between(cutoff, end)].sort_values("date").drop_duplicates("player_id")
    frame = stats.merge(players[["player_id", "name", "date_of_birth", "position"]], on="player_id", validate="one_to_one")
    frame = frame.merge(summer[["player_id", "date", "market_value_in_eur"]], on="player_id", validate="one_to_one")
    frame = frame.rename(columns={"name": "player_name", "market_value_in_eur": "market_value_eur", "date": "valuation_date"})
    frame["age"] = ((frame.valuation_date - pd.to_datetime(frame.date_of_birth)).dt.days / 365.2425).round(2)
    frame["position"] = frame.position.replace({"Attack": "Forward", "Midfield": "Midfielder"})
    frame["season"] = season
    frame = frame.merge(clubs[["club_id", "name"]].rename(columns={"club_id": "player_club_id", "name": "club_name"}), on="player_club_id", how="left", validate="many_to_one")
    before = len(frame)
    frame = frame.dropna(subset=["age", "position", "market_value_eur"])
    frame = frame[frame.age.between(15, 45) & frame.position.isin(["Forward", "Midfielder", "Defender", "Goalkeeper"])]
    frame = frame.sort_values("player_id").reset_index(drop=True)
    validate_data(frame)
    metadata = {
        "source": SOURCE, "download_base": BASE, "license": "CC0-1.0 (publisher declaration)",
        "prepared_at": datetime.now(timezone.utc).isoformat(), "season": f"{season}/{str(season + 1)[-2:]}",
        "competition": "Bundesliga (L1)", "feature_cutoff_exclusive": str(cutoff.date()),
        "valuation_window": [str(cutoff.date()), str(end.date())], "valuation_date_min": str(frame.valuation_date.min().date()),
        "valuation_date_max": str(frame.valuation_date.max().date()), "league_matches": len(league_games),
        "players_with_appearances": len(stats), "players_with_summer_target": before, "training_rows": len(frame),
        "bayern_rows": int((frame.player_club_id == 27).sum()),
        "limitations": "Historical snapshot; not a live 2026/27 roster. Position is from downloaded player profiles, not a historical position table. Summer valuation availability filters the sample.",
        "raw_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
    }
    output = Path("data/bundesliga_players.csv")
    frame.to_csv(output, index=False)
    database = Path("data/bundesliga.sqlite")
    with sqlite3.connect(database) as connection:
        frame.to_sql("training_players", connection, if_exists="replace", index=False)
        frame[frame.player_club_id == 27].to_sql("bayern_players", connection, if_exists="replace", index=False)
        league_games.to_sql("bundesliga_games", connection, if_exists="replace", index=False)
        appearances.to_sql("bundesliga_appearances", connection, if_exists="replace", index=False)
        players[players.player_id.isin(stats.player_id)].to_sql("player_profiles", connection, if_exists="replace", index=False)
        valuations[valuations.player_id.isin(stats.player_id)].to_sql("player_valuations", connection, if_exists="replace", index=False)
        pd.DataFrame([{"key": key, "value": json.dumps(value)} for key, value in metadata.items()]).to_sql("dataset_metadata", connection, if_exists="replace", index=False)
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS training_player_id ON training_players(player_id)")
    Path("data/provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Database: {database.resolve()}\nTraining CSV: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2025)
    prepare(parser.parse_args().season)
