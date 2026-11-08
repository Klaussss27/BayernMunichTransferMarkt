"""Build multi-season, five-league training observations from cached source tables."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import pandas as pd

from prepare_data import download, SOURCE
from valuation.data import validate_data


def main():
    players = pd.read_csv(download("players"))
    games = pd.read_csv(download("games"), low_memory=False)
    valuations = pd.read_csv(download("player_valuations"))
    clubs = pd.read_csv(download("clubs"))
    games["date"] = pd.to_datetime(games.date)
    valuations["date"] = pd.to_datetime(valuations.date)
    selected_games = games[games.competition_id.isin(["L1", "GB1", "ES1", "IT1", "FR1"]) & games.season.between(2016, 2025)]
    chunks = []
    for chunk in pd.read_csv(download("appearances"), chunksize=200_000):
        chunks.append(chunk[chunk.game_id.isin(selected_games.game_id)])
    appearances = pd.concat(chunks, ignore_index=True).merge(
        selected_games[["game_id", "season"]], on="game_id", validate="many_to_one")
    frames, windows = [], []
    for (season, league), matches in selected_games.groupby(["season", "competition_id"]):
        # Use all matches in the source season; targets strictly follow its last match.
        cutoff = pd.Timestamp(matches.date.max().to_pydatetime().replace(hour=0)) + pd.offsets.Day(1)
        end = pd.Timestamp(f"{season + 1}-08-31")
        if cutoff > end:
            continue
        subset = appearances[(appearances.season == season) & (appearances.competition_id == league)]
        keys = ["player_id"]
        stats = subset.groupby(keys, as_index=False)[["goals", "assists", "minutes_played"]].sum()
        club = subset.groupby(["player_id", "player_club_id"], as_index=False).minutes_played.sum()
        club = club.sort_values("minutes_played", ascending=False).drop_duplicates("player_id")
        stats = stats.merge(club[["player_id", "player_club_id"]], on="player_id", validate="one_to_one")
        summer = valuations[valuations.date.between(cutoff, end)].sort_values("date").drop_duplicates("player_id")
        frame = stats.merge(players[["player_id", "name", "date_of_birth", "position"]], on="player_id", validate="one_to_one")
        frame = frame.merge(summer[["player_id", "date", "market_value_in_eur"]], on="player_id", validate="one_to_one")
        frame = frame.rename(columns={"name": "player_name", "date": "valuation_date", "market_value_in_eur": "market_value_eur"})
        frame["age"] = ((frame.valuation_date - pd.to_datetime(frame.date_of_birth)).dt.days / 365.2425).round(2)
        frame["position"] = frame.position.replace({"Attack": "Forward", "Midfield": "Midfielder"})
        frame["season"] = int(season)
        frame["competition_id"] = league
        frame["feature_end_date"] = str(matches.date.max().date())
        frame = frame.dropna(subset=["age", "position", "market_value_eur"])
        frame = frame[frame.age.between(15, 45) & frame.position.isin(["Forward", "Midfielder", "Defender", "Goalkeeper"])]
        frames.append(frame)
        windows.append({"season": int(season), "league": league, "matches": len(matches), "feature_end": str(matches.date.max().date()), "target_start": str(cutoff.date()), "rows": len(frame)})
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.merge(clubs[["club_id", "name"]].rename(columns={"club_id": "player_club_id", "name": "club_name"}), on="player_club_id", how="left", validate="many_to_one")
    frame = frame.sort_values(["player_id", "season", "competition_id"]).reset_index(drop=True)
    validate_data(frame)
    metadata = {
        "source": SOURCE, "prepared_at": datetime.now(timezone.utc).isoformat(), "season": "2016/17–2025/26",
        "competition": "Bundesliga, Premier League, La Liga, Serie A, Ligue 1", "training_rows": len(frame),
        "unique_players": int(frame.player_id.nunique()), "league_matches": len(selected_games),
        "valuation_date_min": str(frame.valuation_date.min().date()), "valuation_date_max": str(frame.valuation_date.max().date()),
        "bayern_rows": int((frame.player_club_id == 27).sum()), "league_season_windows": windows,
        "limitations": "Historical nominal-euro targets; source updates paused in July 2026. Current profile position may differ from historical position. Summer target availability filters the sample. Ligue 1 2019/20 ended early. League and season context must match the prediction profile.",
    }
    database = Path("data/european_players.sqlite")
    with sqlite3.connect(database) as connection:
        frame.to_sql("training_players", connection, if_exists="replace", index=False)
        frame[(frame.player_club_id == 27) & (frame.season == 2025)].to_sql("bayern_players", connection, if_exists="replace", index=False)
        pd.DataFrame([{"key": key, "value": json.dumps(value)} for key, value in metadata.items()]).to_sql("dataset_metadata", connection, if_exists="replace", index=False)
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS observation_id ON training_players(player_id, season, competition_id)")
    frame.to_csv("data/european_players.csv", index=False)
    Path("data/european_provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "league_season_windows"}, indent=2))


if __name__ == "__main__":
    main()
