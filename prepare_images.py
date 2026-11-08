"""Cache real player portraits linked by the downloaded player dataset."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pandas as pd
import requests


def main():
    players = pd.read_csv("data/raw/players.csv.gz")
    data = pd.read_csv("data/european_players.csv")
    latest = data[data.season == data.season.max()]
    featured = set(latest[latest.player_club_id == 27].player_id) | set(latest.nlargest(24, "market_value_eur").player_id)
    portraits = players[players.player_id.isin(featured)][["player_id", "image_url"]].dropna()
    folder = Path("assets/players")
    folder.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        path = folder / f"{row.player_id}.jpg"
        try:
            if not path.exists():
                response = requests.get(row.image_url, timeout=20, headers={"User-Agent": "BayernValuationPrototype/1.0"})
                response.raise_for_status()
                if not response.headers.get("Content-Type", "").startswith("image/"):
                    return None
                path.write_bytes(response.content)
            return str(row.player_id), {"path": path.as_posix(), "source": row.image_url}
        except requests.RequestException:
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        manifest = dict(item for item in pool.map(fetch, portraits.itertuples(index=False)) if item)
    Path("assets/player_images.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Cached {len(manifest)} real player portraits.")


if __name__ == "__main__":
    main()
