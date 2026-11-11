"""Usage: python train.py [--data players.csv] [--output artifacts/model.joblib]."""

import argparse
import json
from pathlib import Path

import joblib

from valuation.data import demo_data, load_data, load_database
from valuation.model import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--data", help="CSV with independent player observations")
    sources.add_argument(
        "--database", help="SQLite database created by prepare_data.py"
    )
    parser.add_argument("--output", default="artifacts/model.joblib")
    parser.add_argument(
        "--task", choices=["valuation", "next-season"], default="valuation"
    )
    args = parser.parse_args()
    if args.database:
        frame = load_database(args.database)
    elif args.data:
        frame = load_data(args.data)
    else:
        frame = demo_data()
    bundle = train(frame)
    bundle["metadata"]["source"] = args.database or args.data or "SYNTHETIC DEMO"
    bundle["metadata"]["task"] = args.task
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output, compress=3)
    bundle["metrics"].to_csv(output.with_suffix(".metrics.csv"))
    bundle["split_assignments"].to_csv(output.with_suffix(".splits.csv"), index=False)
    output.with_suffix(".json").write_text(
        json.dumps(bundle["metadata"], indent=2), encoding="utf-8"
    )
    print(bundle["metrics"].round(3).to_string())
    print(f"Selected: {bundle['selected']}. Saved to {output}")


if __name__ == "__main__":
    main()
