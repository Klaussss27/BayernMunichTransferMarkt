"""Batch prediction with a trusted, locally trained model artifact."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from valuation.model import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="artifacts/model.joblib",
        help="Trusted model file from train.py",
    )
    parser.add_argument("--data", required=True, help="Player profiles CSV")
    parser.add_argument("--output", default="artifacts/predictions.csv")
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    result = predict(bundle, pd.read_csv(args.data))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {len(result)} valuations to {output}")
    print(f"Training source: {bundle['metadata'].get('source', 'unspecified')}")


if __name__ == "__main__":
    main()
