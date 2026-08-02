from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lightgbm_baseline import accuracy, forecast_lightgbm


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "lightgbm_results.json"
    )
    args = parser.parse_args()
    results = {}
    for path in sorted(args.data_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        train = frame.loc[frame["split"] == "train", "value"].to_numpy()
        test = frame.loc[frame["split"] == "test", "value"].to_numpy()
        frequency = path.stem.split("_", 1)[0]
        period = 12 if frequency == "monthly" else 52
        output = forecast_lightgbm(train, len(test), period)
        results[path.stem] = {
            **output.as_dict(),
            **accuracy(test, output.mean),
            "actual": test.tolist(),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
