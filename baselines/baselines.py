"""Lag-1 and Historical Mean using only earlier EMA scores.

S1 is omitted because no previous response exists. Scores remain continuous;
root evaluation.py applies the same >=3 binary threshold as for DyCon.
Run with python -m baselines.baselines --help.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ._common import input_files, positive_integer, score, write_json


def process_user(csv_path, target, output_dir):
    pid = Path(csv_path).stem
    frame = pd.read_csv(csv_path, usecols=[target])
    values = np.asarray([score(value) for value in frame[target]])
    count = 0
    for session in range(1, len(values)):
        predictions = {"lag1": float(values[session - 1]),
                       "historical_mean": float(np.mean(values[:session]))}
        for method, prediction in predictions.items():
            path = Path(output_dir) / method / target / pid / "prediction" / f"{pid}_predict_session_{session}.json"
            write_json(path, {"user": pid, "session": session,
                              "gt_value": float(values[session]), "prediction": prediction})
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", required=True, choices=["anxiety", "depression"])
    parser.add_argument("--limit", type=positive_integer)
    args = parser.parse_args()
    csvs = input_files(args.data_dir, args.limit)
    count = sum(process_user(path, args.target, args.output_dir) for path in csvs)
    print(f"Saved {count} local prediction files.")


if __name__ == "__main__":
    main()
