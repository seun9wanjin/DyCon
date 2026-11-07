"""Local input/output helpers; no dataset or outputs are distributed."""
import json
import math
from pathlib import Path


def positive_integer(value):
    import argparse
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def score(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 6:
        raise ValueError("Expected a finite PHQ-4 subscale score in [0, 6]")
    return number


def input_files(data_dir, limit=None):
    folder = Path(data_dir)
    if not folder.is_dir():
        raise ValueError("The data directory must exist")
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise ValueError("No input CSV files found")
    return files[:limit] if limit is not None else files


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only after serialization succeeds, so interrupted writes are not
    # mistaken for completed predictions when a run is resumed.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)
