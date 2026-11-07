"""Validate locally supplied, preprocessed session tables without downloading data."""
import ast
import math

import pandas as pd


def feature_values(value):
    if isinstance(value, str):
        values = ast.literal_eval(value)
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise ValueError("Sensor features must be serialized numeric daily-value lists")
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("Sensor feature lists must be nonempty")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in values):
        raise ValueError("Sensor lists must contain finite numeric values after preprocessing")
    return values


def window_days(features):
    lengths = {len(feature_values(value)) for value in features.values()}
    if len(lengths) != 1:
        raise ValueError("All sensor lists in a session must have the same nonzero length")
    return lengths.pop()


def load_session_table(path, target, excluded_cols):
    frame = pd.read_csv(path)
    if frame.empty or target not in frame or "correlation" not in frame:
        raise ValueError("Input requires sessions, the target score, and a correlation column")
    if "pid" in frame and frame["pid"].nunique(dropna=False) != 1:
        raise ValueError("Each input file must contain exactly one participant-year sequence")
    if "date" in frame:
        dates = pd.to_datetime(frame["date"], errors="raise")
        if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
            raise ValueError("Session dates must be present, unique, and in chronological order")
    scores = pd.to_numeric(frame[target], errors="raise")
    if not scores.between(0, 6).all():
        raise ValueError("Target scores must be finite and within [0, 6]")
    if frame["correlation"].isna().any():
        raise ValueError("Sensor-only correlation summaries must not be missing")
    features = frame.drop(columns=excluded_cols, errors="ignore")
    if not len(features.columns):
        raise ValueError("No sensor feature columns found")
    for _, row in features.iterrows():
        window_days(row.to_dict())
    return frame
