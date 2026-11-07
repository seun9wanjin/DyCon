"""Per-user Recent 3-shot using current sensing data and prior examples.

The example count is min(3, number of prior sessions) by default. At S1, task
instructions and current sensing data are supplied without prior examples.
Run with python -m baselines.fewshot_baseline --help.
"""
import argparse
import ast
import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ._common import input_files, positive_integer, score, write_json
from .llm_common import ask_llm, client

EXCLUDED = ['index', 'pid', 'date', 'year', 'phq4', 'depression', 'anxiety', 'correlation']

PROMPT = '''You are a mental health professional.
Your task is to predict the {target} level (0-6 scale) for an individual from their sensor data, using their own past sessions as in-context examples.

# BACKGROUND
Each past example contains a summary of this individual's sensor data (per-feature averages over the session window) together with their self-reported {target} level (0-6 scale) for that session. Use these examples to learn how this individual's behavioral patterns relate to their {target} levels.

# INPUT COMPONENTS
{{
    "past_examples": {examples},
    "current_sensor_data_over_past_{days}_days": {data}
}}

# TASK INSTRUCTION
Interpret the current sensor data in light of this individual's past sensor-label examples, considering DSM-5 criteria.

- "prediction_analysis":
  1. Compare current sensor patterns against the past examples
  2. Identify which past sessions the current behavior most resembles
  3. Provide clinical reasoning for your prediction

- "{target}": Predict a {target} level (float 0-6)

# CONSTRAINTS
- Your {target} score must be a continuous value (not restricted to integers)
- Output in JSON format

# OUTPUT FORMAT
{{
    "prediction_analysis": "...",
    "{target}": "..."
}}'''

def parse_day_list(s):
    try:
        vals = ast.literal_eval(str(s).replace("nan", "None"))
        return [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    except (ValueError, SyntaxError):
        return []

def summarize_features(row):
    """Per-feature mean over the session window -> compact dict."""
    out = {}
    for col, val in row.items():
        vals = parse_day_list(val)
        out[col] = round(sum(vals) / len(vals), 3) if vals else None
    return out

def build_jobs(csv_path, target, recent_k=3):
    pid = os.path.splitext(os.path.basename(csv_path))[0]
    df = pd.read_csv(csv_path)
    feats = df.drop(columns=[c for c in EXCLUDED if c in df.columns])
    summaries = [summarize_features(feats.iloc[s].to_dict()) for s in range(len(df))]
    jobs = []
    for s in range(len(df)):
        past = list(range(max(0, s - recent_k), s))
        examples = [
            {"session": p, "sensor_summary": summaries[p],
             f"self_reported_{target}": score(df.iloc[p][target])}
            for p in past
        ]
        row = feats.iloc[s].to_dict()
        try:
            days = len(ast.literal_eval(str(row['Count of unlock episodes at exercise locations during the afternoon'])))
        except (ValueError, SyntaxError, KeyError):
            days = 7
        jobs.append({
            "pid": pid, "session": s, "gt": score(df.iloc[s][target]),
            "examples": examples, "data": row, "days": days,
        })
    return jobs

def run_job(job, target, output_dir):
    pid, session = job["pid"], job["session"]
    pred_path = Path(output_dir) / target / pid / "prediction" / f"{pid}_predict_session_{session}.json"
    if pred_path.exists():
        saved = json.loads(pred_path.read_text(encoding="utf-8"))
        if saved["session"] != session or score(saved["gt_value"]) != job["gt"] or saved.get("n_examples") != len(job["examples"]):
            raise ValueError("Saved prediction does not match this input/settings; use a fresh output directory")
        score(saved["prediction"])
        return True
    prompt = PROMPT.format(
        target=target,
        examples=json.dumps(job["examples"]),
        days=job["days"],
        data=json.dumps({key: str(value) for key, value in job["data"].items()}),
    )
    out = ask_llm(prompt)
    if out is None:
        return False
    write_json(pred_path, {"user": pid, "session": session, "gt_value": job["gt"],
                           "prediction": score(out.get(target)),
                           "analysis": out.get("prediction_analysis"),
                           "n_examples": len(job["examples"])})
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", required=True, choices=["anxiety", "depression"])
    parser.add_argument("--workers", type=positive_integer, default=1)
    parser.add_argument("--limit", type=positive_integer)
    parser.add_argument("--recent-k", "--recent_k", type=positive_integer, default=3,
                        help="Number of most recent preceding examples (default: 3)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    csvs = input_files(args.data_dir, args.limit)
    jobs = []
    for path in csvs:
        jobs.extend(build_jobs(path, args.target, args.recent_k))
    client()
    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, job, args.target, args.output_dir) for job in jobs]
        for future in as_completed(futures):
            try:
                ok = future.result()
            except Exception as error:
                logging.error("Session processing failed (%s)", type(error).__name__)
                ok = False
            done += int(ok)
            failed += int(not ok)
    logging.info("Finished: %d completed, %d failed", done, failed)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
