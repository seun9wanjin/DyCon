"""Label-only Memory using an evolving narrative of earlier EMA responses.

No sensor data enter the prompts. S1 has an empty history and uses task
instructions only; after each prediction the observed score updates memory.
Run with python -m baselines.label_only_baseline --help.
"""
import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ._common import input_files, positive_integer, score, write_json
from .llm_common import ask_llm, client

PREDICT_FIRST_PROMPT = '''You are a mental health professional.
Your task is to predict the {target} level (0-6 scale) for an individual.

# BACKGROUND
This is the first assessment session for this individual. No prior information about this individual is available.

# TASK INSTRUCTION
- "prediction_analysis": Provide your reasoning for the prediction.
- "{target}": Predict a {target} level (float 0-6).

# CONSTRAINTS
- Your {target} score must be a continuous value (not restricted to integers)
- Output in JSON format

# OUTPUT FORMAT
{{
    "prediction_analysis": "...",
    "{target}": "..."
}}'''

PREDICT_PROMPT = '''You are a mental health professional.
Your task is to predict the {target} level (0-6 scale) for an individual based on your accumulated memory of their past self-reported {target} levels.

# BACKGROUND
The personal memory below summarizes this individual's past self-reported {target} levels (Ecological Momentary Assessment scores on a 0-6 scale) across previous sessions. No sensor or behavioral data is available.

# INPUT COMPONENTS
{{
    "current_session_index": {session},
    "personal_memory": {memory}
}}

# TASK INSTRUCTION
Predict this individual's current {target} level using only the personal memory of their past self-report history.

- "prediction_analysis": Provide your reasoning for the prediction based on the memory.
- "{target}": Predict a {target} level (float 0-6).

# CONSTRAINTS
- Your {target} score must be a continuous value (not restricted to integers)
- Output in JSON format

# OUTPUT FORMAT
{{
    "prediction_analysis": "...",
    "{target}": "..."
}}'''

UPDATE_PROMPT = '''You are a mental health professional.
Your task is to update a personal memory that tracks an individual's self-reported {target} levels over time.

# BACKGROUND
The personal memory summarizes this individual's past self-reported {target} levels (0-6 scale). A new self-report has just been observed and must be integrated.

# INPUT COMPONENTS
{{
    "session_index": {session},
    "observed_{target}_level": {gt_value},
    "personal_memory": {memory}
}}

# TASK INSTRUCTION
Update the personal memory to integrate the newly observed {target} level.

- "update_rationale": Explain what was integrated.
- "updated_memory": Updated narrative memory of this individual's {target} self-report history, including level, trajectory (improving, worsening, stable), and variability. You may include numeric values.

# CONSTRAINTS
- Use only the self-report history; do not invent behavioral or demographic information
- Output in JSON format

# OUTPUT FORMAT
{{
    "update_rationale": "...",
    "updated_memory": "..."
}}'''

def process_user(csv_path, target, output_dir):
    pid = Path(csv_path).stem
    user_dir = Path(output_dir) / target / pid
    pred_dir = user_dir / "prediction"
    upd_dir = user_dir / "update"
    df = pd.read_csv(csv_path, usecols=["pid", target])
    values = [score(value) for value in df[target]]
    memory = ""
    # Replay completed sessions in order. Reusing a later memory for an earlier
    # missing prediction would leak future information into that prediction.
    for s, gt in enumerate(values):
        pred_path = pred_dir / f"{pid}_predict_session_{s}.json"
        mem_path = upd_dir / f"{pid}_memory_session_{s}.json"
        if not pred_path.exists():
            if s == 0:
                prompt = PREDICT_FIRST_PROMPT.format(target=target)
            else:
                prompt = PREDICT_PROMPT.format(target=target, session=s,
                                               memory=json.dumps(memory))
            out = ask_llm(prompt)
            if out is None:
                return False
            prediction = score(out.get(target))
            write_json(pred_path, {"user": pid, "session": s, "gt_value": gt,
                                  "prediction": prediction,
                                  "analysis": out.get("prediction_analysis"),
                                  "memory": memory})
        else:
            saved = json.loads(pred_path.read_text(encoding="utf-8"))
            if saved["session"] != s or score(saved["gt_value"]) != gt or saved.get("memory") != memory:
                raise ValueError("Saved prediction does not match this input/history; use a fresh output directory")
            score(saved["prediction"])
        # The current score is supplied only after the initial prediction.
        if not mem_path.exists():
            prompt = UPDATE_PROMPT.format(target=target, session=s, gt_value=gt,
                                          memory=json.dumps(memory))
            out = ask_llm(prompt)
            if out is None:
                return False
            memory = out.get("updated_memory", "")
            if not isinstance(memory, str):
                raise ValueError("Expected a textual memory")
            write_json(mem_path, {"user": pid, "session": s, "memory": memory,
                                 "rationale": out.get("update_rationale")})
        else:
            memory = json.loads(mem_path.read_text(encoding="utf-8"))["memory"]
            if not isinstance(memory, str):
                raise ValueError("Expected a textual memory")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", required=True, choices=["anxiety", "depression"])
    parser.add_argument("--workers", type=positive_integer, default=1)
    parser.add_argument("--limit", type=positive_integer)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    csvs = input_files(args.data_dir, args.limit)
    client()  # Validate credentials before starting workers.
    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_user, path, args.target, args.output_dir) for path in csvs]
        for future in as_completed(futures):
            try:
                ok = future.result()
            except Exception as error:
                logging.error("Participant processing failed (%s)", type(error).__name__)
                ok = False
            done += int(ok)
            failed += int(not ok)
    logging.info("Finished: %d completed, %d failed", done, failed)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
