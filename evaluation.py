"""Session-wise metrics from local DyCon output. No result data are distributed.

Balanced accuracy is calculated from the pooled labels and predictions at each
session/stage, not by averaging one-observation accuracy values. A session with
only one observed class has undefined binary balanced accuracy (reported as NaN).
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error


def score(value):
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 6:
        raise ValueError("Expected a finite PHQ-4 subscale score in [0, 6]")
    return result


def evaluate(base_path, max_refines=10):
    """Return one row per zero-based session and prediction/refinement stage."""
    if max_refines < 0:
        raise ValueError("max_refines must be nonnegative")
    groups = defaultdict(list)
    for path in sorted(Path(base_path).glob("*/prediction/*_predict_session_*.json")):
        initial = json.loads(path.read_text())
        session = int(initial["session"])
        user_dir = path.parent.parent
        user = user_dir.name
        truth, first = score(initial["gt_value"]), score(initial["prediction"])
        groups[(session, "R0")].append((truth, first))
        refine_path = user_dir / "refine" / f"{user}_refine_session_{session}.json"
        if not refine_path.is_file():
            # Missing logs may mean an interrupted run, not successful refinement.
            continue
        refinements = json.loads(refine_path.read_text())
        if not isinstance(refinements, list) or len(refinements) > max_refines:
            raise ValueError("Invalid refinement log length/type")
        for record in refinements:
            if score(record["gt_value"]) != truth:
                raise ValueError("Ground truth differs between prediction and refinement logs")
        predictions = [score(record["prediction"]) for record in refinements]
        final = predictions[-1] if predictions else first
        groups[(session, "R*")].append((truth, final))
        for step in range(1, max_refines + 1):
            # Retain the final prediction after early stopping, as in the source evaluator.
            pred = predictions[step - 1] if step <= len(predictions) else final
            groups[(session, f"R{step}")].append((truth, pred))
    if not groups:
        raise ValueError("No prediction files found in the specified local result directory")
    records = []
    for (session, stage), pairs in sorted(groups.items()):
        truth, prediction = zip(*pairs)
        labels = [int(v >= 3) for v in truth]
        predicted_labels = [int(v >= 3) for v in prediction]
        ba = (balanced_accuracy_score(labels, predicted_labels)
              if len(set(labels)) == 2 else float("nan"))
        records.append(dict(session=session + 1, stage=stage, n=len(pairs),
                            n_negative=labels.count(0), n_positive=labels.count(1),
                            balanced_accuracy=ba, mae=mean_absolute_error(truth, prediction)))
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-refines", type=int, default=10)
    args = parser.parse_args()
    result = evaluate(args.base_path, args.max_refines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} aggregate metric rows locally.")


if __name__ == "__main__":
    main()
