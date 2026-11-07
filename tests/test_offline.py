"""Offline regression checks using synthetic values only; no API calls or study data."""

import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluation
import run_experiment


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_output(self, case, truth, prediction, refinements=None, session=0):
        """Write artificial output records in a temporary directory.

        None means the refinement file is absent; [] means a completed session
        stopped before refinement. These two states must remain distinguishable.
        """
        directory = self.root / f"synthetic_{case}"
        (directory / "prediction").mkdir(parents=True, exist_ok=True)
        initial = {"session": session, "gt_value": truth, "prediction": prediction}
        (directory / "prediction" / f"{directory.name}_predict_session_{session}.json").write_text(
            json.dumps(initial), encoding="utf-8"
        )
        if refinements is not None:
            (directory / "refine").mkdir(exist_ok=True)
            records = [{"gt_value": truth, "prediction": value} for value in refinements]
            (directory / "refine" / f"{directory.name}_refine_session_{session}.json").write_text(
                json.dumps(records), encoding="utf-8"
            )

    def row(self, result, stage, session=1):
        rows = result[(result.stage == stage) & (result.session == session)]
        self.assertEqual(len(rows), 1)
        return rows.iloc[0]

    def test_balanced_accuracy_is_not_accuracy_on_imbalanced_labels(self):
        # Five negative observations and one positive, all predicted negative.
        for case, truth in enumerate([0, 0, 0, 0, 0, 6]):
            self.write_output(case, truth, 0, [])
        row = self.row(evaluation.evaluate(self.root), "R0")
        self.assertEqual(row["n"], 6)
        self.assertEqual(row["n_negative"], 5)
        self.assertEqual(row["n_positive"], 1)
        self.assertAlmostEqual(row["balanced_accuracy"], 0.5)
        self.assertNotAlmostEqual(row["balanced_accuracy"], 5 / 6)
        self.assertAlmostEqual(row["mae"], 1.0)

    def test_r0_is_kept_separate_from_label_guided_refinement(self):
        self.write_output(0, 0, 6, [0])
        self.write_output(1, 6, 0, [6])
        result = evaluation.evaluate(self.root, max_refines=2)
        initial = self.row(result, "R0")
        final = self.row(result, "R*")
        self.assertAlmostEqual(initial["balanced_accuracy"], 0.0)
        self.assertAlmostEqual(initial["mae"], 6.0)
        self.assertAlmostEqual(final["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(final["mae"], 0.0)
        # An early-stopped final prediction carries forward to later stages.
        self.assertAlmostEqual(self.row(result, "R2")["balanced_accuracy"], 1.0)

    def test_missing_refinement_log_does_not_create_final_predictions(self):
        self.write_output(0, 0, 0, refinements=None)
        self.write_output(1, 6, 6, refinements=None)
        result = evaluation.evaluate(self.root)
        self.assertEqual(set(result.stage), {"R0"})
        self.assertEqual(self.row(result, "R0")["n"], 2)

    def test_final_denominator_excludes_incomplete_sessions(self):
        self.write_output(0, 0, 0, [])
        self.write_output(1, 6, 6, None)
        result = evaluation.evaluate(self.root)
        self.assertEqual(self.row(result, "R0")["n"], 2)
        self.assertEqual(self.row(result, "R*")["n"], 1)

    def test_single_observed_class_has_undefined_binary_balanced_accuracy(self):
        self.write_output(0, 0, 0, [])
        self.write_output(1, 2, 1, [])
        result = evaluation.evaluate(self.root)
        for stage in ("R0", "R*"):
            row = self.row(result, stage)
            self.assertTrue(math.isnan(row["balanced_accuracy"]))
            self.assertAlmostEqual(row["mae"], 0.5)

    def test_inconsistent_refinement_ground_truth_is_rejected(self):
        self.write_output(0, 0, 1, [0])
        path = self.root / "synthetic_0" / "refine" / "synthetic_0_refine_session_0.json"
        path.write_text(json.dumps([{"gt_value": 6, "prediction": 0}]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Ground truth differs"):
            evaluation.evaluate(self.root)


class PredictionTimingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "synthetic_case"
        for subdirectory in ("prediction", "refine", "update"):
            (self.root / subdirectory).mkdir(parents=True, exist_ok=True)
        guard = patch("src.llm.get_client", side_effect=AssertionError("API calls are disabled in offline tests"))
        guard.start()
        self.addCleanup(guard.stop)
        self.arguments = dict(
            user="synthetic_case", session=1, features={"synthetic_sensor": [1, 2]},
            correlation="synthetic sensor correlation", gt_value=0.0, days=2,
            target="depression", momentary_context="initial synthetic context",
            longitudinal_context="prior synthetic context", refine_skip_thr=0.5,
            refine_enabled=True, skip_thr_enabled=True, max_refines=1,
            user_dir=str(self.root), config=SimpleNamespace(),
        )

    def load(self, subdirectory):
        paths = list((self.root / subdirectory).glob("*.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text(encoding="utf-8"))

    def test_initial_prediction_is_saved_before_label_guided_refinement(self):
        events = []

        def predict(context, features, session, days, target, **kwargs):
            events.append("predict")
            self.assertEqual(features, self.arguments["features"])
            self.assertNotIn("gt_value", kwargs)
            self.assertEqual(kwargs["longitudinal_context"], "prior synthetic context")
            return ("synthetic rationale", 4.0 if context == "initial synthetic context" else 0.25)

        def refine(context, prediction, truth, *args, **kwargs):
            events.append("refine")
            self.assertEqual(prediction, 4.0)
            self.assertEqual(truth, 0.0)
            # R0 already exists, before the feedback can alter the context.
            self.assertEqual(self.load("prediction")["prediction"], 4.0)
            return "synthetic refinement", "refined synthetic context"

        def update(**kwargs):
            events.append("update")
            self.assertEqual(kwargs["prediction"], 0.25)
            self.assertEqual(kwargs["momentary_context"], "refined synthetic context")
            self.assertEqual(kwargs["longitudinal_context"], "prior synthetic context")
            return "synthetic update", "updated synthetic context"

        with patch.object(run_experiment, "predict_score", side_effect=predict), \
             patch.object(run_experiment, "refine_context", side_effect=refine), \
             patch.object(run_experiment, "update_context", side_effect=update):
            returned_context = run_experiment.process_session_with_update(**self.arguments)

        self.assertEqual(events, ["predict", "refine", "predict", "update"])
        self.assertEqual(self.load("prediction")["prediction"], 4.0)
        self.assertEqual(self.load("refine")[0]["prediction"], 0.25)
        self.assertTrue(self.load("update")["update"])
        self.assertEqual(returned_context, "updated synthetic context")

    def test_failed_error_gate_retains_prior_context(self):
        with patch.object(run_experiment, "predict_score", return_value=("synthetic rationale", 4.0)) as predict, \
             patch.object(run_experiment, "refine_context", return_value=("synthetic refinement", "refined context")), \
             patch.object(run_experiment, "update_context") as update:
            returned_context = run_experiment.process_session_with_update(**self.arguments)
        self.assertEqual(predict.call_count, 2)
        update.assert_not_called()
        self.assertEqual(returned_context, "prior synthetic context")
        self.assertFalse(self.load("update")["update"])
        self.assertEqual(self.load("update")["longitudinal_context"], "prior synthetic context")

    def test_prediction_at_error_threshold_updates_without_refinement(self):
        with patch.object(run_experiment, "predict_score", return_value=("synthetic rationale", 0.5)) as predict, \
             patch.object(run_experiment, "refine_context") as refine, \
             patch.object(run_experiment, "update_context", return_value=("synthetic update", "updated context")) as update:
            returned_context = run_experiment.process_session_with_update(**self.arguments)
        predict.assert_called_once()
        refine.assert_not_called()
        update.assert_called_once()
        self.assertEqual(self.load("refine"), [])
        self.assertTrue(self.load("update")["update"])
        self.assertEqual(returned_context, "updated context")


if __name__ == "__main__":
    unittest.main()
