"""Synthetic, offline checks; no study data or live API requests are used."""
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from baselines import baselines as heuristics
from baselines import fewshot_baseline as fewshot
from baselines import label_only_baseline as memory
from baselines import llm_common
from evaluation import evaluate


class BaselineChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="baseline-code-check-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.csv = self.directory / "synthetic.csv"
        sensor_key = "Count of unlock episodes at exercise locations during the afternoon"
        pd.DataFrame({"pid": ["synthetic"] * 5,
                      "anxiety": [1, 4, 2, 5, 3],
                      "depression": [5, 2, 4, 1, 3],
                      sensor_key: ["[1, 2, 3]"] * 5}).to_csv(self.csv, index=False)

    def test_heuristics_use_prior_responses_and_omit_first_session(self):
        out = self.directory / "heuristics"
        self.assertEqual(heuristics.process_user(self.csv, "anxiety", out), 8)
        lag = out / "lag1" / "anxiety" / "synthetic" / "prediction"
        hist = out / "historical_mean" / "anxiety" / "synthetic" / "prediction"
        self.assertFalse((lag / "synthetic_predict_session_0.json").exists())
        self.assertEqual(json.loads((lag / "synthetic_predict_session_2.json").read_text())["prediction"], 4)
        self.assertEqual(json.loads((hist / "synthetic_predict_session_2.json").read_text())["prediction"], 2.5)
        result = evaluate(out / "lag1" / "anxiety")
        self.assertEqual(set(result.stage), {"R0"})
        self.assertEqual(list(result.session), [2, 3, 4, 5])

    def test_recent_three_examples_never_include_current_response(self):
        jobs = fewshot.build_jobs(self.csv, "anxiety")
        self.assertEqual([len(job["examples"]) for job in jobs], [0, 1, 2, 3, 3])
        self.assertEqual([example["session"] for example in jobs[-1]["examples"]], [1, 2, 3])
        self.assertTrue(all("anxiety" not in job["data"] and "depression" not in job["data"] for job in jobs))
        for job in jobs:
            self.assertTrue(all(example["session"] < job["session"] for example in job["examples"]))
        out = self.directory / "fewshot"
        with patch.object(fewshot, "ask_llm", return_value={"anxiety": 2.0, "prediction_analysis": "synthetic"}) as api:
            self.assertTrue(all(fewshot.run_job(job, "anxiety", out) for job in jobs))
            self.assertEqual(api.call_count, 5)
        self.assertEqual(len(evaluate(out / "anxiety")), 5)

    def test_memory_predicts_before_update_and_resumes_without_calls(self):
        calls = []

        def fake_llm(prompt):
            calls.append(prompt)
            if "Your task is to update a personal memory" in prompt:
                return {"updated_memory": "synthetic memory " + str(len(calls)),
                        "update_rationale": "synthetic"}
            return {"anxiety": 2.0, "prediction_analysis": "synthetic"}

        out = self.directory / "memory"
        with patch.object(memory, "ask_llm", fake_llm):
            self.assertTrue(memory.process_user(self.csv, "anxiety", out))
        self.assertEqual(len(calls), 10)
        self.assertTrue(all("observed_anxiety_level" not in call for call in calls[::2]))
        self.assertTrue(all("observed_anxiety_level" in call for call in calls[1::2]))
        self.assertEqual(calls[0], memory.PREDICT_FIRST_PROMPT.format(target="anxiety"))
        with patch.object(memory, "ask_llm", side_effect=AssertionError("Unexpected API request")):
            self.assertTrue(memory.process_user(self.csv, "anxiety", out))
        self.assertEqual(len(evaluate(out / "anxiety")), 5)

    def test_no_credentials_file_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                llm_common.get_api_key()

    def test_error_logs_do_not_contain_request_or_response_bodies(self):
        class BrokenCompletions:
            def create(self, **kwargs):
                raise RuntimeError("PRIVATE_REQUEST_BODY_SENTINEL")

        class BrokenChat:
            completions = BrokenCompletions()

        class BrokenClient:
            chat = BrokenChat()

        with patch.object(llm_common, "client", return_value=BrokenClient()), patch.object(logging, "warning") as warning:
            self.assertIsNone(llm_common.ask_llm("synthetic request", retries=0))
            self.assertNotIn("PRIVATE_REQUEST_BODY_SENTINEL", str(warning.call_args))

    def test_command_line_requires_explicit_local_paths(self):
        root = Path(__file__).resolve().parents[1]
        for module in ("baselines.baselines", "baselines.label_only_baseline", "baselines.fewshot_baseline"):
            with self.subTest(module=module):
                help_result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=root,
                                             capture_output=True, text=True)
                self.assertEqual(help_result.returncode, 0)
                missing = subprocess.run([sys.executable, "-m", module, "--target", "anxiety"], cwd=root,
                                         capture_output=True, text=True)
                self.assertEqual(missing.returncode, 2)
                self.assertIn("--data-dir", missing.stderr)
                self.assertIn("--output-dir", missing.stderr)


if __name__ == "__main__":
    unittest.main()
