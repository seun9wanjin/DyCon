# Comparison methods

This directory contains code only. Supply your own authorized local input data;
no sensor observations, labels, participant records, model responses, or measured
results are included. Run commands from the repository root.

Inputs follow the same per-user CSV representation as the core implementation:
one CSV per user record, rows ordered chronologically, and `pid`, `anxiety`, and
`depression` columns. Scores use the 0–6 subscale. The few-shot method also reads
the sensor-feature columns; their values are serialized lists over the session
window. Metadata and both target columns are excluded from sensor inputs.

```sh
python -m baselines.baselines --data-dir /path/to/local/csvs --output-dir /path/to/local/heuristics --target anxiety
python -m baselines.label_only_baseline --data-dir /path/to/local/csvs --output-dir /path/to/local/label-memory --target anxiety
python -m baselines.fewshot_baseline --data-dir /path/to/local/csvs --output-dir /path/to/local/recent-three --target anxiety
```

- **Lag-1** predicts the immediately preceding score; **Historical Mean** predicts
  the mean of all preceding scores. Neither method produces an S1 prediction.
- **Label-only Memory** predicts from a narrative memory of preceding responses,
  then updates that memory using the newly observed response. It never receives
  sensor data. The preserved experimental S1 prompt has no individual history.
- **Per-user Recent 3-shot** receives the current sensor window and up to three
  preceding sensor-summary/response pairs from the same record. At S1 there are
  no examples; S2 and S3 use one and two examples, respectively. The current
  response is saved for evaluation but is not sent in the prediction prompt.
  `--recent-k` changes the number of preceding examples; its default is 3.

The two LLM methods default to one worker (`--workers 1`). They use an
OpenAI-compatible API configured exclusively through environment variables:
`DEEPSEEK_API_KEY`, optional `LLM_MODEL` (default `deepseek-chat`), optional
`LLM_BASE_URL` (default `https://api.deepseek.com`), and optional `LLM_KEY_ENV`
(the name of an alternative key environment variable). Set these in your own
shell; do not put credentials in source files. Running the LLM methods sends the
specified prompt inputs to the configured endpoint. Model access and applicable
service terms depend on the endpoint selected by the person running the code.

Prediction files use the core evaluator's `user`, zero-based `session`,
`gt_value`, and `prediction` fields. The heuristic layout is
`<output>/<lag1|historical_mean>/<target>/<user>/prediction/`; the LLM layout is
`<output>/<target>/<user>/prediction/`. Label-only memory also stores local
updates under each user's `update/` directory. These runtime files contain
private inputs or derived outputs and are not release assets.

```sh
python evaluation.py /path/to/local/heuristics/lag1/anxiety --output /path/to/local/lag1-metrics.csv
python evaluation.py /path/to/local/label-memory/anxiety --output /path/to/local/label-memory-metrics.csv
python evaluation.py /path/to/local/recent-three/anxiety --output /path/to/local/recent-three-metrics.csv
```

Only R0 is available for these comparison methods. No oracle-majority metrics
are produced. Use a separate output directory for each model, target, dataset,
and setting; resume checks do not establish that all sensor inputs or endpoint
settings match an earlier run. Original experimental prompt text is preserved;
these prompts do not establish clinical validity or a diagnostic use case.
