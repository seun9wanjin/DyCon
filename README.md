# DyCon

Implementation of **Dynamic Context Modeling for Longitudinal Mental Health Monitoring under Distribution Shift** (NeurIPS 2026).

DyCon refines momentary context using EMA feedback and retains longitudinal context for subsequent predictions without updating model parameters. This repository contains the GLOBEM anxiety/depression implementation and four comparison baselines.

## Installation

Requires Python 3.10 or later.

```bash
git clone https://github.com/seun9wanjin/DyCon.git
cd DyCon
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set `DEEPSEEK_API_KEY` in your environment. For other backbones, use `OPENAI_API_KEY` or `LLAMA_API_KEY`. Optional `DYCON_BASE_URL` and `DYCON_MODEL_ID` specify the endpoint and model version.

## Input format

Prepare one CSV per participant-year sequence, with rows ordered chronologically. The filename identifies the sequence. Raw-data preprocessing is not included.

| Column | Format |
| --- | --- |
| `anxiety` or `depression` | Target subscale score, 0–6. |
| `correlation` | Text representation of sensor relationships within the current observation window. |
| Sensor features | One column per feature, each containing a serialized numeric list of daily values. Lists within a row must have equal lengths. |
| `pid`, `date`, `index`, `year`, `phq4` | Optional metadata, excluded from sensor inputs. |

Sensor windows and correlation summaries must exclude current/future EMA responses and future observations. Supply your own data; keep data, generated outputs, and API keys outside the repository.

## Running experiments

Validate an input, then run DyCon:

```bash
python validate_input.py --input_csv /path/to/sessions.csv --target anxiety

python run_experiment.py \
  --input_csv /path/to/sessions.csv \
  --target anxiety --model deepseek-chat --refine_thr 0.5 \
  --output_dir /path/to/output/dycon/anxiety
```

Use `--target depression` for depression. Other model choices are `gpt-4o-mini` and `llama3.1-8b-instruct`; `--max_refines` defaults to 10. Use a separate output directory for each configuration.

The following commands run Lag-1/Historical Mean, Label-only Memory, and Per-user Recent 3-shot over a directory of CSV files:

```bash
python -m baselines.baselines --data-dir /path/to/csvs --target anxiety --output-dir /path/to/output/history
python -m baselines.label_only_baseline --data-dir /path/to/csvs --target anxiety --output-dir /path/to/output/memory
python -m baselines.fewshot_baseline --data-dir /path/to/csvs --target anxiety --output-dir /path/to/output/recent
```

The LLM baselines use `DEEPSEEK_API_KEY`, with optional `LLM_MODEL` and `LLM_BASE_URL` overrides.

## Evaluation

Pass the directory containing the participant-sequence folders:

```bash
python evaluation.py /path/to/output/dycon/anxiety --output /path/to/metrics.csv
```

The evaluator reports session-wise balanced accuracy and MAE. Balanced accuracy averages the two class recalls using `score >= 3` as the positive label; MAE uses continuous scores. Groups containing only one class have undefined balanced accuracy (`NaN`).

R0 predictions precede current-session EMA feedback. R1–R10 and R* use that feedback and are reported separately; after early stopping, later stages retain the final prediction.

For baseline evaluation, use `/path/to/output/history/lag1/anxiety`, `/path/to/output/history/historical_mean/anxiety`, `/path/to/output/memory/anxiety`, or `/path/to/output/recent/anxiety`. Compare methods on matching records.
