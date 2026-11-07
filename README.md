# DyCon

Code for **Dynamic Context Modeling for Longitudinal Mental Health Monitoring under Distribution Shift** (NeurIPS 2026).

[Paper and discussion](https://openreview.net/forum?id=YFPmKTRY1L)

DyCon performs longitudinal personalization through natural-language momentary and longitudinal contexts. Initial predictions (R0) precede current-session EMA feedback. Feedback then guides momentary refinement, and an error criterion gates incorporation into the context used in subsequent sessions. Backbone parameters remain fixed.

## Scope of this release

This release contains the GLOBEM anxiety/depression implementation from the rebuttal code, its five prompt templates, the four label-history/recent-example controls, and a corrected session-wise evaluator. It also includes input validation, environment-based credentials, and offline tests.

**No datasets, participant records, example observations, predictions, generated contexts, experiment logs, model weights, or API keys are included.** Obtain any data independently from its original distributor under the applicable access requirements. Store inputs and outputs outside this repository. The code does not download datasets.

This is a code release, not a bundle of the paper's archived runs. It does not include every backbone/dataset adaptation, reflection implementation, preprocessing pipeline, or statistical-analysis script used in the paper. Running this code is not a guarantee of reproducing a table exactly: data eligibility, preprocessing, model snapshots, and API behavior must also match. See [implementation notes](docs/IMPLEMENTATION.md).

## Installation

Use Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Provide credentials through your shell or secret manager using `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, or `LLAMA_API_KEY`. Never put credentials in a tracked file. Optional `DYCON_BASE_URL` and `DYCON_MODEL_ID` select an OpenAI-compatible endpoint and its exact model identifier. `DYCON_OPENROUTER=1` selects the original optional OpenRouter route and requires `OPENROUTER_API_KEY`.

Endpoint selection is explicit: inference sends the supplied prompt to that endpoint. Use only an execution environment and provider configuration permitted by your dataset's terms. The offline tests do not contact model services.

## Local input preparation

The runner expects **one already-preprocessed CSV per participant-year sequence**, with one chronologically ordered row per EMA session. See [the input contract](docs/INPUT_FORMAT.md). This repository does not contain an example CSV. The `correlation` column must contain a sensor-only summary available before that session's EMA response; it must not contain current or future labels.

Validate a local input without making model requests:

```bash
python validate_input.py --input_csv /absolute/path/to/private/session_table.csv --target anxiety
```

Run one sequence:

```bash
python run_experiment.py \
  --input_csv /absolute/path/to/private/session_table.csv \
  --target anxiety --model deepseek-chat --refine_thr 0.5 \
  --output_dir /absolute/path/to/private/output/dycon/anxiety
```

Use `--target depression` for depression. `--model` also accepts `gpt-4o-mini` and `llama3.1-8b-instruct`. Provider aliases can change; use `DYCON_MODEL_ID` to select the intended available snapshot. The maximum refinement count defaults to 10; `--max_refines` can change it.

The output tree retains the original `participant/prediction`, `participant/context`, `participant/refine`, and `participant/update` structure. Session indices in JSON filenames are zero-based. The evaluator reports S1 as `session=1`. Outputs contain sensitive information and should remain in your private storage.

## Evaluation

```bash
python evaluation.py /absolute/path/to/private/output/dycon/anxiety \
  --output /absolute/path/to/private/metrics.csv
```

For each session and prediction stage, balanced accuracy is computed from all available ground-truth/prediction pairs as the mean of negative- and positive-class recall, using the subscale threshold `score >= 3`. MAE uses the continuous scores. This corrects the archived evaluator's averaging of one-observation scores, which yields ordinary accuracy rather than balanced accuracy. The released evaluator does not supply or change the paper's reported values.

R0 and feedback-informed R1–R10/R* are reported separately. After early stopping, subsequent refinement-stage summaries carry forward the final prediction. If a refinement log is absent, the evaluator includes the initial prediction only; it does not assume refinement completed. Binary balanced accuracy is undefined when a group contains only one ground-truth class, and is reported as NaN. Counts are included for every group. Fair method comparisons additionally require matching the evaluated records; separate metric files do not establish matched samples.

## Baselines

The `baselines` package provides Lag-1, Historical Mean, Label-only Memory, and Per-user Recent 3-shot. Run each module with `--help` for required private input/output paths. Label-only Memory uses prior EMA information without sensors. Per-user Recent 3-shot uses current sensors and up to three preceding sensor–EMA examples, including one or two examples when fewer than three are available. S1 has no history; see [implementation notes](docs/IMPLEMENTATION.md).

## Offline verification

```bash
python -m unittest discover -s tests -v
python scripts/check_release.py
```

Tests construct artificial values at runtime in temporary directories. They neither load research data nor call an LLM. They check metric aggregation, feedback timing, and context update gating; they do not establish empirical model performance.

## Intended use

DyCon is research software for monitoring self-reported states, not a diagnostic instrument. Its textual contexts are model-generated interpretations, not clinically validated explanations. The original prompt wording is retained for traceability to the implementation, including terms such as "validated"; those terms do not establish clinical validity.

## Citation

```bibtex
@inproceedings{jin2026dycon,
  title={Dynamic Context Modeling for Longitudinal Mental Health Monitoring under Distribution Shift},
  author={Jin, Seungwan and Noh, Taehyung and Kim, Junghyun and Lee, Uichin and Han, Kyungsik},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/forum?id=YFPmKTRY1L}
}
```
