# Implementation and release notes

## Preserved behavior

- The GLOBEM anxiety/depression core and `prompts/prompt.py` come from the rebuttal implementation. The original prompt text is preserved.
- The first session generates a momentary context and records an initial prediction. If refinement reaches the error threshold, that momentary context initializes the longitudinal context directly. Later eligible sessions use the longitudinal context updater.
- Initial prediction precedes current-session EMA-guided refinement. A maximum of 10 refinement calls is the default; early stopping tests the current prediction error.
- The longitudinal update criterion remains active when `--no_skip_thr` disables early stopping.
- `--no_refine` skips momentary refinement. `--no_context_update` omits longitudinal context from prediction; it is not a context-freeze evaluation.

## Release-specific changes

- Secrets are read from environment variables. No original credential file is included.
- API errors fail without manufacturing fallback scores. Error messages do not dump request bodies.
- The prompt-file default, input validation, configurable private output directory, and overwrite protection were added for a usable standalone release.
- The evaluator pools observations before computing balanced accuracy. The archived one-observation averaging routine is not distributed under a BA label.
- Baseline input/output paths are supplied by the caller rather than hardcoded local paths. Original LLM baseline prompts are retained.

## Baseline interpretation

Lag-1 copies the immediately preceding EMA score. Historical Mean averages all preceding EMA scores. Neither has a prediction at S1. Label-only Memory predicts using a natural-language summary of past EMA responses and updates that summary after observing the current response. Its implementation can produce an S1 prediction with no history; this is not a history-informed result, and the paper omits it from Table 1. Per-user Recent 3-shot uses current sensors and the most recent available prior sensor–EMA pairs, up to three; S1 therefore uses current sensors without prior examples.

## Provenance limits

The release contains executable source, not original outputs or empirical fixtures. It does not include the paper's multi-run archives, raw preprocessing pipeline, all four-dataset adaptations, OPPU training code, reflection baselines, or bootstrap/Spearman analysis scripts. Published result tables must not be inferred from the offline unit tests or from a run using a different API snapshot.

No existing repository history was imported. Data, notebooks, figures, model weights, logs, credentials, and generated personal contexts were excluded from the public upload. Dependencies and externally obtained datasets/models retain their own licenses and access terms.
