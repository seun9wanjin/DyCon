# Local input contract

Each file represents one participant-year sequence. Rows must already be ordered chronologically. Participant identities spanning multiple years must not be silently merged or treated as independent people for inferential statistics.

| Column | Meaning |
|---|---|
| `anxiety` / `depression` | Observed target subscale score in the range 0–6. The requested target is required. These columns are excluded from prediction features. |
| `correlation` | Precomputed textual summary of within-window sensor relationships. Despite its legacy name, this must not contain sensor-to-label correlations, current EMA responses, or future observations. |
| Sensor feature columns | Serialized Python-style lists of finite numeric daily values, with a consistent window length within each row. Missing-value handling and feature construction must be completed upstream. |
| `pid` | Optional sequence identifier; if present, it must be constant in the file. Excluded from model inputs. |
| `date` | Optional EMA/session date. If present, dates must be nonmissing, unique, and increasing. |
| `index`, `year`, `phq4` | Optional metadata/label columns; excluded from model inputs. |

No other metadata columns should be passed as sensor features. The runner validates feature lists but cannot determine whether a user-supplied summary was produced without label leakage. Users are responsible for constructing windows only from information available before prediction and for matching the preprocessing and inclusion rules of the intended evaluation.

The original GLOBEM-specific feature name used to determine window length has been replaced by validation of all feature lists. This preserves the length for valid original inputs while avoiding a dependency on one column name. The validator does not implement the original raw-sensor preprocessing pipeline and does not impute missing values.

R0 is computed before the target is used for refinement. The target score is stored locally alongside the prediction for offline evaluation. Subsequent refinement is feedback-informed and must not be interpreted as a prediction made without access to that feedback.
