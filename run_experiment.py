import os
import sys
import argparse
import importlib.util
from datetime import datetime, timezone, timedelta
import logging
import pandas as pd
import ast

from src.context import generate_context, update_context, refine_context
from src.prediction import predict_score
from src.utils import save_json, to_float_safe
from src.input_data import load_session_table, window_days
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def process_session_with_update(
    user: str,
    session: int,
    features: dict,
    correlation: str,
    gt_value: float,
    days: int,
    target: str,
    momentary_context: str,
    longitudinal_context: str,
    refine_skip_thr: float,
    refine_enabled: bool,
    skip_thr_enabled: bool,
    max_refines: int,
    user_dir: str,
    config
) -> None:
    """Process a session with context update enabled."""

    analysis, prediction = predict_score(
        momentary_context, features, session, days, target,
        longitudinal_context=longitudinal_context, config=config
    )
    save_json(
        {
            "user": user,
            "session": session,
            "gt_value": gt_value,
            "prediction": prediction,
            "analysis": analysis,
            "longitudinal_context": longitudinal_context,
            "momentary_context": momentary_context
        },
        os.path.join(user_dir, "prediction", f"{user}_predict_session_{session}.json")
    )

    # Refine loop
    refine_logs = []
    before_prediction = prediction
    attempt = 0
    while refine_enabled and attempt < max_refines:
        if skip_thr_enabled and abs(to_float_safe(prediction) - to_float_safe(gt_value)) <= refine_skip_thr:
            break

        refine_reason, momentary_context = refine_context(
            momentary_context, prediction, gt_value, analysis, target,
            correlation, config=config
        )

        analysis, prediction = predict_score(
            momentary_context, features, session, days, target,
            longitudinal_context=longitudinal_context, config=config
        )

        refine_logs.append({
            "attempt": attempt,
            "before_pred": before_prediction,
            "gt_value": gt_value,
            "prediction": prediction,
            "analysis": analysis,
            "refine_reason": refine_reason,
            "refine_context": momentary_context
        })

        before_prediction = prediction
        attempt += 1

    save_json(
        refine_logs,
        os.path.join(user_dir, "refine", f"{user}_refine_session_{session}.json")
    )

    # Update context if needed
    if abs(to_float_safe(prediction) - to_float_safe(gt_value)) <= refine_skip_thr:
        update_rationale, longitudinal_context = update_context(
            target=target,
            prediction=prediction,
            gt_value=gt_value,
            analysis=analysis,
            momentary_context=momentary_context,
            longitudinal_context=longitudinal_context,
            threshold=refine_skip_thr,
            config=config
        )
        save_json(
            {
                "user": user,
                "update": True,
                "session": session,
                "update_rationale": update_rationale,
                "longitudinal_context": longitudinal_context
            },
            os.path.join(user_dir, "update", f"{user}_context_session_{session}.json")
        )
    else:
        save_json(
            {
                "user": user,
                "update": False,
                "session": session,
                "longitudinal_context": longitudinal_context
            },
            os.path.join(user_dir, "update", f"{user}_context_session_{session}.json")
        )

    return longitudinal_context

def process_session_without_update(
    user: str,
    session: int,
    features: dict,
    correlation: str,
    gt_value: float,
    days: int,
    target: str,
    momentary_context: str,
    refine_skip_thr: float,
    refine_enabled: bool,
    skip_thr_enabled: bool,
    max_refines: int,
    user_dir: str,
    config
) -> None:
    """Process a session without context update."""
    # Initial prediction
    analysis, prediction = predict_score(
        momentary_context, features, session=0, days=days, target=target, config=config
    )
    save_json(
        {
            "user": user,
            "session": session,
            "gt_value": gt_value,
            "prediction": prediction,
            "analysis": analysis,
            "momentary_context": momentary_context
        },
        os.path.join(user_dir, "prediction", f"{user}_predict_session_{session}.json")
    )

    # Refine loop
    refine_logs = []
    before_prediction = prediction
    attempt = 0

    while refine_enabled and attempt < max_refines:
        if skip_thr_enabled and abs(to_float_safe(prediction) - to_float_safe(gt_value)) <= refine_skip_thr:
            break
        print(f"Refining session {session} with attempt {attempt}")
        refine_reason, momentary_context = refine_context(
            momentary_context, prediction, gt_value, analysis, target,
            correlation, config=config
        )

        analysis, prediction = predict_score(
            momentary_context, features, session=0, days=days, target=target, config=config
        )

        refine_logs.append({
            "attempt": attempt,
            "before_pred": before_prediction,
            "gt_value": gt_value,
            "prediction": prediction,
            "analysis": analysis,
            "refine_reason": refine_reason,
            "refine_context": momentary_context
        })

        before_prediction = prediction
        attempt += 1

    save_json(
        refine_logs,
        os.path.join(user_dir, "refine", f"{user}_refine_session_{session}.json")
    )

def process_user_end_to_end(
    user: str,
    input_csv: str,
    target: str,
    excluded_cols: list,
    max_refines: int,
    output_dir: str,
    refine_enabled: bool,
    update_context_enabled: bool,
    skip_thr_enabled: bool,
    refine_skip_thr: float,
    config
) -> None:
    """Process a single user's data end-to-end."""
    # Validate local input before generating outputs or making any model request.
    df = load_session_table(input_csv, target, excluded_cols)
    # Create output directories without overwriting a previous run.
    user_dir = os.path.join(output_dir, user)
    if os.path.isdir(user_dir) and any(os.scandir(user_dir)):
        raise ValueError("Output sequence already exists; choose a new --output_dir")
    for subdir in ["context", "prediction", "refine", "update"]:
        os.makedirs(os.path.join(user_dir, subdir), exist_ok=True)

    # Load and process data
    gt_value = df.iloc[0][target]
    num_sessions = df.shape[0]

    # Process initial session
    initial_features = df.iloc[0].drop(labels=excluded_cols, errors="ignore").to_dict()
    days = window_days(initial_features)

    # Generate initial context
    momentary_context = generate_context(df.iloc[0]["correlation"], target=target, config=config)
    save_json(
        {"user": user, "session": 0, "momentary_context": momentary_context},
        os.path.join(user_dir, "context", f"{user}_context_session_0.json")
    )

    # Initial prediction
    analysis, prediction = predict_score(
        momentary_context, initial_features, session=0, days=days, target=target, config=config
    )
    save_json(
        {
            "user": user,
            "session": 0,
            "analysis": analysis,
            "gt_value": gt_value,
            "prediction": prediction
        },
        os.path.join(user_dir, "prediction", f"{user}_predict_session_0.json")
    )

    refine_logs = []
    before_prediction = prediction
    attempt = 0
    while refine_enabled and attempt < max_refines:
        if skip_thr_enabled and abs(to_float_safe(prediction) - to_float_safe(gt_value)) <= refine_skip_thr:
            break

        print(f"Refining session 0 with attempt {attempt}")

        refine_reason, momentary_context = refine_context(
            momentary_context, prediction, gt_value, analysis, target,
            df.iloc[0]["correlation"], config=config
        )

        analysis, prediction = predict_score(
            momentary_context, initial_features, session=0, days=days, target=target, config=config
        )

        refine_logs.append({
            "attempt": attempt,
            "before_pred": before_prediction,
            "gt_value": gt_value,
            "prediction": prediction,
            "analysis": analysis,
            "refine_reason": refine_reason,
            "refine_context": momentary_context
        })

        before_prediction = prediction
        attempt += 1

    save_json(
        refine_logs,
        os.path.join(user_dir, "refine", f"{user}_refine_session_0.json")
    )

    if abs(to_float_safe(prediction) - to_float_safe(gt_value)) <= refine_skip_thr:
        longitudinal_context = momentary_context
    else:
        longitudinal_context = ""

    for session in range(1, num_sessions):
        features = df.iloc[session].drop(labels=excluded_cols, errors="ignore").to_dict()
        gt_value = df.iloc[session][target]
        days = window_days(features)

        # Generate momentary context
        momentary_context = generate_context(df.iloc[session]["correlation"], target=target, config=config)
        save_json(
            {
                "user": user,
                "session": session,
                "momentary_context": momentary_context,
                "longitudinal_context": longitudinal_context
            },
            os.path.join(user_dir, "context", f"{user}_context_session_{session}.json")
        )

        if update_context_enabled:
            longitudinal_context = process_session_with_update(
                user, session, features, df.iloc[session]["correlation"], gt_value, days, target,
                momentary_context, longitudinal_context, refine_skip_thr,
                refine_enabled, skip_thr_enabled, max_refines,
                user_dir, config
            )
        else:
            process_session_without_update(
                user, session, features, df.iloc[session]["correlation"], gt_value, days, target,
                momentary_context, refine_skip_thr,
                refine_enabled, skip_thr_enabled, max_refines,
                user_dir, config
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["anxiety", "depression"])
    parser.add_argument("--refine_thr", required=True, type=float, help="Refinement threshold")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
    parser.add_argument("--no_refine", action="store_true", help="Skip context refinement step")
    parser.add_argument("--no_context_update", action="store_true", help="Predict without longitudinal context")
    parser.add_argument("--no_skip_thr", action="store_true", help="Disable refine-skip threshold logic")
    parser.add_argument("--prompt_file", default=os.path.join(os.path.dirname(__file__), "prompts", "prompt.py"), help="Path to file containing prompts")
    parser.add_argument("--output_dir", help="Local output directory; existing sequence outputs are not overwritten")
    parser.add_argument("--max_refines", type=int, default=10)
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="Model to use for predictions (default: deepseek-chat)",
        choices=["deepseek-chat", "gpt-4o-mini", "llama3.1-8b-instruct"]
    )

    args = parser.parse_args()
    if not 0 <= args.refine_thr <= 6 or args.max_refines < 0:
        parser.error("refine_thr must be in [0, 6] and max_refines must be nonnegative")
    config.max_refines = args.max_refines

    # Load configuration
    spec = importlib.util.spec_from_file_location("prompts", args.prompt_file)
    prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompts)

    # Update config with prompts and model
    config.model = args.model
    config.context_gen_prompt = prompts.context_gen_prompt
    config.context_update_prompt = prompts.context_update_prompt
    config.predict_prompt_session_1 = prompts.predict_prompt_session_1
    config.predict_prompt_after_session_1 = prompts.predict_prompt_after_session_1
    config.context_refine_prompt = prompts.context_refine_prompt

    # Set up output directory (allow a fixed date via env to avoid UTC-rollover splitting outputs)
    today = os.environ.get("DYCON_DATE") or datetime.now(timezone.utc).strftime("%Y%m%d")

    # Create base directory structure
    base_dir = f"results/{today}/{config.model}/{args.target}"

    # Build experiment parameters
    params = []
    params.append(f"refine{config.max_refines}")

    if args.no_skip_thr:
        params.append("noskip")
    else:
        params.append(f"skip{args.refine_thr}")

    if args.no_refine:
        params.append("norefine")
    if args.no_context_update:
        params.append("noupdate")

    # Combine all parameters with double underscore
    output_dir = args.output_dir or f"{base_dir}/{'__'.join(params)}"

    # Process user
    user = os.path.splitext(os.path.basename(args.input_csv))[0]
    logging.info(f"Processing user: {user}")

    process_user_end_to_end(
        user=user,
        input_csv=args.input_csv,
        target=args.target,
        excluded_cols=config.excluded_cols,
        max_refines=config.max_refines,
        output_dir=output_dir,
        refine_enabled=not args.no_refine,
        update_context_enabled=not args.no_context_update,
        skip_thr_enabled=not args.no_skip_thr,
        refine_skip_thr=float(args.refine_thr),
        config=config
    )

    logging.info("Processing completed!")

if __name__ == "__main__":
    main()
