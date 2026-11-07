"""Check one private, preprocessed input without loading any model client."""
import argparse
from config.config import excluded_cols
from src.input_data import load_session_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--target", required=True, choices=["anxiety", "depression"])
    args = parser.parse_args()
    frame = load_session_table(args.input_csv, args.target, excluded_cols)
    print(f"Validated {len(frame)} chronologically ordered session rows; no model request made.")


if __name__ == "__main__":
    main()
