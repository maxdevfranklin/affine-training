#!/usr/bin/env python3
"""Format evaluation results in leaderboard and validator-summary style."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.validator_summary import (  # noqa: E402
    ENVIRONMENT_ORDER,
    build_validator_summary,
    format_environment_scores,
)


def _load_results(path: Path) -> Dict[str, any]:
    if not path.exists():
        print(f"Error: Results file not found: {path}")
        print("Please run evaluation first: python scripts/4_evaluate.py")
        return {}
    with path.open("r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Format evaluation results in leaderboard and validator summary style"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="logs/evaluation_results.json",
        help="Path to evaluation results JSON file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="my-model",
        help="Model name/identifier",
    )
    parser.add_argument(
        "--uid",
        type=int,
        default=0,
        help="UID value to display in summary table",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="local",
        help="Revision identifier to display in summary table",
    )
    parser.add_argument(
        "--first-block",
        type=int,
        default=0,
        help="First block value to display in summary table",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        help="Optional path to write validator-style summary text",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    results = _load_results(results_path)
    if not results:
        sys.exit(1)

    per_env = results.get("per_environment", {})
    env_scores = format_environment_scores(per_env)

    print("\n" + "=" * 80)
    print("LEADERBOARD FORMAT RESULTS")
    print("=" * 80)
    print(f"\nModel: {args.model_name}")
    print("\nEnvironment Scores (in leaderboard order):")
    print("-" * 80)

    for env_name, score_str in zip(ENVIRONMENT_ORDER, env_scores):
        env_display = env_name.replace("affine:", "").replace("agentgym:", "")
        print(f"  {env_display:20s}: {score_str}")

    print("\n" + "-" * 80)
    print("Leaderboard Format (copy this line):")
    print("-" * 80)
    print(f"\n{'  '.join(env_scores)}\n")

    weighted_mean = results.get("weighted_mean_score", 0.0) * 100
    weighted_acc = results.get("weighted_accuracy", 0.0) * 100
    overall_success = results.get("overall_success_rate", 0.0) * 100

    print("Aggregate Metrics:")
    print(f"  Weighted Mean Score: {weighted_mean:.2f}%")
    print(f"  Weighted Accuracy:   {weighted_acc:.2f}%")
    print(f"  Overall Success:     {overall_success:.2f}%")

    summary = build_validator_summary(
        per_env=per_env,
        model_name=args.model_name,
        uid=args.uid,
        revision=args.revision,
        first_block=args.first_block,
    )

    summary_text = summary.to_text()

    print("\n" + "=" * 80)
    print("VALIDATOR-STYLE SUMMARY ROW")
    print("=" * 80)
    print(summary_text)
    print("\n" + "=" * 80)

    if args.summary_output:
        output_path = Path(args.summary_output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(summary_text + "\n")
        print(f"Validator summary written to: {output_path}")


if __name__ == "__main__":
    main()
