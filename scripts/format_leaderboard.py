#!/usr/bin/env python3
"""Format evaluation results in leaderboard style"""

import json
import sys
from pathlib import Path

# Expected order of environments (matches leaderboard)
ENVIRONMENT_ORDER = [
    "affine:sat",
    "affine:abd",
    "affine:ded",
    "agentgym:webshop",
    "agentgym:alfworld",
    "agentgym:babyai",
    "agentgym:sciworld",
    "agentgym:textcraft",
]


def format_leaderboard_score(results_path: Path):
    """Format evaluation results in leaderboard style"""
    
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        print(f"Please run evaluation first: python scripts/4_evaluate.py")
        return None
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Get per-environment metrics
    per_env = results.get("per_environment", {})
    
    # Format scores for each environment in order
    formatted_scores = []
    
    for env_name in ENVIRONMENT_ORDER:
        if env_name not in per_env:
            # Missing environment - use zeros
            formatted_scores.append("0.00/[0.00,0.00]/0")
            continue
        
        env_metrics = per_env[env_name]
        
        # Extract metrics
        mean_score = env_metrics.get("mean_score", 0.0)
        ci = env_metrics.get("confidence_interval", {})
        ci_lower = ci.get("lower", 0.0)
        ci_upper = ci.get("upper", 0.0)
        num_samples = env_metrics.get("num_samples", 0)
        
        # Format: score/[lower_ci,upper_ci]/num_samples
        # Multiply by 100 to get percentage
        score_str = f"{mean_score * 100:.2f}/[{ci_lower * 100:.2f},{ci_upper * 100:.2f}]/{num_samples}"
        formatted_scores.append(score_str)
    
    return formatted_scores


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Format evaluation results in leaderboard style"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="logs/evaluation_results.json",
        help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="my-model",
        help="Model name/identifier"
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    scores = format_leaderboard_score(results_path)
    
    if scores is None:
        sys.exit(1)
    
    # Print formatted output
    print("\n" + "=" * 80)
    print("LEADERBOARD FORMAT RESULTS")
    print("=" * 80)
    print(f"\nModel: {args.model_name}")
    print("\nEnvironment Scores (in leaderboard order):")
    print("-" * 80)
    
    for env_name, score_str in zip(ENVIRONMENT_ORDER, scores):
        env_display = env_name.replace("affine:", "").replace("agentgym:", "")
        print(f"  {env_display:20s}: {score_str}")
    
    print("\n" + "-" * 80)
    print("Leaderboard Format (copy this line):")
    print("-" * 80)
    
    # Single line format (matches leaderboard)
    leaderboard_line = "  ".join(scores)
    print(f"\n{leaderboard_line}\n")
    
    # Also compute aggregate metrics
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    weighted_mean = results.get("weighted_mean_score", 0.0) * 100
    weighted_acc = results.get("weighted_accuracy", 0.0) * 100
    overall_success = results.get("overall_success_rate", 0.0) * 100
    
    print("Aggregate Metrics:")
    print(f"  Weighted Mean Score: {weighted_mean:.2f}%")
    print(f"  Weighted Accuracy: {weighted_acc:.2f}%")
    print(f"  Overall Success Rate: {overall_success:.2f}%")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
