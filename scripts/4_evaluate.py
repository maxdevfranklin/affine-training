#!/usr/bin/env python3
"""Script to evaluate trained model"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.evaluator import ModelEvaluator
from utils.config import load_config
from utils.logger import setup_logger


async def main():
    """Main evaluation workflow"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate Affine model")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model to evaluate (overrides auto-detection)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/evaluation_results.json",
        help="Path to save evaluation results"
    )
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger("evaluation", Path("logs"))
    logger.info("Starting model evaluation...")

    # Load config
    config = load_config(args.config)

    # Determine model path
    if args.model:
        # User specified a model path
        model_path = args.model
        if not Path(model_path).exists():
            logger.error(f"Specified model path does not exist: {model_path}")
            return
        logger.info(f"Using user-specified model: {model_path}")
    else:
        # Auto-detect: Try RL model first, fall back to SFT, then base
        model_paths = [
            Path(config["training"]["save_dir"]) / "rl_final",
            Path(config["training"]["save_dir"]) / "sft_final",
            config["model"]["base_model_path"]
        ]

        model_path = None
        for path in model_paths:
            if Path(path).exists():
                model_path = str(path)
                logger.info(f"Auto-detected model: {model_path}")
                break

        if model_path is None:
            logger.error("No trained model found! Use --model to specify a path.")
            return

    # Create evaluator
    evaluator = ModelEvaluator(config, model_path=model_path)

    try:
        # Setup
        logger.info("Setting up evaluator...")
        evaluator.setup()

        # Evaluate
        logger.info("Running evaluation...")
        results = await evaluator.evaluate_all()

        # Save results
        output_file = Path(args.output)
        evaluator.save_results(results, output_file)

        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"\nTo format results in leaderboard style, run:")
        logger.info(f"  python scripts/format_leaderboard.py --results {output_file}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        logger.exception(e)
        raise


if __name__ == "__main__":
    asyncio.run(main())
