"""Main benchmark orchestrator for ZenMux HLE evaluations."""

import argparse
import asyncio
import os
import sys
from typing import Optional

from config import get_config
from hle import HLERunner


async def main():
    """Main entry point for benchmark."""
    parser = argparse.ArgumentParser(
        description="ZenMux HLE Benchmark - Evaluate AI models on Humanity's Last Exam"
    )

    # Evaluation mode
    parser.add_argument(
        "--mode",
        choices=["all", "single", "filter"],
        default="all",
        help="Evaluation mode: 'all' for all models, 'single' for specific model, 'filter' for filtered models"
    )

    # Model specification (for single mode)
    parser.add_argument(
        "--model-slug",
        type=str,
        help="Model slug (e.g., 'openai/gpt-4.1-mini') for single model evaluation"
    )

    parser.add_argument(
        "--provider-slug",
        type=str,
        help="Provider slug (e.g., 'openai') for single model evaluation"
    )

    # Model filtering (for filter mode)
    parser.add_argument(
        "--model-filter",
        type=str,
        help="Filter models by substring (case-insensitive)"
    )

    # Dataset options
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only evaluate on text-only questions (filter out image questions)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to first N samples (useful for testing)"
    )

    # Evaluation options
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip automatic judging of results"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of concurrent workers for evaluation"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )

    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Directory for prediction files (default: {output-dir}/predictions)"
    )

    parser.add_argument(
        "--judged-dir",
        type=str,
        default=None,
        help="Directory for judged files (default: {output-dir}/judged)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "single":
        if not args.model_slug or not args.provider_slug:
            print("❌ Error: --model-slug and --provider-slug are required for single mode")
            sys.exit(1)

    if args.mode == "filter":
        if not args.model_filter:
            print("❌ Error: --model-filter is required for filter mode")
            sys.exit(1)

    # Check required environment variables
    if not os.getenv("ZENMUX_API_KEY"):
        print("❌ Error: ZENMUX_API_KEY environment variable is required")
        print("💡 Please set it with your ZenMux API key:")
        print("   export ZENMUX_API_KEY='your_api_key_here'")
        sys.exit(1)

    # Setup configuration
    config = get_config()

    # Override config with command line arguments
    if args.num_workers:
        config.hle.num_workers = args.num_workers

    # Setup output directories
    config.output_dir = args.output_dir
    config.predictions_dir = args.predictions_dir or os.path.join(args.output_dir, "predictions")
    config.judged_dir = args.judged_dir or os.path.join(args.output_dir, "judged")

    # Create directories
    for dir_path in [config.output_dir, config.predictions_dir, config.judged_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Create runner
    runner = HLERunner(config)

    auto_judge = not args.no_judge

    print("🌟 ZenMux HLE Benchmark")
    print(f"🔧 Mode: {args.mode}")
    print(f"📝 Text only: {args.text_only}")
    print(f"📊 Max samples: {args.max_samples}")
    print(f"🏛️ Auto judge: {auto_judge}")
    print(f"👥 Workers: {config.hle.num_workers}")
    print(f"📁 Output directory: {config.output_dir}")

    # Run evaluation based on mode
    try:
        if args.mode == "single":
            print(f"🎯 Evaluating single model: {args.model_slug}:{args.provider_slug}")
            result = await runner.run_specific_model_evaluation(
                model_slug=args.model_slug,
                provider_slug=args.provider_slug,
                text_only=args.text_only,
                max_samples=args.max_samples,
                auto_judge=auto_judge
            )
            results = [result]

        elif args.mode == "filter":
            print(f"🔍 Evaluating filtered models: {args.model_filter}")
            results = await runner.run_zenmux_models_evaluation(
                text_only=args.text_only,
                max_samples=args.max_samples,
                auto_judge=auto_judge,
                model_filter=args.model_filter
            )

        else:  # args.mode == "all"
            print("🌍 Evaluating all available models")
            results = await runner.run_zenmux_models_evaluation(
                text_only=args.text_only,
                max_samples=args.max_samples,
                auto_judge=auto_judge
            )

        # Print summary
        runner.print_summary(results)

        print("\n🎉 Benchmark completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())