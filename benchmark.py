"""Main benchmark orchestrator for ZenMux HLE evaluations."""

import argparse
import asyncio
import os
import sys
import logging
from datetime import datetime
from typing import Optional

from config import get_config
from hle import HLERunner
from utils.logging import get_runner_logger


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

    # Fix mode
    parser.add_argument(
        "--fix",
        type=str,
        help="Fix evaluation and judge failures from a specific timestamp directory (e.g., 'results/20250919_011623')"
    )

    # Resume mode
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume evaluation from a specific timestamp directory (e.g., 'results/20250922_122904')"
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

    # Model exclusion
    parser.add_argument(
        "--exclude-model",
        type=str,
        nargs="*",
        help="Exclude specific models. Supports: 'vendor/model' (all providers), 'vendor/model:provider' (specific), 'vendor' (all models from vendor)."
    )

    parser.add_argument(
        "--exclude-provider",
        type=str,
        nargs="*",
        help="Exclude all models from specific providers (e.g., 'theta', 'openai', 'anthropic')."
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
        default=None,
        help="Number of concurrent workers for evaluation (default: from config file)"
    )

    parser.add_argument(
        "--max-concurrent-models",
        type=int,
        default=None,
        help="Maximum number of models to evaluate concurrently (default: from config file)"
    )

    parser.add_argument(
        "--print-streaming",
        action="store_true",
        help="Print streaming responses to console in real-time (bypasses logging)"
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

    # Check for fix mode
    fix_mode = bool(args.fix)
    if fix_mode:
        if not os.path.exists(args.fix):
            print(f"❌ Error: Timestamp directory not found: {args.fix}")
            sys.exit(1)

    # Check for resume mode
    resume_mode = bool(args.resume)
    if resume_mode:
        if not os.path.exists(args.resume):
            print(f"❌ Error: Resume timestamp directory not found: {args.resume}")
            sys.exit(1)

    # Ensure fix and resume are mutually exclusive
    if fix_mode and resume_mode:
        print("❌ Error: --fix and --resume cannot be used together")
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

    if args.max_concurrent_models:
        config.hle.max_concurrent_models = args.max_concurrent_models

    if args.print_streaming:
        config.hle.print_streaming_output = True


    # Setup base output directory
    config.output_dir = args.output_dir

    # Ensure base output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    # Handle fix and resume modes differently than regular evaluation
    if fix_mode:
        # For fix mode, use the existing timestamp from the directory path
        timestamp_dir = args.fix
        batch_timestamp = os.path.basename(timestamp_dir)

        # Create runner with existing batch timestamp
        runner = HLERunner(config, batch_timestamp=batch_timestamp)
    elif resume_mode:
        # For resume mode, use the existing timestamp from the directory path
        timestamp_dir = args.resume
        batch_timestamp = os.path.basename(timestamp_dir)

        # Create runner with existing batch timestamp (will reuse existing directories)
        runner = HLERunner(config, batch_timestamp=batch_timestamp)
    else:
        # Generate batch timestamp for this evaluation run
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create runner with batch timestamp (this will create the timestamped directories and initialize logging)
        runner = HLERunner(config, batch_timestamp=batch_timestamp)

    # Now that logging is initialized, get logger
    logger = get_runner_logger()

    auto_judge = not args.no_judge

    logger.info("🌟 ZenMux HLE Benchmark")

    if fix_mode:
        logger.info(f"🔧 Mode: fix")
        logger.info(f"📁 Target directory: {timestamp_dir}")
        if args.mode != "all":
            logger.info(f"🔧 Evaluation mode: {args.mode}")
        logger.info(f"📝 Text only: {args.text_only}")
        logger.info(f"📊 Max samples: {args.max_samples}")
        logger.info(f"🏛️ Auto judge: {auto_judge}")
        if args.exclude_model:
            logger.info(f"🚫 Excluded models: {', '.join(args.exclude_model)}")
        if args.exclude_provider:
            logger.info(f"🚫 Excluded providers: {', '.join(args.exclude_provider)}")
        if args.model_filter:
            logger.info(f"🔍 Model filter: {args.model_filter}")
        if args.mode == "single" and args.model_slug and args.provider_slug:
            logger.info(f"🎯 Target model: {args.model_slug}:{args.provider_slug}")
    elif resume_mode:
        logger.info(f"▶️ Mode: resume")
        logger.info(f"📁 Resume directory: {timestamp_dir}")
        logger.info(f"🔧 Evaluation mode: {args.mode}")
        logger.info(f"📝 Text only: {args.text_only}")
        logger.info(f"📊 Max samples: {args.max_samples}")
        logger.info(f"🏛️ Auto judge: {auto_judge}")
        if args.exclude_model:
            logger.info(f"🚫 Excluded models: {', '.join(args.exclude_model)}")
        if args.exclude_provider:
            logger.info(f"🚫 Excluded providers: {', '.join(args.exclude_provider)}")
    else:
        logger.info(f"🔧 Mode: {args.mode}")
        logger.info(f"📝 Text only: {args.text_only}")
        logger.info(f"📊 Max samples: {args.max_samples}")
        logger.info(f"🏛️ Auto judge: {auto_judge}")
        if args.exclude_model:
            logger.info(f"🚫 Excluded models: {', '.join(args.exclude_model)}")
        if args.exclude_provider:
            logger.info(f"🚫 Excluded providers: {', '.join(args.exclude_provider)}")

    logger.info(f"👥 Workers per model: {config.hle.num_workers}")
    logger.info(f"🔄 Max concurrent models: {config.hle.max_concurrent_models}")
    logger.info(f"🎯 Judge model: {config.hle.judge_model}")
    logger.info(f"🌡️ Temperature: {config.hle.temperature}")
    logger.info(f"⏰ Timeout: {config.zenmux.timeout}s")
    logger.info(f"🔄 Max retries: {config.zenmux.max_retries}")
    logger.info(f"🎫 Max completion tokens: {config.hle.max_completion_tokens}")
    logger.info(f"📁 Base output directory: {config.output_dir}")
    logger.info(f"📁 Run directory: {config.run_dir}")
    logger.info(f"🕒 Batch timestamp: {batch_timestamp}")

    # Save question IDs for new evaluation runs (skip for fix and resume modes)
    if not fix_mode and not resume_mode:
        runner.save_question_ids(text_only=args.text_only, max_samples=args.max_samples)

    # Run evaluation or fix based on mode
    try:
        if fix_mode:
            logger.info(f"🔧 Fixing evaluation and judge failures in: {timestamp_dir}")

            # Build filtering parameters for fix mode
            fix_params = {
                "timestamp_dir": timestamp_dir,
                "mode": args.mode,
                "model_slug": args.model_slug,
                "provider_slug": args.provider_slug,
                "model_filter": args.model_filter,
                "exclude_models": args.exclude_model,
                "exclude_providers": args.exclude_provider,
                "text_only": args.text_only,
                "max_samples": args.max_samples,
                "auto_judge": auto_judge
            }

            fix_result = await runner.fix_models(**fix_params)

            if "error" in fix_result:
                logger.error(f"❌ Fix failed: {fix_result['error']}")
                sys.exit(1)

            logger.info(f"\n🎉 Fix completed!")
            logger.info(f"✅ Models with fixes applied: {fix_result['fixed_count']}")
            logger.info(f"✅ Models with no failures: {fix_result['no_failures_count']}")
            logger.info(f"❌ Models still with evaluation failures: {fix_result['still_eval_failures_count']}")
            logger.info(f"❌ Models still with judge failures: {fix_result['still_judge_failures_count']}")
            logger.info(f"❌ Models with processing errors: {fix_result['processing_errors_count']}")
            logger.info(f"📁 Results saved to: {fix_result['metrics_summary_file']}")

        else:
            # Regular evaluation modes (including resume mode)
            if resume_mode:
                logger.info(f"▶️ Resuming evaluation in: {timestamp_dir}")

            if args.mode == "single":
                logger.info(f"🎯 Evaluating single model: {args.model_slug}:{args.provider_slug}")
                result = await runner.run_specific_model_evaluation(
                    model_slug=args.model_slug,
                    provider_slug=args.provider_slug,
                    text_only=args.text_only,
                    max_samples=args.max_samples,
                    auto_judge=auto_judge
                )
                results = [result]

            elif args.mode == "filter":
                logger.info(f"🔍 Evaluating filtered models: {args.model_filter}")
                results = await runner.run_zenmux_models_evaluation(
                    text_only=args.text_only,
                    max_samples=args.max_samples,
                    auto_judge=auto_judge,
                    model_filter=args.model_filter,
                    exclude_models=args.exclude_model,
                    exclude_providers=args.exclude_provider
                )

            else:  # args.mode == "all"
                logger.info("🌍 Evaluating all available models")
                results = await runner.run_zenmux_models_evaluation(
                    text_only=args.text_only,
                    max_samples=args.max_samples,
                    auto_judge=auto_judge,
                    exclude_models=args.exclude_model,
                    exclude_providers=args.exclude_provider
                )

            # Save metrics summary for regular evaluation modes
            run_metadata = {
                "mode": args.mode,
                "text_only": args.text_only,
                "max_samples": args.max_samples,
                "auto_judge": auto_judge,
                "num_workers": config.hle.num_workers,
                "max_concurrent_models": config.hle.max_concurrent_models,
                "model_filter": getattr(args, 'model_filter', None),
                "model_slug": getattr(args, 'model_slug', None),
                "provider_slug": getattr(args, 'provider_slug', None)
            }

            # Always save metrics summary, regardless of success/failure
            runner.save_metrics_summary(results, run_metadata)

            # Log summary
            runner.log_summary(results)

            logger.info("\n🎉 Benchmark completed successfully!")

    except KeyboardInterrupt:
        # For keyboard interrupt, we might not have a logger yet, so use print
        if 'logger' in locals():
            logger.warning("\n⚠️ Benchmark interrupted by user")
        else:
            print("\n⚠️ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        # For exceptions, try to use logger if available, otherwise use print
        if 'logger' in locals():
            logger.error(f"\n❌ Benchmark failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            print(f"\n❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())