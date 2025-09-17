#!/usr/bin/env python3
"""Example usage of ZenMux Benchmark."""

import asyncio
import os
import sys
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from hle import HLERunner


async def test_basic_functionality():
    """Test basic functionality with a small sample."""
    print("🧪 Testing ZenMux Benchmark basic functionality...")

    # Check environment variables
    if not os.getenv("ZENMUX_API_KEY"):
        print("❌ ZENMUX_API_KEY not set. Please set it to test.")
        print("💡 Example: export ZENMUX_API_KEY='your_key_here'")
        return False

    try:
        # Get config
        config = get_config()
        print(f"✅ Configuration loaded")

        # Create runner
        runner = HLERunner(config)
        print(f"✅ HLE Runner created")

        # Test ZenMux API connection
        models = runner.zenmux_api.get_available_models()
        print(f"✅ ZenMux API connected - found {len(models)} models")

        # Test dataset loading
        dataset = runner.evaluator.dataset
        total_questions = dataset.get_total_count()
        text_only_questions = dataset.get_text_only_count()
        print(f"✅ HLE Dataset loaded - {total_questions} total, {text_only_questions} text-only")

        print("\n🎯 Available models (first 5):")
        for i, model in enumerate(models[:5]):
            endpoints_count = len(model.endpoints)
            print(f"  {i+1}. {model.slug} ({endpoints_count} endpoints)")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_small_test():
    """Run a very small test evaluation."""
    print("\n🚀 Running small test evaluation...")

    # Generate timestamp for unique directory naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = f"results/example_test_{timestamp}"

    # Set environment for testing
    os.makedirs(test_dir, exist_ok=True)

    config = get_config()
    config.output_dir = test_dir
    config.predictions_dir = f"{test_dir}/predictions"
    config.judged_dir = f"{test_dir}/judged"

    # Create directories
    for dir_path in [config.predictions_dir, config.judged_dir]:
        os.makedirs(dir_path, exist_ok=True)

    print(f"📁 Test results will be saved to: {test_dir}")

    runner = HLERunner(config)

    # Run evaluation with a very small sample
    try:
        results = await runner.run_zenmux_models_evaluation(
            text_only=True,  # Only text questions
            max_samples=2,   # Very small sample
            auto_judge=True,
            model_filter="gpt-4o-mini"  # Test with a common model
        )

        print(f"\n✅ Test evaluation completed!")
        print(f"📊 Results: {len(results)} model endpoints tested")
        print(f"📁 Results saved to: {test_dir}")
        print(f"   - Predictions: {config.predictions_dir}")
        print(f"   - Judged: {config.judged_dir}")

        return True

    except Exception as e:
        print(f"❌ Test evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🌟 ZenMux Benchmark - Example & Test Script")
    print("=" * 50)

    # Test basic functionality
    basic_test_passed = await test_basic_functionality()

    if not basic_test_passed:
        print("\n❌ Basic functionality test failed. Cannot proceed.")
        return

    print("\n" + "=" * 50)
    print("💡 Basic functionality test passed!")

    # Ask user if they want to run a small evaluation test
    try:
        response = input("\n🤔 Run a small evaluation test? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            await run_small_test()
        else:
            print("👍 Skipping evaluation test.")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

    print("\n" + "=" * 50)
    print("📚 Usage Examples:")
    print()
    print("# Test with 5 samples, text-only:")
    print("uv run python benchmark.py --text-only --max-samples 5")
    print()
    print("# Evaluate specific model:")
    print("uv run python benchmark.py --mode single --model-slug openai/gpt-4o-mini --provider-slug openai --max-samples 5")
    print()
    print("# Evaluate all GPT models:")
    print("uv run python benchmark.py --mode filter --model-filter gpt --text-only --max-samples 5")
    print()
    print("# Full evaluation (will take a long time!):")
    print("uv run python benchmark.py --mode all")


if __name__ == "__main__":
    asyncio.run(main())