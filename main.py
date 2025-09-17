"""ZenMux Benchmark - Main entry point."""

import sys
import os

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import main as benchmark_main
import asyncio


def main():
    """Main entry point that delegates to the benchmark module."""
    print("🌟 ZenMux Benchmark - AI Model Evaluation Framework")
    print("📚 Evaluating models on Humanity's Last Exam (HLE)")
    print()

    # Run the async benchmark
    asyncio.run(benchmark_main())


if __name__ == "__main__":
    main()
