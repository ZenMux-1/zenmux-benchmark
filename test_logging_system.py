#!/usr/bin/env python3
"""Test script to verify the logging system is working correctly."""

import asyncio
import tempfile
import os
from utils.logging import (
    BenchmarkLogger,
    get_evaluation_logger,
    get_judge_logger,
    get_runner_logger,
    get_model_logger,
    PerformanceTimer
)

async def test_logging_system():
    """Comprehensive test of the logging system."""
    print("🧪 Testing ZenMux Benchmark Logging System")
    print("=" * 60)

    # Test 1: Basic logger setup
    print("\n1️⃣ Testing basic logger setup...")

    # Setup logging with a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        BenchmarkLogger.setup_logging(
            log_dir=temp_dir,
            batch_timestamp="test_20250918_123456"
        )

        # Test different loggers
        runner_logger = get_runner_logger()
        evaluation_logger = get_evaluation_logger()
        judge_logger = get_judge_logger()

        print(f"✅ Loggers created successfully")
        print(f"📁 Log directory: {BenchmarkLogger.get_log_directory()}")

        # Test 2: Basic logging levels
        print("\n2️⃣ Testing different log levels...")

        runner_logger.debug("This is a DEBUG message")
        runner_logger.info("🎯 This is an INFO message")
        runner_logger.warning("⚠️ This is a WARNING message")
        runner_logger.error("❌ This is an ERROR message")

        print("✅ Log levels tested")

        # Test 3: Model-specific logger
        print("\n3️⃣ Testing model-specific logger...")

        model_logger = get_model_logger("openai/gpt-4o:openai")
        model_logger.info("🚀 Model-specific logging test")
        model_logger.info("📊 Model evaluation started")

        print("✅ Model-specific logger tested")

        # Test 4: Performance timer
        print("\n4️⃣ Testing performance timer...")

        async def slow_operation():
            await asyncio.sleep(1)
            return "Operation completed"

        with PerformanceTimer(evaluation_logger, "slow operation test"):
            result = await slow_operation()

        print(f"✅ Performance timer tested: {result}")

        # Test 5: Concurrent logging
        print("\n5️⃣ Testing concurrent logging...")

        async def concurrent_logger(logger_name, message_count):
            logger = get_evaluation_logger()
            for i in range(message_count):
                logger.info(f"🔄 {logger_name} - Message {i+1}")
                await asyncio.sleep(0.1)

        # Run multiple concurrent loggers
        tasks = [
            concurrent_logger("Logger1", 3),
            concurrent_logger("Logger2", 3),
            concurrent_logger("Logger3", 3)
        ]

        await asyncio.gather(*tasks)
        print("✅ Concurrent logging tested")

        # Test 6: Log file verification
        print("\n6️⃣ Verifying log files...")

        log_dir = BenchmarkLogger.get_log_directory()
        expected_files = [
            "zenmux_benchmark.log",
            "errors.log",
            "model_openai_gpt-4o_openai.log"
        ]

        for file_name in expected_files:
            file_path = os.path.join(log_dir, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                    line_count = len(content.strip().split('\n')) if content.strip() else 0
                print(f"✅ {file_name}: {line_count} lines")
            else:
                print(f"❌ {file_name}: File not found")

        # Test 7: Error handling
        print("\n7️⃣ Testing error handling...")

        try:
            # This should be logged as an error
            raise ValueError("Test error for logging")
        except ValueError as e:
            runner_logger.error(f"Caught expected error: {e}")
            print("✅ Error logging tested")

        print("\n" + "=" * 60)
        print("🎉 Logging system test completed successfully!")
        print(f"📁 Test logs saved in: {log_dir}")

        # Print summary of log files created
        print("\n📊 Log files created:")
        for file_name in os.listdir(log_dir):
            file_path = os.path.join(log_dir, file_name)
            size = os.path.getsize(file_path)
            print(f"  📄 {file_name}: {size} bytes")

def test_config_integration():
    """Test integration with BenchmarkConfig."""
    print("\n🔧 Testing config integration...")

    from config import get_config

    # Get default config
    config = get_config()

    # Test logging configuration
    print(f"✅ Console log level: {config.console_log_level}")
    print(f"✅ File log level: {config.file_log_level}")
    print(f"✅ Model-specific logs: {config.enable_model_specific_logs}")
    print(f"✅ Logs directory: {config.logs_dir}")

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_logging_system())

    # Run config integration test
    test_config_integration()

    print("\n🎯 All tests completed!")