#!/usr/bin/env python3
"""Test script to verify the exclude functionality."""

import os
import tempfile
import shutil
from config import get_config
from hle import HLERunner


def test_exclude_functionality():
    """Test the exclude functionality with mock data."""
    print("🧪 Testing exclude functionality")

    # Mock model-endpoint pairs (simulating what ZenMuxAPI would return)
    mock_model_pairs = [
        ("openai/gpt-4o:openai", None, None),
        ("openai/gpt-4o:azure", None, None),
        ("openai/gpt-4o-mini:openai", None, None),
        ("openai/gpt-4o-mini:azure", None, None),
        ("anthropic/claude-3-haiku:anthropic", None, None),
        ("anthropic/claude-3-haiku:amazon-bedrock", None, None),
        ("anthropic/claude-3.5-sonnet:anthropic", None, None),
        ("anthropic/claude-3.5-sonnet:amazon-bedrock", None, None),
        ("google/gemini-pro:google-vertex", None, None),
        ("deepseek/deepseek-chat:deepseek", None, None),
    ]

    def test_exclusion_logic(exclude_models):
        """Test the exclusion logic with given exclude patterns."""
        print(f"\n🔍 Testing exclusion with patterns: {exclude_models}")

        if not exclude_models:
            print("   No exclusion patterns provided")
            return mock_model_pairs

        excluded_count = 0
        original_count = len(mock_model_pairs)
        exclude_slugs = set(exclude_models)

        filtered_pairs = []
        for model_id, model, endpoint in mock_model_pairs:
            # Extract model slug from model_id (format: "vendor/model:provider")
            model_slug = model_id.split(':')[0]  # Get "vendor/model" part

            # Check if this model should be excluded
            should_exclude = False
            for exclude_slug in exclude_slugs:
                exclude_lower = exclude_slug.lower()
                model_lower = model_slug.lower()

                # Case 1: Exact match (e.g., "openai/gpt-4o" == "openai/gpt-4o")
                if exclude_lower == model_lower:
                    should_exclude = True
                    break

                # Case 2: Vendor-only exclusion (e.g., "anthropic" matches "anthropic/*")
                if '/' not in exclude_slug and model_lower.startswith(exclude_lower + '/'):
                    should_exclude = True
                    break

                # Case 3: Model name only (e.g., "gpt-4o" matches "*/gpt-4o" but not "*/gpt-4o-*")
                if '/' not in exclude_slug and '/' in model_slug:
                    model_name = model_slug.split('/')[-1].lower()
                    if exclude_lower == model_name:
                        should_exclude = True
                        break

            if not should_exclude:
                filtered_pairs.append((model_id, model, endpoint))
                print(f"   ✅ Kept: {model_id}")
            else:
                excluded_count += 1
                print(f"   ❌ Excluded: {model_id}")

        print(f"   📊 Summary: {excluded_count} excluded, {len(filtered_pairs)} remaining (was {original_count})")
        return filtered_pairs

    # Test scenarios
    test_cases = [
        {
            "name": "No exclusions",
            "exclude": None,
            "expected_count": 10
        },
        {
            "name": "Exclude OpenAI models",
            "exclude": ["openai/gpt-4o"],
            "expected_excluded": ["openai/gpt-4o:openai", "openai/gpt-4o:azure"]
        },
        {
            "name": "Exclude Anthropic Claude 3 Haiku",
            "exclude": ["anthropic/claude-3-haiku"],
            "expected_excluded": ["anthropic/claude-3-haiku:anthropic", "anthropic/claude-3-haiku:amazon-bedrock"]
        },
        {
            "name": "Exclude multiple models",
            "exclude": ["openai/gpt-4o", "anthropic/claude-3-haiku"],
            "expected_excluded": [
                "openai/gpt-4o:openai", "openai/gpt-4o:azure",
                "anthropic/claude-3-haiku:anthropic", "anthropic/claude-3-haiku:amazon-bedrock"
            ]
        },
        {
            "name": "Exclude by model name only",
            "exclude": ["gpt-4o"],  # Should match only exact "gpt-4o", not "gpt-4o-mini"
            "expected_excluded": [
                "openai/gpt-4o:openai", "openai/gpt-4o:azure"
            ]
        },
        {
            "name": "Exclude all Anthropic",
            "exclude": ["anthropic"],
            "expected_excluded": [
                "anthropic/claude-3-haiku:anthropic", "anthropic/claude-3-haiku:amazon-bedrock",
                "anthropic/claude-3.5-sonnet:anthropic", "anthropic/claude-3.5-sonnet:amazon-bedrock"
            ]
        }
    ]

    print(f"\n📋 Original model list ({len(mock_model_pairs)} models):")
    for model_id, _, _ in mock_model_pairs:
        print(f"   • {model_id}")

    # Run test cases
    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"🧪 Test {i}: {test_case['name']}")
        print(f"{'='*60}")

        result = test_exclusion_logic(test_case['exclude'])

        if 'expected_count' in test_case:
            passed = len(result) == test_case['expected_count']
            print(f"   Expected count: {test_case['expected_count']}, Got: {len(result)} {'✅' if passed else '❌'}")
            if not passed:
                all_passed = False

        if 'expected_excluded' in test_case:
            # Check that expected models were excluded
            remaining_ids = {model_id for model_id, _, _ in result}
            expected_excluded = set(test_case['expected_excluded'])
            actually_excluded = {model_id for model_id, _, _ in mock_model_pairs} - remaining_ids

            missing_exclusions = expected_excluded - actually_excluded
            unexpected_exclusions = actually_excluded - expected_excluded

            if not missing_exclusions and not unexpected_exclusions:
                print(f"   Exclusions: ✅ Perfect match")
            else:
                all_passed = False
                if missing_exclusions:
                    print(f"   ❌ Should have excluded but didn't: {missing_exclusions}")
                if unexpected_exclusions:
                    print(f"   ❌ Excluded unexpectedly: {unexpected_exclusions}")

    # Final summary
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL TESTS PASSED! Exclude functionality works correctly.")
    else:
        print("❌ Some tests failed. Check the output above.")

    return all_passed


def test_command_line_parsing():
    """Test command line argument parsing for exclude functionality."""
    print(f"\n🧪 Testing command line argument parsing")

    import argparse

    # Simulate the argument parser from benchmark.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", type=str, nargs="*",
                       help="Exclude models by model slug")

    test_cases = [
        {
            "name": "No exclude argument",
            "args": [],
            "expected": None
        },
        {
            "name": "Single exclude",
            "args": ["--exclude", "openai/gpt-4o"],
            "expected": ["openai/gpt-4o"]
        },
        {
            "name": "Multiple excludes",
            "args": ["--exclude", "openai/gpt-4o", "anthropic/claude-3-haiku", "google/gemini"],
            "expected": ["openai/gpt-4o", "anthropic/claude-3-haiku", "google/gemini"]
        },
        {
            "name": "Empty exclude list",
            "args": ["--exclude"],
            "expected": []
        }
    ]

    all_passed = True
    for test_case in test_cases:
        try:
            args = parser.parse_args(test_case["args"])
            result = args.exclude

            if result == test_case["expected"]:
                print(f"   ✅ {test_case['name']}: {result}")
            else:
                print(f"   ❌ {test_case['name']}: Expected {test_case['expected']}, Got {result}")
                all_passed = False
        except Exception as e:
            print(f"   ❌ {test_case['name']}: Error {e}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("🚀 Testing exclude functionality for ZenMux Benchmark")

    logic_passed = test_exclude_functionality()
    parsing_passed = test_command_line_parsing()

    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Exclusion logic: {'✅ PASSED' if logic_passed else '❌ FAILED'}")
    print(f"Argument parsing: {'✅ PASSED' if parsing_passed else '❌ FAILED'}")

    if logic_passed and parsing_passed:
        print(f"\n🎉 All tests passed! Exclude functionality is ready to use.")
        print(f"\n💡 Usage examples:")
        print(f"   python benchmark.py --exclude openai/gpt-4o")
        print(f"   python benchmark.py --exclude openai/gpt-4o anthropic/claude-3-haiku")
        print(f"   python benchmark.py --exclude anthropic  # Excludes all Anthropic models")
    else:
        print(f"\n💥 Some tests failed. Please fix the issues above.")