#!/usr/bin/env python3
"""
Fix model identifiers in judged prediction JSON files.

This script reads JSON files in a specified judged directory and updates the 'model' field
in each judged prediction entry to match the correct model_identifier from judging_metadata.evaluation_metadata.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any


def fix_model_identifier_in_judged_file(file_path: Path) -> bool:
    """
    Fix model identifiers in a single judged JSON file.

    Args:
        file_path: Path to the judged JSON file to fix

    Returns:
        True if the file was modified, False otherwise
    """
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract the correct model identifier from judging_metadata.evaluation_metadata
        if 'judging_metadata' not in data:
            print(f"Warning: No judging_metadata found in {file_path}")
            return False

        evaluation_metadata = data['judging_metadata'].get('evaluation_metadata')
        if not evaluation_metadata:
            print(f"Warning: No evaluation_metadata found in judging_metadata in {file_path}")
            return False

        correct_model_id = evaluation_metadata.get('model_identifier')
        if not correct_model_id:
            print(f"Warning: No model_identifier found in evaluation_metadata in {file_path}")
            return False

        # Check if judged_predictions exist
        if 'judged_predictions' not in data:
            print(f"Warning: No judged_predictions found in {file_path}")
            return False

        # Track if any changes were made
        changes_made = False

        # Update model field in each judged prediction
        for question_id, prediction in data['judged_predictions'].items():
            if 'model' in prediction:
                old_model = prediction['model']
                if old_model != correct_model_id:
                    prediction['model'] = correct_model_id
                    changes_made = True
                    print(f"Updated model in {file_path}, question {question_id}: {old_model} -> {correct_model_id}")
            else:
                # Add model field if it doesn't exist
                prediction['model'] = correct_model_id
                changes_made = True
                print(f"Added model field in {file_path}, question {question_id}: {correct_model_id}")

        # Write back the file if changes were made
        if changes_made:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=True, indent=4)
            print(f"Successfully updated {file_path}")
            return True
        else:
            print(f"No changes needed for {file_path}")
            return False

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def fix_judged_model_identifiers_in_directory(directory_path: Path) -> None:
    """
    Fix model identifiers in all judged JSON files in the specified directory.

    Args:
        directory_path: Path to the judged directory containing JSON files
    """
    if not directory_path.exists():
        print(f"Error: Directory {directory_path} does not exist")
        return

    if not directory_path.is_dir():
        print(f"Error: {directory_path} is not a directory")
        return

    # Find all JSON files in the directory
    json_files = list(directory_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return

    print(f"Found {len(json_files)} JSON files in {directory_path}")

    # Process each JSON file
    files_modified = 0
    for json_file in json_files:
        print(f"\nProcessing {json_file.name}...")
        if fix_model_identifier_in_judged_file(json_file):
            files_modified += 1

    print(f"\n=== Summary ===")
    print(f"Total files processed: {len(json_files)}")
    print(f"Files modified: {files_modified}")
    print(f"Files unchanged: {len(json_files) - files_modified}")


def main():
    """Main function to handle command line arguments and run the fix."""
    if len(sys.argv) != 2:
        print("Usage: python fix_judged_model_identifiers.py <judged_directory_path>")
        print("\nExample:")
        print("  python fix_judged_model_identifiers.py results/20250922_122904/judged/")
        sys.exit(1)

    directory_path = Path(sys.argv[1])

    print(f"Fixing model identifiers in judged JSON files in: {directory_path}")
    print("=" * 60)

    fix_judged_model_identifiers_in_directory(directory_path)


if __name__ == "__main__":
    main()