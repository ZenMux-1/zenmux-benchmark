"""Statistics generation functions for HLE evaluations and judging."""

import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


def generate_evaluation_statistics(results_dir: str, output_dir: Optional[str] = None) -> str:
    """Generate comprehensive evaluation statistics for all models in a results directory.

    Args:
        results_dir: Directory containing prediction files
        output_dir: Optional output directory (defaults to results_dir)

    Returns:
        Path to the generated statistics file
    """
    if output_dir is None:
        output_dir = results_dir

    # Find all prediction files
    prediction_pattern = os.path.join(results_dir, "predictions", "hle_*.json")
    prediction_files = glob.glob(prediction_pattern)

    statistics = {
        "generation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "results_directory": results_dir,
            "total_files_processed": len(prediction_files)
        },
        "model_statistics": {},
        "failure_lists": {
            "incomplete_evaluation_models": [],
            "models_without_responses": []
        },
        "summary": {
            "total_models": 0,
            "models_with_responses": 0,
            "models_with_complete_evaluation": 0,
            "total_questions_across_all_models": 0,
            "total_responses_across_all_models": 0
        }
    }

    total_questions = None

    for prediction_file in prediction_files:
        try:
            with open(prediction_file, 'r') as f:
                data = json.load(f)

            # Handle both old format (direct predictions) and new format (with metadata)
            if "predictions" in data:
                predictions = data["predictions"]
                evaluation_metadata = data.get("evaluation_metadata", {})
                model_identifier = evaluation_metadata.get("model_identifier", "unknown")
            else:
                predictions = data
                model_identifier = "unknown"

            # Calculate statistics for this model
            total_predictions = len(predictions)
            responses_with_content = sum(1 for pred in predictions.values() if pred.get("response", "").strip())
            responses_without_content = total_predictions - responses_with_content

            # Set total questions if not set (should be same for all models)
            if total_questions is None:
                total_questions = total_predictions

            model_stats = {
                "prediction_file": os.path.basename(prediction_file),
                "total_questions": total_predictions,
                "responses_with_content": responses_with_content,
                "responses_without_content": responses_without_content,
                "completion_rate": round((responses_with_content / total_predictions * 100), 2) if total_predictions > 0 else 0.0,
                "is_complete": responses_without_content == 0
            }

            statistics["model_statistics"][model_identifier] = model_stats

            # Track failure lists
            if responses_without_content > 0:
                statistics["failure_lists"]["incomplete_evaluation_models"].append(model_identifier)
            if responses_with_content == 0:
                statistics["failure_lists"]["models_without_responses"].append(model_identifier)

            # Update summary
            statistics["summary"]["total_models"] += 1
            if responses_with_content > 0:
                statistics["summary"]["models_with_responses"] += 1
            if responses_without_content == 0:
                statistics["summary"]["models_with_complete_evaluation"] += 1
            statistics["summary"]["total_responses_across_all_models"] += responses_with_content

        except Exception as e:
            # Record error for this file
            statistics["model_statistics"][f"ERROR_{os.path.basename(prediction_file)}"] = {
                "error": str(e),
                "prediction_file": os.path.basename(prediction_file)
            }

    # Set total questions in summary
    if total_questions is not None:
        statistics["summary"]["total_questions_across_all_models"] = total_questions * statistics["summary"]["total_models"]

    # Generate output filename with timestamp from results directory
    results_basename = os.path.basename(results_dir.rstrip('/'))
    if results_basename:
        output_filename = f"evaluation_statistics_{results_basename}.json"
    else:
        output_filename = f"evaluation_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_filepath = os.path.join(output_dir, output_filename)

    # Write statistics file
    with open(output_filepath, 'w') as f:
        json.dump(statistics, f, indent=4)

    return output_filepath


def generate_judge_statistics(results_dir: str, output_dir: Optional[str] = None) -> str:
    """Generate comprehensive judge statistics for all models in a results directory.

    Args:
        results_dir: Directory containing judged files
        output_dir: Optional output directory (defaults to results_dir)

    Returns:
        Path to the generated statistics file
    """
    if output_dir is None:
        output_dir = results_dir

    # Find all judged files
    judged_pattern = os.path.join(results_dir, "judged", "judged_hle_*.json")
    judged_files = glob.glob(judged_pattern)

    statistics = {
        "generation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "results_directory": results_dir,
            "total_files_processed": len(judged_files)
        },
        "model_statistics": {},
        "failure_lists": {
            "incomplete_judging_models": [],
            "models_without_judgments": []
        },
        "summary": {
            "total_models": 0,
            "models_with_judgments": 0,
            "models_with_complete_judging": 0,
            "total_questions_across_all_models": 0,
            "total_judgments_across_all_models": 0
        }
    }

    total_questions = None

    for judged_file in judged_files:
        try:
            with open(judged_file, 'r') as f:
                data = json.load(f)

            # Handle both old format (direct predictions) and new format (with metadata)
            if "judged_predictions" in data:
                predictions = data["judged_predictions"]
                judging_metadata = data.get("judging_metadata", {})
                # Try to get model identifier from evaluation metadata
                evaluation_metadata = judging_metadata.get("evaluation_metadata", {})
                model_identifier = evaluation_metadata.get("model_identifier", "unknown")
            else:
                predictions = data
                model_identifier = "unknown"

            # Calculate statistics for this model
            total_predictions = len(predictions)
            judgments_with_content = sum(1 for pred in predictions.values()
                                       if pred.get("judge_response", {}).get("reasoning", "").strip())
            judgments_without_content = total_predictions - judgments_with_content

            # Set total questions if not set (should be same for all models)
            if total_questions is None:
                total_questions = total_predictions

            model_stats = {
                "judged_file": os.path.basename(judged_file),
                "total_questions": total_predictions,
                "judgments_with_content": judgments_with_content,
                "judgments_without_content": judgments_without_content,
                "completion_rate": round((judgments_with_content / total_predictions * 100), 2) if total_predictions > 0 else 0.0,
                "is_complete": judgments_without_content == 0
            }

            statistics["model_statistics"][model_identifier] = model_stats

            # Track failure lists
            if judgments_without_content > 0:
                statistics["failure_lists"]["incomplete_judging_models"].append(model_identifier)
            if judgments_with_content == 0:
                statistics["failure_lists"]["models_without_judgments"].append(model_identifier)

            # Update summary
            statistics["summary"]["total_models"] += 1
            if judgments_with_content > 0:
                statistics["summary"]["models_with_judgments"] += 1
            if judgments_without_content == 0:
                statistics["summary"]["models_with_complete_judging"] += 1
            statistics["summary"]["total_judgments_across_all_models"] += judgments_with_content

        except Exception as e:
            # Record error for this file
            statistics["model_statistics"][f"ERROR_{os.path.basename(judged_file)}"] = {
                "error": str(e),
                "judged_file": os.path.basename(judged_file)
            }

    # Set total questions in summary
    if total_questions is not None:
        statistics["summary"]["total_questions_across_all_models"] = total_questions * statistics["summary"]["total_models"]

    # Generate output filename with timestamp from results directory
    results_basename = os.path.basename(results_dir.rstrip('/'))
    if results_basename:
        output_filename = f"judge_statistics_{results_basename}.json"
    else:
        output_filename = f"judge_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_filepath = os.path.join(output_dir, output_filename)

    # Write statistics file
    with open(output_filepath, 'w') as f:
        json.dump(statistics, f, indent=4)

    return output_filepath


def generate_metrics_statistics(results_dir: str, output_dir: Optional[str] = None) -> str:
    """Generate meta statistics about the metrics summary file.

    Args:
        results_dir: Directory containing metrics summary file
        output_dir: Optional output directory (defaults to results_dir)

    Returns:
        Path to the generated statistics file
    """
    if output_dir is None:
        output_dir = results_dir

    # Find metrics summary file
    results_basename = os.path.basename(results_dir.rstrip('/'))
    metrics_summary_file = os.path.join(results_dir, f"metrics_summary_{results_basename}.json")

    if not os.path.exists(metrics_summary_file):
        raise FileNotFoundError(f"Metrics summary file not found: {metrics_summary_file}")

    with open(metrics_summary_file, 'r') as f:
        summary_data = json.load(f)

    # Initialize statistics
    statistics = {
        "generation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "results_directory": results_dir,
            "metrics_summary_file": os.path.basename(metrics_summary_file)
        },
        "metrics_summary_statistics": {
            "total_models": 0,
            "models_with_valid_metrics": 0,
            "models_excluded_from_metrics": 0,
            "models_with_errors": 0
        },
        "failure_lists": {
            "excluded_models": [],
            "models_with_errors": []
        },
        "exclusion_reasons": {},
        "error_summary": {}
    }

    model_results = summary_data.get("model_results", [])

    for model_result in model_results:
        model_identifier = model_result.get("model_identifier", "unknown")
        metrics = model_result.get("metrics")
        excluded = model_result.get("excluded_from_metrics", False)
        error = model_result.get("error")
        exclusion_reason = model_result.get("exclusion_reason", "")

        # Count totals
        statistics["metrics_summary_statistics"]["total_models"] += 1

        # Count valid metrics (not None and not excluded)
        if metrics is not None and not excluded:
            statistics["metrics_summary_statistics"]["models_with_valid_metrics"] += 1

        # Count exclusions
        if excluded:
            statistics["metrics_summary_statistics"]["models_excluded_from_metrics"] += 1
            statistics["failure_lists"]["excluded_models"].append(model_identifier)
            # Track exclusion reasons
            if exclusion_reason:
                for reason in exclusion_reason.split(", "):
                    reason = reason.strip()
                    if reason not in statistics["exclusion_reasons"]:
                        statistics["exclusion_reasons"][reason] = 0
                    statistics["exclusion_reasons"][reason] += 1

        # Count errors
        if error:
            statistics["metrics_summary_statistics"]["models_with_errors"] += 1
            statistics["failure_lists"]["models_with_errors"].append(model_identifier)
            # Track error types/messages (simplified)
            error_type = "timeout" if "timeout" in error.lower() else "evaluation_failure"
            if error_type not in statistics["error_summary"]:
                statistics["error_summary"][error_type] = 0
            statistics["error_summary"][error_type] += 1

    # Generate output filename
    output_filename = f"metrics_statistics_{results_basename}.json"
    output_filepath = os.path.join(output_dir, output_filename)

    # Write statistics file
    with open(output_filepath, 'w') as f:
        json.dump(statistics, f, indent=4)

    return output_filepath