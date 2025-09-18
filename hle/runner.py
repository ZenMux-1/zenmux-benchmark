"""HLE evaluation runner with ZenMux integration."""

import asyncio
import json
import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from .evaluation import HLEEvaluator
from .judge import HLEJudge
from zenmux import ZenMuxAPI
from config import BenchmarkConfig
from utils.logging import get_runner_logger


class HLERunner:
    """Main runner for HLE evaluations."""

    def __init__(self, config: BenchmarkConfig, batch_timestamp: str = None):
        self.config = config

        # Setup timestamped directories (this also initializes logging)
        if batch_timestamp:
            self.config.setup_timestamped_directories(batch_timestamp)
        else:
            self.config.setup_timestamped_directories()

        self.batch_timestamp = self.config.batch_timestamp
        self.logger = get_runner_logger()
        self.zenmux_api = ZenMuxAPI(config.zenmux)

        # Initialize evaluator with timestamped predictions directory
        self.evaluator = HLEEvaluator(
            config.hle,
            config.zenmux,
            config.get_predictions_dir(),
            batch_timestamp=self.batch_timestamp
        )
        self.judge = HLEJudge(config.hle, config.zenmux)

    async def run_single_model_evaluation(
        self,
        model_identifier: str,
        endpoint,
        text_only: bool = False,
        max_samples: Optional[int] = None,
        auto_judge: bool = True
    ) -> Dict[str, Any]:
        """Run evaluation for a single model endpoint."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🚀 EVALUATING: {model_identifier}")
        self.logger.info(f"{'='*60}")

        # Run prediction
        predictions_file = await self.evaluator.evaluate_model(
            model_identifier=model_identifier,
            endpoint=endpoint,
            text_only=text_only,
            max_samples=max_samples
        )

        results = {
            "model_identifier": model_identifier,
            "predictions_file": predictions_file,
            "judged_file": None,
            "metrics": None
        }

        # Validate evaluation completeness before judging
        evaluation_complete = self.validate_evaluation_completeness(predictions_file)

        # Run judging if requested and evaluation is complete
        if auto_judge:
            if evaluation_complete:
                self.logger.info(f"\n🏛️ JUDGING: {model_identifier}")
                judged_file = await self.judge.judge_predictions(
                    predictions_file=predictions_file,
                    dataset_name=self.config.hle.dataset_name,
                    output_dir=self.config.get_judged_dir()
                )
                results["judged_file"] = judged_file

                # Extract metrics from judged file
                results["metrics"] = self.extract_metrics_from_judged_file(judged_file)
            else:
                self.logger.warning(f"\n⚠️ SKIPPING JUDGING: {model_identifier} - evaluation incomplete after retries")
                results["error"] = "Evaluation incomplete after maximum retries"

        return results

    async def run_zenmux_models_evaluation(
        self,
        text_only: bool = False,
        max_samples: Optional[int] = None,
        auto_judge: bool = True,
        model_filter: Optional[str] = None,
        exclude_models: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Run evaluation for all ZenMux models with dual-layer concurrency."""
        self.logger.info("🌟 Starting ZenMux Models Evaluation")
        self.logger.info(f"📝 Text only: {text_only}")
        self.logger.info(f"📊 Max samples: {max_samples}")
        self.logger.info(f"🏛️ Auto judge: {auto_judge}")
        self.logger.info(f"🔄 Max concurrent models: {self.config.hle.max_concurrent_models}")
        self.logger.info(f"🔄 Workers per model: {self.config.hle.num_workers}")

        # Get all model-endpoint pairs
        model_endpoint_pairs = self.zenmux_api.get_all_model_endpoint_pairs(text_only=text_only)

        # Apply model filter if specified
        if model_filter:
            model_endpoint_pairs = [
                (model_id, model, endpoint)
                for model_id, model, endpoint in model_endpoint_pairs
                if model_filter.lower() in model_id.lower()
            ]

        # Apply model exclusion if specified
        if exclude_models:
            excluded_count = 0
            original_count = len(model_endpoint_pairs)

            # Extract model slugs from exclude list (handle both full names and partial matches)
            exclude_slugs = set(exclude_models)

            filtered_pairs = []
            for model_id, model, endpoint in model_endpoint_pairs:
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
                else:
                    excluded_count += 1

            model_endpoint_pairs = filtered_pairs

            if excluded_count > 0:
                self.logger.info(f"🚫 Excluded {excluded_count} model endpoints based on exclude patterns")
                self.logger.info(f"📉 Remaining models: {len(model_endpoint_pairs)} (was {original_count})")

        self.logger.info(f"🎯 Total model endpoints to evaluate: {len(model_endpoint_pairs)}")

        # Outer layer: Model-level concurrency control
        async def bound_model_evaluation(model_data):
            model_identifier, model, endpoint = model_data
            async with models_semaphore:
                self.logger.info(f"🚀 Starting evaluation: {model_identifier}")
                try:
                    result = await self.run_single_model_evaluation(
                        model_identifier=model_identifier,
                        endpoint=endpoint,
                        text_only=text_only,
                        max_samples=max_samples,
                        auto_judge=auto_judge
                    )
                    self.logger.info(f"✅ Completed evaluation: {model_identifier}")
                    return result

                except Exception as e:
                    self.logger.error(f"❌ Error evaluating {model_identifier}: {e}")
                    return {
                        "model_identifier": model_identifier,
                        "error": str(e),
                        "predictions_file": None,
                        "judged_file": None,
                        "metrics": None
                    }

        # Create semaphore for outer layer concurrency
        models_semaphore = asyncio.Semaphore(self.config.hle.max_concurrent_models)

        # Run all models concurrently with outer layer control
        tasks = [bound_model_evaluation(model_data) for model_data in model_endpoint_pairs]
        results = await asyncio.gather(*tasks)

        self.logger.info(f"\n✅ Completed evaluation of {len(results)} model endpoints")
        return results

    async def run_specific_model_evaluation(
        self,
        model_slug: str,
        provider_slug: str,
        text_only: bool = False,
        max_samples: Optional[int] = None,
        auto_judge: bool = True
    ) -> Dict[str, Any]:
        """Run evaluation for a specific model:provider combination."""
        # Get the specific model and endpoint
        model_endpoint_pairs = self.zenmux_api.get_all_model_endpoint_pairs(text_only=text_only)

        target_identifier = f"{model_slug}:{provider_slug}"

        for model_identifier, model, endpoint in model_endpoint_pairs:
            if model_identifier == target_identifier:
                return await self.run_single_model_evaluation(
                    model_identifier=model_identifier,
                    endpoint=endpoint,
                    text_only=text_only,
                    max_samples=max_samples,
                    auto_judge=auto_judge
                )

        raise ValueError(f"Model {target_identifier} not found in available models")

    def validate_evaluation_completeness(self, predictions_file: str) -> bool:
        """Validate that evaluation is complete by checking total_predictions == total_questions."""
        try:
            with open(predictions_file, "r") as f:
                data = json.load(f)

            # Extract metadata
            metadata = data.get("evaluation_metadata", {})
            statistics = metadata.get("statistics", {})

            total_questions = statistics.get("total_questions", 0)
            total_predictions = statistics.get("total_predictions", 0)

            if total_questions == 0:
                self.logger.warning(f"⚠️ Warning: No total_questions found in metadata")
                return False

            if total_predictions == total_questions:
                self.logger.info(f"✅ Evaluation complete: {total_predictions}/{total_questions} predictions")
                return True
            else:
                missing_count = total_questions - total_predictions
                self.logger.warning(f"❌ Evaluation incomplete: {total_predictions}/{total_questions} predictions ({missing_count} missing)")
                return False

        except Exception as e:
            self.logger.warning(f"⚠️ Warning: Could not validate evaluation completeness: {e}")
            return False

    def extract_metrics_from_judged_file(self, judged_file: str) -> Optional[Dict[str, Any]]:
        """Extract metrics from a judged file."""
        try:
            with open(judged_file, "r") as f:
                data = json.load(f)
                return data.get("metrics")
        except Exception as e:
            self.logger.warning(f"⚠️ Warning: Could not extract metrics from {judged_file}: {e}")
            return None

    def save_metrics_summary(self, results: List[Dict[str, Any]], run_metadata: Dict[str, Any] = None) -> str:
        """Save a unified metrics summary for all evaluations with comprehensive statistics."""
        if run_metadata is None:
            run_metadata = {}

        # Create enhanced summary data structure
        summary = {
            "summary_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_models": len(results),
                "run_metadata": run_metadata
            },
            "overall_statistics": {
                "models_with_predictions": 0,
                "models_with_complete_evaluations": 0,
                "models_with_judgments": 0,
                "models_with_complete_judgments": 0,
                "models_with_errors": 0
            },
            "failed_question_ids": {
                "evaluation_failures": {},  # model_id -> [failed_question_ids]
                "judge_failures": {}  # model_id -> [failed_question_ids]
            },
            "model_results": []
        }

        # Process each model result
        for result in results:
            model_identifier = result["model_identifier"]
            model_summary = {
                "model_identifier": model_identifier,
                "predictions_file": result.get("predictions_file"),
                "judged_file": result.get("judged_file"),
                "metrics": result.get("metrics"),
                "error": result.get("error"),
                "evaluation_statistics": {
                    "total_questions": 0,
                    "successful_predictions": 0,
                    "failed_predictions": 0,
                    "evaluation_complete": False,
                    "failed_question_ids": []
                },
                "judge_statistics": {
                    "total_questions": 0,
                    "successful_judgments": 0,
                    "failed_judgments": 0,
                    "judge_complete": False,
                    "failed_question_ids": []
                }
            }

            # Analyze evaluation statistics
            if result.get("predictions_file"):
                eval_stats = self._analyze_evaluation_file(result["predictions_file"])
                model_summary["evaluation_statistics"] = eval_stats

                if eval_stats["successful_predictions"] > 0:
                    summary["overall_statistics"]["models_with_predictions"] += 1

                if eval_stats["evaluation_complete"]:
                    summary["overall_statistics"]["models_with_complete_evaluations"] += 1

                if eval_stats["failed_question_ids"]:
                    summary["failed_question_ids"]["evaluation_failures"][model_identifier] = eval_stats["failed_question_ids"]

            # Analyze judge statistics
            if result.get("judged_file"):
                judge_stats = self._analyze_judge_file(result["judged_file"])
                model_summary["judge_statistics"] = judge_stats

                if judge_stats["successful_judgments"] > 0:
                    summary["overall_statistics"]["models_with_judgments"] += 1

                if judge_stats["judge_complete"]:
                    summary["overall_statistics"]["models_with_complete_judgments"] += 1

                if judge_stats["failed_question_ids"]:
                    summary["failed_question_ids"]["judge_failures"][model_identifier] = judge_stats["failed_question_ids"]

            # Check for general errors
            if result.get("error"):
                summary["overall_statistics"]["models_with_errors"] += 1

            summary["model_results"].append(model_summary)

        # Save summary file in the timestamped run directory
        summary_file = os.path.join(self.config.run_dir, f"metrics_summary_{self.batch_timestamp}.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)

        self.logger.info(f"📊 Enhanced metrics summary saved to: {summary_file}")
        return summary_file

    def _analyze_evaluation_file(self, predictions_file: str) -> Dict[str, Any]:
        """Analyze predictions file to extract evaluation statistics."""
        eval_stats = {
            "total_questions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "evaluation_complete": False,
            "failed_question_ids": []
        }

        try:
            with open(predictions_file, "r") as f:
                data = json.load(f)

            # Handle both old format (direct predictions) and new format (with metadata)
            if "predictions" in data:
                predictions = data["predictions"]
                evaluation_metadata = data.get("evaluation_metadata", {})
                statistics = evaluation_metadata.get("statistics", {})
                eval_stats["total_questions"] = statistics.get("total_questions", 0)
            else:
                predictions = data
                # For old format, we don't have total_questions metadata
                eval_stats["total_questions"] = len(predictions)

            # Count successful predictions
            eval_stats["successful_predictions"] = len(predictions)
            eval_stats["failed_predictions"] = eval_stats["total_questions"] - eval_stats["successful_predictions"]

            # Check if evaluation is complete
            if eval_stats["total_questions"] > 0:
                eval_stats["evaluation_complete"] = (eval_stats["successful_predictions"] == eval_stats["total_questions"])

            # For missing question IDs, we need to load the dataset to know which questions should exist
            if eval_stats["failed_predictions"] > 0:
                # Load dataset to get all question IDs
                try:
                    from .dataset import HLEDataset
                    dataset = HLEDataset(self.config.hle.dataset_name, self.config.hle.dataset_split)

                    # Get the max_samples from evaluation metadata if available
                    max_samples = None
                    text_only = False
                    if "evaluation_metadata" in data:
                        dataset_config = data["evaluation_metadata"].get("dataset_config", {})
                        max_samples = dataset_config.get("max_samples")
                        text_only = dataset_config.get("text_only", False)

                    all_questions = dataset.get_questions(text_only=text_only, max_samples=max_samples)
                    all_question_ids = {q["id"] for q in all_questions}
                    predicted_question_ids = set(predictions.keys())

                    eval_stats["failed_question_ids"] = list(all_question_ids - predicted_question_ids)

                except Exception as e:
                    self.logger.warning(f"⚠️ Warning: Could not determine failed question IDs for {predictions_file}: {e}")

        except Exception as e:
            self.logger.warning(f"⚠️ Warning: Could not analyze evaluation file {predictions_file}: {e}")

        return eval_stats

    def _analyze_judge_file(self, judged_file: str) -> Dict[str, Any]:
        """Analyze judged file to extract judge statistics."""
        judge_stats = {
            "total_questions": 0,
            "successful_judgments": 0,
            "failed_judgments": 0,
            "judge_complete": False,
            "failed_question_ids": []
        }

        try:
            with open(judged_file, "r") as f:
                data = json.load(f)

            # Handle both old format (direct predictions) and new format (with metadata)
            if "judged_predictions" in data:
                judged_predictions = data["judged_predictions"]
                judging_metadata = data.get("judging_metadata", {})
                statistics = judging_metadata.get("statistics", {})
                judge_stats["total_questions"] = statistics.get("total_questions", 0)
            else:
                judged_predictions = data
                judge_stats["total_questions"] = len(judged_predictions)

            # Count successful judgments (those with judge_response)
            successful_count = 0
            failed_ids = []

            for question_id, prediction in judged_predictions.items():
                if "judge_response" in prediction:
                    successful_count += 1
                else:
                    failed_ids.append(question_id)

            judge_stats["successful_judgments"] = successful_count
            judge_stats["failed_judgments"] = len(failed_ids)
            judge_stats["failed_question_ids"] = failed_ids

            # Check if judging is complete
            if judge_stats["total_questions"] > 0:
                judge_stats["judge_complete"] = (judge_stats["failed_judgments"] == 0)

        except Exception as e:
            self.logger.warning(f"⚠️ Warning: Could not analyze judge file {judged_file}: {e}")

        return judge_stats

    def log_summary(self, results: List[Dict[str, Any]]):
        """Log an enhanced summary of evaluation results."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("📊 ENHANCED EVALUATION SUMMARY")
        self.logger.info(f"{'='*60}")

        # Count overall statistics
        models_with_predictions = 0
        models_with_complete_evaluations = 0
        models_with_judgments = 0
        models_with_complete_judgments = 0
        models_with_errors = 0
        models_with_metrics = 0

        for result in results:
            if result.get("predictions_file"):
                models_with_predictions += 1
                # Check if evaluation is complete
                if self.validate_evaluation_completeness(result["predictions_file"]):
                    models_with_complete_evaluations += 1

            if result.get("judged_file"):
                models_with_judgments += 1

            if result.get("error"):
                models_with_errors += 1

            if result.get("metrics"):
                models_with_metrics += 1

        # Display overall statistics
        self.logger.info(f"📈 Total models: {len(results)}")
        self.logger.info(f"📄 Models with predictions: {models_with_predictions}")
        self.logger.info(f"✅ Models with complete evaluations: {models_with_complete_evaluations}")
        self.logger.info(f"🏛️ Models with judgments: {models_with_judgments}")
        self.logger.info(f"📊 Models with metrics: {models_with_metrics}")
        self.logger.info(f"❌ Models with errors: {models_with_errors}")

        # Show detailed breakdown
        if models_with_errors > 0:
            self.logger.info(f"\n❌ MODELS WITH ERRORS:")
            for result in results:
                if result.get("error"):
                    error_msg = result.get('error', 'Unknown error')
                    self.logger.info(f"  - {result['model_identifier']}: {error_msg}")

        if models_with_predictions > 0:
            self.logger.info(f"\n📊 EVALUATION STATUS:")
            for result in results:
                if result.get("predictions_file"):
                    model_id = result['model_identifier']
                    eval_complete = self.validate_evaluation_completeness(result["predictions_file"])
                    judge_complete = result.get("judged_file") is not None
                    has_metrics = result.get("metrics") is not None

                    status_parts = []
                    if eval_complete:
                        status_parts.append("✅ Eval")
                    else:
                        status_parts.append("⚠️ Eval")

                    if judge_complete:
                        status_parts.append("🏛️ Judge")
                    else:
                        status_parts.append("❌ Judge")

                    if has_metrics:
                        status_parts.append("📊 Metrics")

                    status = " | ".join(status_parts)
                    self.logger.info(f"  {model_id}: {status}")

        # Log metrics for models that have them
        models_with_complete_metrics = [r for r in results if r.get("metrics") is not None]
        if models_with_complete_metrics:
            self.logger.info(f"\n📊 METRICS DETAILS")
            self.logger.info(f"{'='*60}")
            for result in models_with_complete_metrics:
                metrics = result["metrics"]
                self.logger.info(f"\n🎯 {result['model_identifier']}")
                self.logger.info(f"📊 Accuracy: {metrics['accuracy']}% +/- {metrics['confidence_interval']}% | n = {metrics['total_questions']}")
                self.logger.info(f"📏 Calibration Error: {metrics['calibration_error']}")
                self.logger.info(f"✅ Evaluated: {metrics['total_evaluated']} / {metrics['total_questions']}")

        # File locations
        self.logger.info(f"\n📁 FILES:")
        self.logger.info(f"📁 Run directory: {self.config.run_dir}")
        self.logger.info(f"📁 Prediction files: {self.config.get_predictions_dir()}")
        self.logger.info(f"📁 Judged files: {self.config.get_judged_dir()}")