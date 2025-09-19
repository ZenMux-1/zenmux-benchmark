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

    def save_question_ids(self, text_only: bool = False, max_samples: Optional[int] = None) -> str:
        """Save the question IDs used in this evaluation run to a separate file."""
        from .dataset import HLEDataset

        # Get the same questions that would be used in evaluation
        dataset = HLEDataset(self.config.hle.dataset_name, self.config.hle.dataset_split)
        questions = dataset.get_questions(text_only=text_only, max_samples=max_samples)

        # Create question IDs data structure
        question_ids_data = {
            "run_metadata": {
                "timestamp": datetime.now().isoformat(),
                "batch_timestamp": self.batch_timestamp,
                "dataset_name": self.config.hle.dataset_name,
                "dataset_split": self.config.hle.dataset_split,
                "text_only": text_only,
                "max_samples": max_samples,
                "total_questions": len(questions)
            },
            "question_ids": [q["id"] for q in questions]
        }

        # Save to the timestamped run directory
        question_ids_file = os.path.join(self.config.run_dir, f"question_ids_{self.batch_timestamp}.json")
        with open(question_ids_file, "w") as f:
            json.dump(question_ids_data, f, indent=4)

        self.logger.info(f"📝 Question IDs saved to: {question_ids_file}")
        return question_ids_file

    def save_available_models(self, model_endpoint_pairs: List, text_only: bool = False, model_filter: Optional[str] = None, exclude_models: Optional[List[str]] = None) -> str:
        """Save the available models list to a separate file."""
        # Create models data structure
        models_data = {
            "discovery_metadata": {
                "timestamp": datetime.now().isoformat(),
                "batch_timestamp": self.batch_timestamp,
                "text_only": text_only,
                "model_filter": model_filter,
                "exclude_models": exclude_models,
                "total_available_models": len(model_endpoint_pairs)
            },
            "available_models": []
        }

        # Extract model information with detailed metadata
        available_models = {}
        for model_id, model, endpoint in model_endpoint_pairs:
            # Only include models and endpoints with visible=1
            if model.visible == 1 and endpoint.visible == 1:
                available_models[model_id] = {
                    "max_completion_tokens": endpoint.max_completion_tokens,
                    "context_length": endpoint.context_length,
                    "supports_reasoning": endpoint.supports_reasoning
                }

        models_data["available_models"] = available_models
        models_data["discovery_metadata"]["total_available_models"] = len(available_models)

        # Save to the timestamped run directory
        models_file = os.path.join(self.config.run_dir, f"available_models_{self.batch_timestamp}.json")
        with open(models_file, "w") as f:
            json.dump(models_data, f, indent=4)

        self.logger.info(f"📋 Available models saved to: {models_file}")
        return models_file

    def save_failures_to_files(self, results: List[Dict[str, Any]]) -> tuple[str, str]:
        """Save evaluation and judge failures to separate files."""
        evaluation_failures = {}
        judge_failures = {}

        for result in results:
            model_identifier = result["model_identifier"]

            # Analyze evaluation failures
            if result.get("predictions_file"):
                eval_stats = self._analyze_evaluation_file(result["predictions_file"])
                if eval_stats["failed_question_ids"]:
                    evaluation_failures[model_identifier] = eval_stats["failed_question_ids"]

            # Analyze judge failures
            if result.get("judged_file"):
                judge_stats = self._analyze_judge_file(result["judged_file"])
                if judge_stats["failed_question_ids"]:
                    judge_failures[model_identifier] = judge_stats["failed_question_ids"]

        # Save evaluation failures
        eval_failures_data = {
            "failure_metadata": {
                "timestamp": datetime.now().isoformat(),
                "batch_timestamp": self.batch_timestamp,
                "total_models_with_evaluation_failures": len(evaluation_failures)
            },
            "evaluation_failures": evaluation_failures
        }

        eval_failures_file = os.path.join(self.config.run_dir, f"evaluation_failures_{self.batch_timestamp}.json")
        with open(eval_failures_file, "w") as f:
            json.dump(eval_failures_data, f, indent=4)

        # Save judge failures
        judge_failures_data = {
            "failure_metadata": {
                "timestamp": datetime.now().isoformat(),
                "batch_timestamp": self.batch_timestamp,
                "total_models_with_judge_failures": len(judge_failures)
            },
            "judge_failures": judge_failures
        }

        judge_failures_file = os.path.join(self.config.run_dir, f"judge_failures_{self.batch_timestamp}.json")
        with open(judge_failures_file, "w") as f:
            json.dump(judge_failures_data, f, indent=4)

        self.logger.info(f"❌ Evaluation failures saved to: {eval_failures_file}")
        self.logger.info(f"⚖️ Judge failures saved to: {judge_failures_file}")

        return eval_failures_file, judge_failures_file

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

        # Run judging if requested (always run, regardless of evaluation completeness)
        if auto_judge:
            self.logger.info(f"\n🏛️ JUDGING: {model_identifier}")
            judged_file = await self.judge.judge_predictions(
                predictions_file=predictions_file,
                dataset_name=self.config.hle.dataset_name,
                output_dir=self.config.get_judged_dir()
            )
            results["judged_file"] = judged_file

            # Extract metrics from judged file only if evaluation was complete
            if evaluation_complete:
                results["metrics"] = self.extract_metrics_from_judged_file(judged_file)
            else:
                self.logger.warning(f"\n⚠️ Evaluation incomplete: {model_identifier} - metrics will not be calculated")
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

        # Save available models list to file
        self.save_available_models(model_endpoint_pairs, text_only=text_only, model_filter=model_filter, exclude_models=exclude_models)

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

        # Find the target model and save it to available_models file
        target_model_pair = None
        for model_identifier, model, endpoint in model_endpoint_pairs:
            if model_identifier == target_identifier:
                target_model_pair = (model_identifier, model, endpoint)
                break

        if target_model_pair is None:
            raise ValueError(f"Model {target_identifier} not found in available models")

        # Save the single model info to available_models file
        self.save_available_models([target_model_pair], text_only=text_only, model_filter=None, exclude_models=None)

        # Run the evaluation
        model_identifier, model, endpoint = target_model_pair
        return await self.run_single_model_evaluation(
            model_identifier=model_identifier,
            endpoint=endpoint,
            text_only=text_only,
            max_samples=max_samples,
            auto_judge=auto_judge
        )

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

        # Save failures to separate files first
        eval_failures_file, judge_failures_file = self.save_failures_to_files(results)

        # Load question IDs file to get expected question count for validation
        question_ids_file = os.path.join(self.config.run_dir, f"question_ids_{self.batch_timestamp}.json")
        expected_question_count = 0

        if os.path.exists(question_ids_file):
            try:
                with open(question_ids_file, "r") as f:
                    question_ids_data = json.load(f)
                    expected_question_count = len(question_ids_data.get("question_ids", []))
                    self.logger.info(f"📊 Expected questions for metrics validation: {expected_question_count}")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load question IDs for validation: {e}")

        # Identify models with failures (should be excluded from metrics calculation)
        models_with_failures = set()

        for result in results:
            model_identifier = result["model_identifier"]

            # Check for evaluation failures
            if result.get("predictions_file"):
                eval_stats = self._analyze_evaluation_file(result["predictions_file"])
                if eval_stats["failed_question_ids"] or not eval_stats["evaluation_complete"]:
                    models_with_failures.add(model_identifier)

            # Check for judge failures
            if result.get("judged_file"):
                judge_stats = self._analyze_judge_file(result["judged_file"])
                if judge_stats["failed_question_ids"] or not judge_stats["judge_complete"]:
                    models_with_failures.add(model_identifier)

                # ENHANCED: Validate judge completeness against expected question count
                if expected_question_count > 0 and result.get("judged_file"):
                    try:
                        with open(result["judged_file"], "r") as f:
                            judged_data = json.load(f)

                        if "judged_predictions" in judged_data:
                            judged_predictions = judged_data["judged_predictions"]
                        else:
                            judged_predictions = judged_data

                        # Count successful judgments
                        successful_judgments = sum(1 for pred in judged_predictions.values() if "judge_response" in pred)

                        if successful_judgments != expected_question_count:
                            models_with_failures.add(model_identifier)
                            self.logger.warning(f"⚠️ {model_identifier}: Only {successful_judgments}/{expected_question_count} judgments - excluded from metrics")
                        else:
                            self.logger.info(f"✅ {model_identifier}: {successful_judgments}/{expected_question_count} judgments - valid for metrics")

                    except Exception as e:
                        self.logger.error(f"❌ Error validating judge completeness for {model_identifier}: {e}")
                        models_with_failures.add(model_identifier)

            # Check for general errors
            if result.get("error"):
                models_with_failures.add(model_identifier)

        # Filter results to only include models without failures for metrics calculation
        clean_results = [r for r in results if r["model_identifier"] not in models_with_failures]

        # Create enhanced summary data structure
        summary = {
            "summary_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_models": len(results),
                "models_with_failures": len(models_with_failures),
                "models_included_in_metrics": len(clean_results),
                "evaluation_failures_file": eval_failures_file,
                "judge_failures_file": judge_failures_file,
                "run_metadata": run_metadata
            },
            "overall_statistics": {
                "models_with_predictions": 0,
                "models_with_complete_evaluations": 0,
                "models_with_judgments": 0,
                "models_with_complete_judgments": 0,
                "models_with_errors": 0,
                "models_excluded_from_metrics": len(models_with_failures)
            },
            "model_results": []
        }

        # Process each model result (including failed ones for statistics)
        for result in results:
            model_identifier = result["model_identifier"]
            is_excluded = model_identifier in models_with_failures

            model_summary = {
                "model_identifier": model_identifier,
                "predictions_file": result.get("predictions_file"),
                "judged_file": result.get("judged_file"),
                "metrics": result.get("metrics") if not is_excluded else None,  # Exclude metrics for failed models
                "error": result.get("error"),
                "excluded_from_metrics": is_excluded,
                "exclusion_reason": self._get_exclusion_reason(result) if is_excluded else None,
                "evaluation_statistics": {
                    "total_questions": 0,
                    "successful_predictions": 0,
                    "failed_predictions": 0,
                    "evaluation_complete": False
                },
                "judge_statistics": {
                    "total_questions": 0,
                    "successful_judgments": 0,
                    "failed_judgments": 0,
                    "judge_complete": False
                }
            }

            # Analyze evaluation statistics
            if result.get("predictions_file"):
                eval_stats = self._analyze_evaluation_file(result["predictions_file"])
                model_summary["evaluation_statistics"] = {
                    "total_questions": eval_stats["total_questions"],
                    "successful_predictions": eval_stats["successful_predictions"],
                    "failed_predictions": eval_stats["failed_predictions"],
                    "evaluation_complete": eval_stats["evaluation_complete"]
                }

                if eval_stats["successful_predictions"] > 0:
                    summary["overall_statistics"]["models_with_predictions"] += 1

                if eval_stats["evaluation_complete"]:
                    summary["overall_statistics"]["models_with_complete_evaluations"] += 1

            # Analyze judge statistics
            if result.get("judged_file"):
                judge_stats = self._analyze_judge_file(result["judged_file"])
                model_summary["judge_statistics"] = {
                    "total_questions": judge_stats["total_questions"],
                    "successful_judgments": judge_stats["successful_judgments"],
                    "failed_judgments": judge_stats["failed_judgments"],
                    "judge_complete": judge_stats["judge_complete"]
                }

                if judge_stats["successful_judgments"] > 0:
                    summary["overall_statistics"]["models_with_judgments"] += 1

                if judge_stats["judge_complete"]:
                    summary["overall_statistics"]["models_with_complete_judgments"] += 1

            # Check for general errors
            if result.get("error"):
                summary["overall_statistics"]["models_with_errors"] += 1

            summary["model_results"].append(model_summary)

        # Save summary file in the timestamped run directory
        summary_file = os.path.join(self.config.run_dir, f"metrics_summary_{self.batch_timestamp}.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)

        self.logger.info(f"📊 Enhanced metrics summary saved to: {summary_file}")
        self.logger.info(f"🔍 Models excluded from metrics: {len(models_with_failures)}/{len(results)}")

        return summary_file

    def _get_exclusion_reason(self, result: Dict[str, Any]) -> str:
        """Get the reason why a model was excluded from metrics calculation."""
        reasons = []

        if result.get("error"):
            reasons.append("general_error")

        if result.get("predictions_file"):
            eval_stats = self._analyze_evaluation_file(result["predictions_file"])
            if eval_stats["failed_question_ids"]:
                reasons.append("evaluation_failures")
            if not eval_stats["evaluation_complete"]:
                reasons.append("incomplete_evaluation")

        if result.get("judged_file"):
            judge_stats = self._analyze_judge_file(result["judged_file"])
            if judge_stats["failed_question_ids"]:
                reasons.append("judge_failures")
            if not judge_stats["judge_complete"]:
                reasons.append("incomplete_judging")

        return ", ".join(reasons) if reasons else "unknown"

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

    async def fix_evaluation_failures(self, timestamp_dir: str) -> Dict[str, Any]:
        """Fix evaluation failures by re-evaluating failed questions."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🔧 FIXING EVALUATION FAILURES")
        self.logger.info(f"{'='*60}")

        # Load evaluation failures file
        eval_failures_file = os.path.join(timestamp_dir, f"evaluation_failures_{os.path.basename(timestamp_dir)}.json")
        if not os.path.exists(eval_failures_file):
            self.logger.error(f"❌ Evaluation failures file not found: {eval_failures_file}")
            return {"error": "Evaluation failures file not found"}

        with open(eval_failures_file, "r") as f:
            failures_data = json.load(f)

        evaluation_failures = failures_data.get("evaluation_failures", {})
        if not evaluation_failures:
            self.logger.info("✅ No evaluation failures to fix")
            return {
                "fixed_models": [],
                "still_failed_models": [],
                "fixed_count": 0,
                "remaining_failures": 0
            }

        self.logger.info(f"🔍 Found {len(evaluation_failures)} models with evaluation failures")

        # Get all available models to find endpoints
        model_endpoint_pairs = self.zenmux_api.get_all_model_endpoint_pairs()
        model_endpoints = {model_id: endpoint for model_id, model, endpoint in model_endpoint_pairs}

        # Load dataset questions
        from .dataset import HLEDataset
        dataset = HLEDataset(self.config.hle.dataset_name, self.config.hle.dataset_split)

        # Create a mapping of question_id to question
        all_questions = dataset.get_questions()
        question_map = {q["id"]: q for q in all_questions}

        fixed_models = []
        still_failed_models = []

        for model_identifier, failed_question_ids in evaluation_failures.items():
            self.logger.info(f"\n🔧 Fixing {len(failed_question_ids)} failures for {model_identifier}")

            # Find the endpoint for this model
            if model_identifier not in model_endpoints:
                self.logger.error(f"❌ Endpoint not found for {model_identifier}")
                still_failed_models.append(model_identifier)
                continue

            endpoint = model_endpoints[model_identifier]
            model_name = model_identifier.split(':')[0]  # Extract model name from identifier

            # Load existing predictions file
            safe_model_name = model_identifier.replace("/", "_").replace(":", "_")
            batch_timestamp = os.path.basename(timestamp_dir)
            predictions_file = os.path.join(timestamp_dir, "predictions", f"hle_{safe_model_name}_{batch_timestamp}.json")

            if not os.path.exists(predictions_file):
                self.logger.error(f"❌ Predictions file not found: {predictions_file}")
                still_failed_models.append(model_identifier)
                continue

            # Load existing predictions
            with open(predictions_file, "r") as f:
                predictions_data = json.load(f)

            if "predictions" in predictions_data:
                predictions = predictions_data["predictions"]
                metadata = predictions_data["evaluation_metadata"]
            else:
                predictions = predictions_data
                metadata = {}

            # Try to fix each failed question
            fixed_questions = []
            remaining_failed = []

            for question_id in failed_question_ids:
                if question_id not in question_map:
                    self.logger.warning(f"⚠️ Question {question_id} not found in dataset")
                    remaining_failed.append(question_id)
                    continue

                question = question_map[question_id]
                self.logger.debug(f"🔄 Re-evaluating question {question_id}")

                try:
                    result = await self.evaluator.evaluate_single_question(question, model_name, endpoint)
                    if result:
                        question_id_result, content, usage, performance_metrics, generation_id = result

                        # Add to predictions
                        predictions[question_id] = {
                            "model": model_identifier,
                            "response": content,
                            "usage": usage,
                            "performance": performance_metrics,
                            "generation_id": generation_id
                        }

                        fixed_questions.append(question_id)
                        self.logger.info(f"✅ Fixed question {question_id}")
                    else:
                        remaining_failed.append(question_id)
                        self.logger.warning(f"❌ Still failed question {question_id}")

                except Exception as e:
                    self.logger.error(f"❌ Error re-evaluating question {question_id}: {e}")
                    remaining_failed.append(question_id)

            # Update predictions file
            if fixed_questions:
                # Update metadata statistics if present
                if metadata and "statistics" in metadata:
                    metadata["statistics"]["total_predictions"] = len(predictions)
                    metadata["statistics"]["remaining_questions"] = metadata["statistics"]["total_questions"] - len(predictions)

                # Save updated predictions
                if "predictions" in predictions_data:
                    predictions_data["predictions"] = predictions
                    predictions_data["evaluation_metadata"] = metadata
                else:
                    predictions_data = predictions

                with open(predictions_file, "w") as f:
                    json.dump(predictions_data, f, indent=4)

                self.logger.info(f"💾 Updated predictions file with {len(fixed_questions)} fixed questions")

            # Update failure tracking
            if remaining_failed:
                evaluation_failures[model_identifier] = remaining_failed
                still_failed_models.append(model_identifier)
                self.logger.warning(f"⚠️ {len(remaining_failed)} questions still failed for {model_identifier}")
            else:
                del evaluation_failures[model_identifier]
                fixed_models.append(model_identifier)
                self.logger.info(f"🎉 All failures fixed for {model_identifier}")

        # Update evaluation failures file
        failures_data["evaluation_failures"] = evaluation_failures
        failures_data["failure_metadata"]["total_models_with_evaluation_failures"] = len(evaluation_failures)

        with open(eval_failures_file, "w") as f:
            json.dump(failures_data, f, indent=4)

        self.logger.info(f"\n📊 Fix Summary:")
        self.logger.info(f"✅ Fixed models: {len(fixed_models)}")
        self.logger.info(f"❌ Still failed models: {len(still_failed_models)}")

        return {
            "fixed_models": fixed_models,
            "still_failed_models": still_failed_models,
            "fixed_count": len(fixed_models),
            "remaining_failures": len(still_failed_models)
        }

    async def fix_judge_failures(self, timestamp_dir: str) -> Dict[str, Any]:
        """Fix judge failures by re-judging failed questions."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("⚖️ FIXING JUDGE FAILURES")
        self.logger.info(f"{'='*60}")

        # Load judge failures file
        judge_failures_file = os.path.join(timestamp_dir, f"judge_failures_{os.path.basename(timestamp_dir)}.json")
        if not os.path.exists(judge_failures_file):
            self.logger.error(f"❌ Judge failures file not found: {judge_failures_file}")
            return {"error": "Judge failures file not found"}

        with open(judge_failures_file, "r") as f:
            failures_data = json.load(f)

        judge_failures = failures_data.get("judge_failures", {})
        if not judge_failures:
            self.logger.info("✅ No judge failures to fix")
            return {
                "fixed_models": [],
                "still_failed_models": [],
                "fixed_count": 0,
                "remaining_failures": 0
            }

        self.logger.info(f"🔍 Found {len(judge_failures)} models with judge failures")

        # Load dataset questions
        from .dataset import HLEDataset
        dataset = HLEDataset(self.config.hle.dataset_name, self.config.hle.dataset_split)

        # Create a mapping of question_id to question
        all_questions = dataset.get_questions()
        question_map = {q["id"]: q for q in all_questions}

        fixed_models = []
        still_failed_models = []

        for model_identifier, failed_question_ids in judge_failures.items():
            self.logger.info(f"\n⚖️ Fixing {len(failed_question_ids)} judge failures for {model_identifier}")

            # Load existing judged file
            safe_model_name = model_identifier.replace("/", "_").replace(":", "_")
            batch_timestamp = os.path.basename(timestamp_dir)
            judged_file = os.path.join(timestamp_dir, "judged", f"judged_hle_{safe_model_name}_{batch_timestamp}.json")

            if not os.path.exists(judged_file):
                self.logger.error(f"❌ Judged file not found: {judged_file}")
                still_failed_models.append(model_identifier)
                continue

            # Load existing judged data
            with open(judged_file, "r") as f:
                judged_data = json.load(f)

            if "judged_predictions" in judged_data:
                judged_predictions = judged_data["judged_predictions"]
                metadata = judged_data["judging_metadata"]
            else:
                judged_predictions = judged_data
                metadata = {}

            # Try to fix each failed question
            fixed_questions = []
            remaining_failed = []

            for question_id in failed_question_ids:
                if question_id not in question_map:
                    self.logger.warning(f"⚠️ Question {question_id} not found in dataset")
                    remaining_failed.append(question_id)
                    continue

                if question_id not in judged_predictions:
                    self.logger.warning(f"⚠️ Question {question_id} not found in predictions")
                    remaining_failed.append(question_id)
                    continue

                question = question_map[question_id]
                self.logger.debug(f"🔄 Re-judging question {question_id}")

                try:
                    result = await self.judge.judge_single_response(question, {question_id: judged_predictions[question_id]})
                    if result and result[0]:  # Check if we got a valid result
                        unique_id, updated_prediction, performance_metrics = result

                        # Update the judged predictions
                        judged_predictions[question_id] = updated_prediction

                        fixed_questions.append(question_id)
                        self.logger.info(f"✅ Fixed judge for question {question_id}")
                    else:
                        remaining_failed.append(question_id)
                        self.logger.warning(f"❌ Still failed to judge question {question_id}")

                except Exception as e:
                    self.logger.error(f"❌ Error re-judging question {question_id}: {e}")
                    remaining_failed.append(question_id)

            # Update judged file
            if fixed_questions:
                # Update metadata statistics if present
                if metadata and "statistics" in metadata:
                    # Count successful judgments
                    successful_judgments = sum(1 for pred in judged_predictions.values() if "judge_response" in pred)
                    metadata["statistics"]["total_judged"] = successful_judgments

                # Save updated judged data
                if "judged_predictions" in judged_data:
                    judged_data["judged_predictions"] = judged_predictions
                    judged_data["judging_metadata"] = metadata
                else:
                    judged_data = judged_predictions

                with open(judged_file, "w") as f:
                    json.dump(judged_data, f, indent=4)

                self.logger.info(f"💾 Updated judged file with {len(fixed_questions)} fixed judgments")

            # Update failure tracking
            if remaining_failed:
                judge_failures[model_identifier] = remaining_failed
                still_failed_models.append(model_identifier)
                self.logger.warning(f"⚠️ {len(remaining_failed)} judge failures still remain for {model_identifier}")
            else:
                del judge_failures[model_identifier]
                fixed_models.append(model_identifier)
                self.logger.info(f"🎉 All judge failures fixed for {model_identifier}")

        # Update judge failures file
        failures_data["judge_failures"] = judge_failures
        failures_data["failure_metadata"]["total_models_with_judge_failures"] = len(judge_failures)

        with open(judge_failures_file, "w") as f:
            json.dump(failures_data, f, indent=4)

        self.logger.info(f"\n📊 Fix Summary:")
        self.logger.info(f"✅ Fixed models: {len(fixed_models)}")
        self.logger.info(f"❌ Still failed models: {len(still_failed_models)}")

        return {
            "fixed_models": fixed_models,
            "still_failed_models": still_failed_models,
            "fixed_count": len(fixed_models),
            "remaining_failures": len(still_failed_models)
        }

    def run_metrics_only(self, timestamp_dir: str) -> Dict[str, Any]:
        """Run metrics calculation only for models that have complete and successful evaluations."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("📊 RUNNING METRICS CALCULATION ONLY")
        self.logger.info(f"{'='*60}")

        # Load question IDs file to get total expected questions
        batch_timestamp = os.path.basename(timestamp_dir)
        question_ids_file = os.path.join(timestamp_dir, f"question_ids_{batch_timestamp}.json")

        if not os.path.exists(question_ids_file):
            self.logger.error(f"❌ Question IDs file not found: {question_ids_file}")
            return {"error": "Question IDs file not found"}

        with open(question_ids_file, "r") as f:
            question_ids_data = json.load(f)

        expected_question_count = len(question_ids_data.get("question_ids", []))
        self.logger.info(f"📊 Expected questions count: {expected_question_count}")

        # Load failure files to identify models with failures
        eval_failures_file = os.path.join(timestamp_dir, f"evaluation_failures_{batch_timestamp}.json")
        judge_failures_file = os.path.join(timestamp_dir, f"judge_failures_{batch_timestamp}.json")

        models_with_eval_failures = set()
        models_with_judge_failures = set()

        if os.path.exists(eval_failures_file):
            with open(eval_failures_file, "r") as f:
                eval_failures_data = json.load(f)
                models_with_eval_failures = set(eval_failures_data.get("evaluation_failures", {}).keys())

        if os.path.exists(judge_failures_file):
            with open(judge_failures_file, "r") as f:
                judge_failures_data = json.load(f)
                models_with_judge_failures = set(judge_failures_data.get("judge_failures", {}).keys())

        self.logger.info(f"🚫 Models with evaluation failures: {len(models_with_eval_failures)}")
        self.logger.info(f"⚖️ Models with judge failures: {len(models_with_judge_failures)}")

        # Find all judged files in the timestamp directory
        judged_dir = os.path.join(timestamp_dir, "judged")
        if not os.path.exists(judged_dir):
            self.logger.error(f"❌ Judged directory not found: {judged_dir}")
            return {"error": "Judged directory not found"}

        judged_files = [f for f in os.listdir(judged_dir) if f.startswith("judged_") and f.endswith(".json")]
        self.logger.info(f"📁 Found {len(judged_files)} judged files")

        # Analyze each judged file for completeness
        valid_models = []
        excluded_models = []

        for judged_file in judged_files:
            # Extract model identifier from filename
            # Format: judged_hle_MODEL_PROVIDER_TIMESTAMP.json
            filename_parts = judged_file.replace("judged_hle_", "").replace(f"_{batch_timestamp}.json", "")
            # Reconstruct model identifier: replace last underscore with colon
            parts = filename_parts.split("_")
            if len(parts) >= 2:
                model_identifier = "_".join(parts[:-1]).replace("_", "/") + ":" + parts[-1]
            else:
                self.logger.warning(f"⚠️ Could not parse model identifier from {judged_file}")
                continue

            # Check if model has failures
            if model_identifier in models_with_eval_failures:
                excluded_models.append({
                    "model_identifier": model_identifier,
                    "reason": "evaluation_failures",
                    "file": judged_file
                })
                continue

            if model_identifier in models_with_judge_failures:
                excluded_models.append({
                    "model_identifier": model_identifier,
                    "reason": "judge_failures",
                    "file": judged_file
                })
                continue

            # Load and analyze judged file
            judged_file_path = os.path.join(judged_dir, judged_file)
            try:
                with open(judged_file_path, "r") as f:
                    judged_data = json.load(f)

                if "judged_predictions" in judged_data:
                    judged_predictions = judged_data["judged_predictions"]
                else:
                    judged_predictions = judged_data

                # Count successful judgments (those with judge_response)
                successful_judgments = 0
                for prediction in judged_predictions.values():
                    if "judge_response" in prediction:
                        successful_judgments += 1

                # Check if model has complete judgments
                if successful_judgments == expected_question_count:
                    # Calculate metrics for this model
                    metrics = self.judge.calculate_metrics(judged_predictions, expected_question_count)

                    valid_models.append({
                        "model_identifier": model_identifier,
                        "judged_file": judged_file_path,
                        "metrics": metrics,
                        "successful_judgments": successful_judgments,
                        "expected_questions": expected_question_count
                    })
                    self.logger.info(f"✅ {model_identifier}: {successful_judgments}/{expected_question_count} judgments - metrics calculated")
                else:
                    excluded_models.append({
                        "model_identifier": model_identifier,
                        "reason": "incomplete_judgments",
                        "file": judged_file,
                        "successful_judgments": successful_judgments,
                        "expected_questions": expected_question_count
                    })
                    self.logger.warning(f"❌ {model_identifier}: {successful_judgments}/{expected_question_count} judgments - incomplete")

            except Exception as e:
                self.logger.error(f"❌ Error processing {judged_file}: {e}")
                excluded_models.append({
                    "model_identifier": model_identifier,
                    "reason": "processing_error",
                    "file": judged_file,
                    "error": str(e)
                })

        # Create metrics summary data
        metrics_summary = {
            "metrics_metadata": {
                "timestamp": datetime.now().isoformat(),
                "batch_timestamp": batch_timestamp,
                "source_directory": timestamp_dir,
                "expected_question_count": expected_question_count,
                "total_judged_files": len(judged_files),
                "valid_models": len(valid_models),
                "excluded_models": len(excluded_models)
            },
            "valid_models": valid_models,
            "excluded_models": excluded_models,
            "exclusion_summary": {
                "evaluation_failures": len([m for m in excluded_models if m["reason"] == "evaluation_failures"]),
                "judge_failures": len([m for m in excluded_models if m["reason"] == "judge_failures"]),
                "incomplete_judgments": len([m for m in excluded_models if m["reason"] == "incomplete_judgments"]),
                "processing_errors": len([m for m in excluded_models if m["reason"] == "processing_error"])
            }
        }

        # Save metrics summary
        metrics_output_file = os.path.join(timestamp_dir, f"metrics_only_{batch_timestamp}.json")
        with open(metrics_output_file, "w") as f:
            json.dump(metrics_summary, f, indent=4)

        self.logger.info(f"\n📊 Metrics Summary:")
        self.logger.info(f"✅ Valid models with complete metrics: {len(valid_models)}")
        self.logger.info(f"❌ Excluded models: {len(excluded_models)}")
        self.logger.info(f"📁 Metrics saved to: {metrics_output_file}")

        # Log detailed exclusion reasons
        if excluded_models:
            self.logger.info(f"\n📋 Exclusion Details:")
            for reason, count in metrics_summary["exclusion_summary"].items():
                if count > 0:
                    self.logger.info(f"  {reason.replace('_', ' ').title()}: {count}")

        return {
            "metrics_file": metrics_output_file,
            "valid_models_count": len(valid_models),
            "excluded_models_count": len(excluded_models),
            "expected_questions": expected_question_count,
            "exclusion_summary": metrics_summary["exclusion_summary"]
        }

    def _validate_model_completeness(self, model_identifier: str, timestamp_dir: str, expected_question_count: int) -> Dict[str, Any]:
        """Validate that a model has complete evaluations and judgments without failures."""
        batch_timestamp = os.path.basename(timestamp_dir)

        # Check evaluation failures
        eval_failures_file = os.path.join(timestamp_dir, f"evaluation_failures_{batch_timestamp}.json")
        if os.path.exists(eval_failures_file):
            with open(eval_failures_file, "r") as f:
                eval_failures_data = json.load(f)
                if model_identifier in eval_failures_data.get("evaluation_failures", {}):
                    return {
                        "valid": False,
                        "reason": "evaluation_failures",
                        "details": "Model has evaluation failures"
                    }

        # Check judge failures
        judge_failures_file = os.path.join(timestamp_dir, f"judge_failures_{batch_timestamp}.json")
        if os.path.exists(judge_failures_file):
            with open(judge_failures_file, "r") as f:
                judge_failures_data = json.load(f)
                if model_identifier in judge_failures_data.get("judge_failures", {}):
                    return {
                        "valid": False,
                        "reason": "judge_failures",
                        "details": "Model has judge failures"
                    }

        # Check judge completeness
        safe_model_name = model_identifier.replace("/", "_").replace(":", "_")
        judged_file = os.path.join(timestamp_dir, "judged", f"judged_hle_{safe_model_name}_{batch_timestamp}.json")

        if not os.path.exists(judged_file):
            return {
                "valid": False,
                "reason": "no_judged_file",
                "details": f"Judged file not found: {judged_file}"
            }

        try:
            with open(judged_file, "r") as f:
                judged_data = json.load(f)

            if "judged_predictions" in judged_data:
                judged_predictions = judged_data["judged_predictions"]
            else:
                judged_predictions = judged_data

            # Count successful judgments
            successful_judgments = sum(1 for pred in judged_predictions.values() if "judge_response" in pred)

            if successful_judgments == expected_question_count:
                return {
                    "valid": True,
                    "successful_judgments": successful_judgments,
                    "expected_questions": expected_question_count
                }
            else:
                return {
                    "valid": False,
                    "reason": "incomplete_judgments",
                    "details": f"Only {successful_judgments}/{expected_question_count} judgments completed"
                }

        except Exception as e:
            return {
                "valid": False,
                "reason": "processing_error",
                "details": f"Error processing judged file: {e}"
            }

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