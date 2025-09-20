"""HLE evaluation runner with ZenMux integration."""

import asyncio
import json
import os
import glob
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

    def save_available_models(self, model_endpoint_pairs: List, text_only: bool = False, model_filter: Optional[str] = None, exclude_models: Optional[List[str]] = None, exclude_providers: Optional[List[str]] = None) -> str:
        """Save the available models list to a separate file."""
        # Create models data structure
        models_data = {
            "discovery_metadata": {
                "timestamp": datetime.now().isoformat(),
                "batch_timestamp": self.batch_timestamp,
                "text_only": text_only,
                "model_filter": model_filter,
                "exclude_models": exclude_models,
                "exclude_providers": exclude_providers,
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
        exclude_models: Optional[List[str]] = None,
        exclude_providers: Optional[List[str]] = None
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

        # Apply model exclusions if specified
        if exclude_models:
            excluded_count = 0
            original_count = len(model_endpoint_pairs)

            exclude_model_slugs = set(exclude_models)
            filtered_pairs = []

            for model_id, model, endpoint in model_endpoint_pairs:
                # Extract model slug and provider from model_id (format: "vendor/model:provider")
                parts = model_id.split(':')
                model_slug = parts[0]  # Get "vendor/model" part
                provider = parts[1] if len(parts) > 1 else ""  # Get provider part

                should_exclude = False
                for exclude_slug in exclude_model_slugs:
                    exclude_lower = exclude_slug.lower()
                    model_lower = model_slug.lower()

                    # Case 1: Full model ID match (e.g., "openai/gpt-4o:openai")
                    if ':' in exclude_slug and exclude_lower == model_id.lower():
                        should_exclude = True
                        break

                    # Case 2: Exact model slug match (e.g., "openai/gpt-4o")
                    elif exclude_lower == model_lower:
                        should_exclude = True
                        break

                    # Case 3: Vendor-only exclusion (e.g., "anthropic" matches "anthropic/*")
                    elif '/' not in exclude_slug and model_lower.startswith(exclude_lower + '/'):
                        should_exclude = True
                        break

                if not should_exclude:
                    filtered_pairs.append((model_id, model, endpoint))
                else:
                    excluded_count += 1

            model_endpoint_pairs = filtered_pairs

            if excluded_count > 0:
                self.logger.info(f"🚫 Excluded {excluded_count} model endpoints based on model patterns")
                self.logger.info(f"📉 Remaining models: {len(model_endpoint_pairs)} (was {original_count})")

        # Apply provider exclusions if specified
        if exclude_providers:
            excluded_count = 0
            original_count = len(model_endpoint_pairs)

            exclude_provider_slugs = set(p.lower() for p in exclude_providers)
            filtered_pairs = []

            for model_id, model, endpoint in model_endpoint_pairs:
                # Extract provider from model_id (format: "vendor/model:provider")
                parts = model_id.split(':')
                provider = parts[1].lower() if len(parts) > 1 else ""

                if provider not in exclude_provider_slugs:
                    filtered_pairs.append((model_id, model, endpoint))
                else:
                    excluded_count += 1

            model_endpoint_pairs = filtered_pairs

            if excluded_count > 0:
                self.logger.info(f"🚫 Excluded {excluded_count} model endpoints based on provider patterns")
                self.logger.info(f"📉 Remaining models: {len(model_endpoint_pairs)} (was {original_count})")

        self.logger.info(f"🎯 Total model endpoints to evaluate: {len(model_endpoint_pairs)}")

        # Save available models list to file
        self.save_available_models(model_endpoint_pairs, text_only=text_only, model_filter=model_filter, exclude_models=exclude_models, exclude_providers=exclude_providers)

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
                    error_msg = str(e)
                    if "Too many open files" in error_msg:
                        self.logger.error(f"❌ File handle limit reached for {model_identifier}: {error_msg}")
                    else:
                        self.logger.error(f"❌ Error evaluating {model_identifier}: {e}")

                    # Clean up any connections for this model to free resources
                    try:
                        await self.evaluator.zenmux_client.close()
                    except:
                        pass

                    return {
                        "model_identifier": model_identifier,
                        "error": str(e),
                        "predictions_file": None,
                        "judged_file": None,
                        "metrics": None
                    }

                finally:
                    # Always try to clean up connections after each model
                    try:
                        await self.evaluator.zenmux_client.close()
                        # Also clean up judge client connections
                        await self.judge.zenmux_client.close()
                    except Exception as cleanup_error:
                        self.logger.warning(f"⚠️ Warning: Failed to cleanup connections for {model_identifier}: {cleanup_error}")
                        pass

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
        """Validate that evaluation is complete by checking if all responses are non-empty."""
        try:
            with open(predictions_file, "r") as f:
                data = json.load(f)

            # Get predictions and calculate completeness directly
            predictions = data.get("predictions", data)
            if not predictions:
                self.logger.warning(f"⚠️ Warning: No predictions found in file")
                return False

            total_questions = len(predictions)
            successful_predictions = sum(1 for pred in predictions.values() if pred.get("response", "").strip())

            if successful_predictions == total_questions:
                self.logger.info(f"✅ Evaluation complete: {successful_predictions}/{total_questions} predictions")
                return True
            else:
                missing_count = total_questions - successful_predictions
                self.logger.warning(f"❌ Evaluation incomplete: {successful_predictions}/{total_questions} predictions ({missing_count} missing)")
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

        # No longer need separate failure files - response content indicates success/failure

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

            # Check for evaluation failures (empty responses)
            if result.get("predictions_file"):
                if not self._has_complete_evaluations(result["predictions_file"]):
                    models_with_failures.add(model_identifier)

            # Check for judge failures (empty judge responses)
            if result.get("judged_file"):
                if not self._has_complete_judgments(result["judged_file"]):
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

                        # Count successful judgments (non-empty judge responses)
                        successful_judgments = sum(1 for pred in judged_predictions.values()
                                                 if pred.get("judge_response", {}).get("reasoning", "").strip())

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

        # Create simplified summary data structure
        summary = {
            "summary_metadata": {
                "timestamp": datetime.now().isoformat(),
                "run_metadata": run_metadata
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
                "exclusion_reason": self._get_exclusion_reason(result) if is_excluded else None
            }

            # No longer collect per-model statistics in summary - use dedicated statistics files instead

            summary["model_results"].append(model_summary)

        # Save summary file in the timestamped run directory
        summary_file = os.path.join(self.config.run_dir, f"metrics_summary_{self.batch_timestamp}.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)

        self.logger.info(f"📊 Enhanced metrics summary saved to: {summary_file}")
        self.logger.info(f"🔍 Models excluded from metrics: {len(models_with_failures)}/{len(results)}")

        # Generate comprehensive statistics files
        try:
            from .statistics import generate_evaluation_statistics, generate_judge_statistics, generate_metrics_statistics

            # Generate evaluation statistics
            eval_stats_file = generate_evaluation_statistics(self.config.run_dir)
            self.logger.info(f"📈 Evaluation statistics saved to: {eval_stats_file}")

            # Generate judge statistics
            judge_stats_file = generate_judge_statistics(self.config.run_dir)
            self.logger.info(f"⚖️ Judge statistics saved to: {judge_stats_file}")

            # Generate metrics statistics
            metrics_stats_file = generate_metrics_statistics(self.config.run_dir)
            self.logger.info(f"📋 Metrics statistics saved to: {metrics_stats_file}")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate statistics files: {e}")

        return summary_file

    def _get_exclusion_reason(self, result: Dict[str, Any]) -> str:
        """Get the reason why a model was excluded from metrics calculation."""
        reasons = []

        if result.get("error"):
            reasons.append("general_error")

        if result.get("predictions_file"):
            if not self._has_complete_evaluations(result["predictions_file"]):
                reasons.append("incomplete_evaluations")

        if result.get("judged_file"):
            if not self._has_complete_judgments(result["judged_file"]):
                reasons.append("incomplete_judgments")

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

    def _has_complete_evaluations(self, predictions_file: str) -> bool:
        """Check if all predictions have non-empty responses."""
        try:
            with open(predictions_file, "r") as f:
                data = json.load(f)

            predictions = data.get("predictions", data)
            if not predictions:
                return False

            # Check if all predictions have non-empty responses
            for pred in predictions.values():
                if not pred.get("response", "").strip():
                    return False
            return True
        except Exception:
            return False

    def _has_complete_judgments(self, judged_file: str) -> bool:
        """Check if all judge results have non-empty responses."""
        try:
            with open(judged_file, "r") as f:
                data = json.load(f)

            judged_predictions = data.get("judged_predictions", data)
            if not judged_predictions:
                return False

            # Check if all predictions that have responses also have valid judge responses
            for pred in judged_predictions.values():
                # Only check judge completeness for predictions that have responses
                if pred.get("response", "").strip():
                    if not pred.get("judge_response", {}).get("reasoning", "").strip():
                        return False
            return True
        except Exception:
            return False

    async def fix_models(self, timestamp_dir: str) -> Dict[str, Any]:
        """Fix evaluation and judge failures by re-processing questions with empty responses."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🔧 FIXING EVALUATION AND JUDGE FAILURES")
        self.logger.info(f"{'='*60}")

        batch_timestamp = os.path.basename(timestamp_dir)

        # Find all prediction files in the timestamp directory
        predictions_dir = os.path.join(timestamp_dir, "predictions")
        if not os.path.exists(predictions_dir):
            self.logger.error(f"❌ Predictions directory not found: {predictions_dir}")
            return {"error": "Predictions directory not found"}

        prediction_files = glob.glob(os.path.join(predictions_dir, "hle_*.json"))
        if not prediction_files:
            self.logger.info("✅ No prediction files found to fix")
            return {
                "fixed_models": [],
                "still_failed_models": [],
                "fixed_count": 0,
                "remaining_failures": 0
            }

        self.logger.info(f"🔍 Found {len(prediction_files)} prediction files to check for failures")

        # Load dataset questions
        from .dataset import HLEDataset
        dataset = HLEDataset(self.config.hle.dataset_name, self.config.hle.dataset_split)
        all_questions = dataset.get_questions()
        question_map = {q["id"]: q for q in all_questions}

        fixed_models = []
        still_failed_models = []

        for predictions_file in prediction_files:
            try:
                # Load evaluation file to get model info and endpoint from metadata
                with open(predictions_file, "r") as f:
                    predictions_data = json.load(f)

                # Get model info from evaluation metadata
                evaluation_metadata = predictions_data.get("evaluation_metadata", {})
                model_identifier = evaluation_metadata.get("model_identifier")
                endpoint_data = evaluation_metadata.get("endpoint", {})

                if not model_identifier:
                    self.logger.warning(f"⚠️ No model identifier found in {os.path.basename(predictions_file)}")
                    continue

                if not endpoint_data:
                    self.logger.warning(f"⚠️ No endpoint data found in {os.path.basename(predictions_file)}")
                    continue

                self.logger.info(f"\n🔧 Checking {model_identifier} for failures")

                # Reconstruct endpoint object from metadata
                from zenmux.models import ZenMuxEndpoint
                endpoint = ZenMuxEndpoint(
                    pricing_completion="0",
                    pricing_prompt="0",
                    context_length=endpoint_data.get("context_length", 200000),
                    provider=endpoint_data.get("provider", "unknown"),
                    provider_slug=endpoint_data.get("provider_slug", "unknown"),
                    max_completion_tokens=endpoint_data.get("max_completion_tokens", 4096),
                    supports_streaming=endpoint_data.get("supports_streaming", True),
                    supports_reasoning=False,
                    supports_tool_parameters=True,
                    supported_parameters=[],
                    can_abort=True,
                    visible=1,
                    suitable_api=endpoint_data.get("suitable_api", "chat.completions")
                )

                model_name = model_identifier.split(':')[0]
                safe_model_name = model_identifier.replace("/", "_").replace(":", "_")
                judged_file = os.path.join(timestamp_dir, "judged", f"judged_hle_{safe_model_name}_{batch_timestamp}.json")

            except Exception as e:
                self.logger.error(f"❌ Error processing {os.path.basename(predictions_file)}: {e}")
                still_failed_models.append(f"ERROR_{os.path.basename(predictions_file)}")
                continue

            model_fixed = False

            # Fix evaluation failures (empty responses)
            if os.path.exists(predictions_file):
                with open(predictions_file, "r") as f:
                    predictions_data = json.load(f)

                predictions = predictions_data.get("predictions", predictions_data)
                evaluation_metadata = predictions_data.get("evaluation_metadata", {})

                # Find questions with empty responses
                failed_eval_questions = [
                    qid for qid, pred in predictions.items()
                    if not pred.get("response", "").strip()
                ]

                if failed_eval_questions:
                    self.logger.info(f"🔄 Re-evaluating {len(failed_eval_questions)} failed questions")

                    for question_id in failed_eval_questions:
                        if question_id in question_map:
                            question = question_map[question_id]
                            try:
                                question_id_result, result = await self.evaluator.evaluate_single_question(
                                    question, model_name, endpoint
                                )
                                predictions[question_id] = result
                                if result.get("response", "").strip():
                                    model_fixed = True
                                    self.logger.debug(f"✅ Fixed evaluation for question {question_id}")
                                else:
                                    self.logger.debug(f"❌ Still failed evaluation for question {question_id}")
                            except Exception as e:
                                self.logger.error(f"❌ Error re-evaluating question {question_id}: {e}")

                    # Save updated predictions
                    final_predictions_data = {
                        "evaluation_metadata": evaluation_metadata,
                        "predictions": predictions
                    } if evaluation_metadata else predictions

                    with open(predictions_file, "w") as f:
                        json.dump(final_predictions_data, f, indent=4)

            # Fix judge failures (empty judge responses)
            if os.path.exists(judged_file) and os.path.exists(predictions_file):
                with open(judged_file, "r") as f:
                    judged_data = json.load(f)

                judged_predictions = judged_data.get("judged_predictions", judged_data)
                judging_metadata = judged_data.get("judging_metadata", {})

                # Find questions that need judging (have responses but empty judge responses)
                failed_judge_questions = [
                    qid for qid, pred in judged_predictions.items()
                    if pred.get("response", "").strip() and not pred.get("judge_response", {}).get("reasoning", "").strip()
                ]

                if failed_judge_questions:
                    self.logger.info(f"⚖️ Re-judging {len(failed_judge_questions)} failed judgments")

                    for question_id in failed_judge_questions:
                        if question_id in question_map:
                            question = question_map[question_id]
                            try:
                                unique_id, updated_prediction, performance_metrics = await self.judge.judge_single_response(
                                    question, {question_id: judged_predictions[question_id]}
                                )
                                judged_predictions[question_id] = updated_prediction
                                if updated_prediction.get("judge_response", {}).get("reasoning", "").strip():
                                    model_fixed = True
                                    self.logger.debug(f"✅ Fixed judgment for question {question_id}")
                                else:
                                    self.logger.debug(f"❌ Still failed judgment for question {question_id}")
                            except Exception as e:
                                self.logger.error(f"❌ Error re-judging question {question_id}: {e}")

                    # Save updated judged data
                    final_judged_data = {
                        "judging_metadata": judging_metadata,
                        "judged_predictions": judged_predictions
                    } if judging_metadata else judged_predictions

                    with open(judged_file, "w") as f:
                        json.dump(final_judged_data, f, indent=4)

            if model_fixed:
                fixed_models.append(model_identifier)
                self.logger.info(f"🎉 Fixed some failures for {model_identifier}")
            else:
                still_failed_models.append(model_identifier)
                self.logger.info(f"⚠️ No fixes needed or all fixes failed for {model_identifier}")

        # Run metrics calculation for all models
        self.logger.info(f"\n📊 Running metrics calculation")
        metrics_result = self.run_metrics_only(timestamp_dir)

        self.logger.info(f"\n📊 Fix Summary:")
        self.logger.info(f"✅ Fixed models: {len(fixed_models)}")
        self.logger.info(f"❌ Still failed models: {len(still_failed_models)}")

        return {
            "fixed_models": fixed_models,
            "still_failed_models": still_failed_models,
            "fixed_count": len(fixed_models),
            "remaining_failures": len(still_failed_models),
            "metrics_file": metrics_result.get("metrics_file", "")
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

        # No longer need to load separate failure files - we check response content directly

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

            # Load and analyze judged file
            judged_file_path = os.path.join(judged_dir, judged_file)

            # Check if model has complete evaluations and judgments using the new fields
            predictions_file = os.path.join(timestamp_dir, "predictions", f"hle_{filename_parts}_{batch_timestamp}.json")

            # Check evaluation completeness
            if os.path.exists(predictions_file) and not self._has_complete_evaluations(predictions_file):
                excluded_models.append({
                    "model_identifier": model_identifier,
                    "reason": "incomplete_evaluations",
                    "file": judged_file
                })
                continue

            # Check judgment completeness
            if not self._has_complete_judgments(judged_file_path):
                excluded_models.append({
                    "model_identifier": model_identifier,
                    "reason": "incomplete_judgments",
                    "file": judged_file
                })
                continue
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
                "incomplete_evaluations": len([m for m in excluded_models if m["reason"] == "incomplete_evaluations"]),
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
        """Validate that a model has complete evaluations and judgments using response content."""
        batch_timestamp = os.path.basename(timestamp_dir)

        # Check evaluation completeness using response content
        safe_model_name = model_identifier.replace("/", "_").replace(":", "_")
        predictions_file = os.path.join(timestamp_dir, "predictions", f"hle_{safe_model_name}_{batch_timestamp}.json")

        if os.path.exists(predictions_file) and not self._has_complete_evaluations(predictions_file):
            return {
                "valid": False,
                "reason": "incomplete_evaluations",
                "details": "Model has incomplete evaluations"
            }

        # Check judge completeness using judge response content
        judged_file = os.path.join(timestamp_dir, "judged", f"judged_hle_{safe_model_name}_{batch_timestamp}.json")

        if not os.path.exists(judged_file):
            return {
                "valid": False,
                "reason": "no_judged_file",
                "details": f"Judged file not found: {judged_file}"
            }

        if not self._has_complete_judgments(judged_file):
            return {
                "valid": False,
                "reason": "incomplete_judgments",
                "details": "Model has incomplete judgments"
            }

        try:
            with open(judged_file, "r") as f:
                judged_data = json.load(f)

            if "judged_predictions" in judged_data:
                judged_predictions = judged_data["judged_predictions"]
            else:
                judged_predictions = judged_data

            # Count successful judgments using judge response content
            successful_judgments = sum(1 for pred in judged_predictions.values()
                                     if pred.get("judge_response", {}).get("reasoning", "").strip())

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
                self.logger.info(f"📊 Accuracy: {metrics['accuracy']}% +/- {metrics['confidence_interval']}%")
                self.logger.info(f"📏 Calibration Error: {metrics['calibration_error']}")

        # File locations
        self.logger.info(f"\n📁 FILES:")
        self.logger.info(f"📁 Run directory: {self.config.run_dir}")
        self.logger.info(f"📁 Prediction files: {self.config.get_predictions_dir()}")
        self.logger.info(f"📁 Judged files: {self.config.get_judged_dir()}")