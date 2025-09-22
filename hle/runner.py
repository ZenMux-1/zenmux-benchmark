"""HLE evaluation runner with ZenMux integration."""

import asyncio
import json
import os
import glob
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

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

            # Calculate metrics in real-time from judged file (REFACTORED: no longer cached)
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
        """REFACTORED: Calculate metrics in real-time from judged file instead of reading cached values."""
        try:
            with open(judged_file, "r") as f:
                data = json.load(f)

            # Get judged predictions (original data)
            judged_predictions = data.get("judged_predictions", data)
            if not judged_predictions:
                self.logger.warning(f"⚠️ No judged_predictions found in {judged_file}")
                return None

            # Get total questions count - first try from question_ids file
            total_questions = len(judged_predictions)  # fallback
            try:
                question_ids_file = os.path.join(self.config.run_dir, f"question_ids_{self.batch_timestamp}.json")
                if os.path.exists(question_ids_file):
                    with open(question_ids_file, "r") as f:
                        question_ids_data = json.load(f)
                        total_questions = len(question_ids_data.get("question_ids", []))
            except Exception as e:
                self.logger.debug(f"Could not load question IDs for metrics calculation: {e}")

            # Calculate metrics in real-time using judge's calculate_metrics method
            metrics = self.judge.calculate_metrics(judged_predictions, total_questions)

            return metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Warning: Could not calculate metrics from {judged_file}: {e}")
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
        """Fix evaluation and judge failures by re-processing questions with empty responses using concurrent processing."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🔧 FIXING EVALUATION AND JUDGE FAILURES (CONCURRENT)")
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
        self.logger.info(f"🔄 Using concurrent processing: max_concurrent_models={self.config.hle.max_concurrent_models}, num_workers={self.config.hle.num_workers}")

        # Load dataset questions
        from .dataset import HLEDataset
        dataset = HLEDataset(self.config.hle.dataset_name, self.config.hle.dataset_split)
        all_questions = dataset.get_questions()
        question_map = {q["id"]: q for q in all_questions}

        # Concurrent processing semaphore for outer level (models)
        models_semaphore = asyncio.Semaphore(self.config.hle.max_concurrent_models)

        async def fix_single_model(predictions_file: str) -> Tuple[str, bool]:
            """Fix a single model with concurrent processing of failed questions."""
            async with models_semaphore:
                return await self._fix_single_model_internal(predictions_file, timestamp_dir, batch_timestamp, question_map)

        # Process all models concurrently with progress tracking
        self.logger.info(f"🚀 Starting concurrent fix processing for {len(prediction_files)} models")
        self.logger.info(f"📊 Progress will be reported for individual model fixes...")

        fix_tasks = [fix_single_model(pf) for pf in prediction_files]
        fix_results = await asyncio.gather(*fix_tasks, return_exceptions=True)

        # Process results and categorize models more precisely
        fixed_models = []
        models_with_no_failures = []
        still_failed_evaluation_models = []
        still_failed_judge_models = []
        processing_error_models = []

        for i, result in enumerate(fix_results):
            if isinstance(result, Exception):
                model_file = os.path.basename(prediction_files[i])
                processing_error_models.append(f"ERROR_{model_file}")
                self.logger.error(f"❌ Error processing {model_file}: {result}")
            else:
                model_identifier, fix_info = result

                # fix_info should now be a dict with details about what was fixed
                if isinstance(fix_info, dict):
                    if fix_info.get("had_failures", False):
                        if fix_info.get("was_fixed", False):
                            fixed_models.append(model_identifier)
                            self.logger.info(f"🎉 Fixed some failures for {model_identifier}")
                        else:
                            # Still has failures - categorize by type
                            if fix_info.get("still_has_eval_failures", False):
                                still_failed_evaluation_models.append(model_identifier)
                            if fix_info.get("still_has_judge_failures", False):
                                still_failed_judge_models.append(model_identifier)

                            if not fix_info.get("still_has_eval_failures", False) and not fix_info.get("still_has_judge_failures", False):
                                # This shouldn't happen, but log for debugging
                                self.logger.warning(f"⚠️ {model_identifier}: Had failures but unclear what type")
                    else:
                        models_with_no_failures.append(model_identifier)
                        self.logger.info(f"✅ No failures found for {model_identifier}")
                else:
                    # Backward compatibility - treat as boolean
                    if fix_info:  # was_fixed
                        fixed_models.append(model_identifier)
                        self.logger.info(f"🎉 Fixed some failures for {model_identifier}")
                    else:
                        # Need to check what type of failures still exist
                        self.logger.warning(f"⚠️ {model_identifier}: No fixes applied (unclear failure type)")

        # Run metrics calculation for all models (consistent with normal benchmark flow)
        self.logger.info(f"\n📊 Running metrics calculation and statistics generation...")
        self.logger.info(f"📁 Analyzing files in: {timestamp_dir}")

        # Build results structure from existing files to match normal benchmark flow
        self.logger.info(f"🔍 Building results structure from existing files...")
        results = self._build_results_from_timestamp_dir(timestamp_dir)
        self.logger.info(f"📋 Found {len(results)} models to process for metrics")

        # Temporarily set run_dir to timestamp directory
        original_run_dir = self.config.run_dir
        original_batch_timestamp = self.batch_timestamp
        self.config.run_dir = timestamp_dir
        self.batch_timestamp = batch_timestamp

        try:
            # Use the same metrics flow as normal benchmark (generates metrics_summary + statistics files)
            self.logger.info(f"📊 Generating comprehensive metrics and statistics files...")
            run_metadata = {
                "mode": "fix",
                "text_only": True,
                "auto_judge": True,
                "num_workers": self.config.hle.num_workers,
                "max_concurrent_models": self.config.hle.max_concurrent_models,
                "model_filter": None,
                "model_slug": None,
                "provider_slug": None
            }
            metrics_summary_file = self.save_metrics_summary(results, run_metadata)
            self.logger.info(f"✅ Metrics calculation completed successfully")
        finally:
            # Restore original configuration
            self.config.run_dir = original_run_dir
            self.batch_timestamp = original_batch_timestamp

        # Calculate totals
        total_still_failed = len(still_failed_evaluation_models) + len(still_failed_judge_models) + len(processing_error_models)
        # Note: A model can have both eval and judge failures, so we need to count unique models
        unique_still_failed_models = set(still_failed_evaluation_models + still_failed_judge_models + processing_error_models)

        self.logger.info(f"\n📊 Fix Summary:")
        self.logger.info(f"✅ Models with fixes applied: {len(fixed_models)}")
        self.logger.info(f"✅ Models with no failures: {len(models_with_no_failures)}")
        self.logger.info(f"❌ Models still with evaluation failures: {len(still_failed_evaluation_models)}")
        if still_failed_evaluation_models:
            self.logger.info(f"   📋 Evaluation failure models:")
            for model in still_failed_evaluation_models:
                self.logger.info(f"     - {model}")
        self.logger.info(f"❌ Models still with judge failures: {len(still_failed_judge_models)}")
        if still_failed_judge_models:
            self.logger.info(f"   📋 Judge failure models:")
            for model in still_failed_judge_models:
                self.logger.info(f"     - {model}")
        self.logger.info(f"❌ Models with processing errors: {len(processing_error_models)}")
        if processing_error_models:
            self.logger.info(f"   📋 Processing error models:")
            for model in processing_error_models:
                self.logger.info(f"     - {model}")
        self.logger.info(f"📊 Total models processed: {len(prediction_files)}")
        self.logger.info(f"📊 Total unique models still with issues: {len(unique_still_failed_models)}")

        return {
            "fixed_models": fixed_models,
            "models_with_no_failures": models_with_no_failures,
            "still_failed_evaluation_models": still_failed_evaluation_models,
            "still_failed_judge_models": still_failed_judge_models,
            "processing_error_models": processing_error_models,
            "fixed_count": len(fixed_models),
            "no_failures_count": len(models_with_no_failures),
            "still_eval_failures_count": len(still_failed_evaluation_models),
            "still_judge_failures_count": len(still_failed_judge_models),
            "processing_errors_count": len(processing_error_models),
            "remaining_failures": len(unique_still_failed_models),
            "metrics_summary_file": metrics_summary_file
        }

    def _build_results_from_timestamp_dir(self, timestamp_dir: str) -> List[Dict[str, Any]]:
        """Build results structure from existing files in timestamp directory."""
        results = []
        batch_timestamp = os.path.basename(timestamp_dir)

        # Find all prediction files
        predictions_dir = os.path.join(timestamp_dir, "predictions")
        judged_dir = os.path.join(timestamp_dir, "judged")

        if not os.path.exists(predictions_dir):
            return results

        prediction_files = glob.glob(os.path.join(predictions_dir, "hle_*.json"))
        self.logger.info(f"🔍 Processing {len(prediction_files)} prediction files for results structure...")

        for i, prediction_file in enumerate(prediction_files, 1):
            if i % 10 == 0 or i == len(prediction_files):  # Progress every 10 files or at the end
                self.logger.info(f"📋 Processing prediction files: {i}/{len(prediction_files)} completed")
            try:
                # Extract model info from filename
                filename = os.path.basename(prediction_file)
                # Format: hle_MODEL_PROVIDER_TIMESTAMP.json
                filename_parts = filename.replace("hle_", "").replace(f"_{batch_timestamp}.json", "")
                parts = filename_parts.split("_")
                if len(parts) >= 2:
                    model_identifier = "_".join(parts[:-1]).replace("_", "/") + ":" + parts[-1]
                else:
                    continue

                # Build result entry
                result = {
                    "model_identifier": model_identifier,
                    "predictions_file": prediction_file,
                    "judged_file": None,
                    "metrics": None,
                    "error": None
                }

                # Check for corresponding judged file
                judged_file = os.path.join(judged_dir, f"judged_{filename}")
                if os.path.exists(judged_file):
                    result["judged_file"] = judged_file
                    # Calculate metrics in real-time (REFACTORED: no longer cached)
                    result["metrics"] = self.extract_metrics_from_judged_file(judged_file)

                results.append(result)

            except Exception as e:
                self.logger.warning(f"⚠️ Could not process prediction file {prediction_file}: {e}")

        self.logger.info(f"✅ Successfully built results structure for {len(results)} models")
        return results

    async def _fix_single_model_internal(self, predictions_file: str, timestamp_dir: str, batch_timestamp: str, question_map: Dict[str, Any]) -> Tuple[str, bool]:
        """Internal method to fix a single model with concurrent question processing."""
        # Load evaluation file to get model info and endpoint from metadata
        with open(predictions_file, "r") as f:
            predictions_data = json.load(f)

        # Get model info from evaluation metadata
        evaluation_metadata = predictions_data.get("evaluation_metadata", {})
        model_identifier = evaluation_metadata.get("model_identifier")
        endpoint_data = evaluation_metadata.get("endpoint", {})

        if not model_identifier:
            self.logger.warning(f"⚠️ No model identifier found in {os.path.basename(predictions_file)}")
            return "UNKNOWN_MODEL", False

        if not endpoint_data:
            self.logger.warning(f"⚠️ No endpoint data found in {os.path.basename(predictions_file)}")
            return model_identifier, False

        self.logger.info(f"\n🔧 Checking {model_identifier} for failures...")

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

        model_fixed = False
        failed_eval_questions = []
        failed_judge_questions = []

        # Fix evaluation failures (empty responses) with concurrent processing
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
                self.logger.info(f"🔄 Re-evaluating {len(failed_eval_questions)} failed questions concurrently (workers: {self.config.hle.num_workers})")

                # Progress tracking
                completed_count = 0
                total_questions = len(failed_eval_questions)
                progress_interval = max(1, total_questions // 10)  # Report every 10% or at least every question

                # Concurrent evaluation processing
                semaphore = asyncio.Semaphore(self.config.hle.num_workers)

                async def fix_single_eval_question(question_id: str):
                    """Fix a single evaluation question."""
                    nonlocal completed_count
                    async with semaphore:
                        if question_id in question_map:
                            question = question_map[question_id]
                            try:
                                question_id_result, result = await self.evaluator.evaluate_single_question(
                                    question, model_name, endpoint
                                )
                                # Update progress
                                completed_count += 1
                                if completed_count % progress_interval == 0 or completed_count == total_questions:
                                    progress_pct = (completed_count / total_questions) * 100
                                    self.logger.info(f"🔄 Evaluation progress: {completed_count}/{total_questions} ({progress_pct:.0f}%) completed")

                                return question_id, result, result.get("response", "").strip() != ""
                            except Exception as e:
                                completed_count += 1
                                if completed_count % progress_interval == 0 or completed_count == total_questions:
                                    progress_pct = (completed_count / total_questions) * 100
                                    self.logger.info(f"🔄 Evaluation progress: {completed_count}/{total_questions} ({progress_pct:.0f}%) completed")
                                self.logger.error(f"❌ Error re-evaluating question {question_id}: {e}")
                                return question_id, None, False

                        completed_count += 1
                        if completed_count % progress_interval == 0 or completed_count == total_questions:
                            progress_pct = (completed_count / total_questions) * 100
                            self.logger.info(f"🔄 Evaluation progress: {completed_count}/{total_questions} ({progress_pct:.0f}%) completed")
                        return question_id, None, False

                # Process failed questions concurrently
                eval_fix_tasks = [fix_single_eval_question(qid) for qid in failed_eval_questions]
                eval_fix_results = await asyncio.gather(*eval_fix_tasks, return_exceptions=True)

                # Process results
                fixed_count = 0
                for result in eval_fix_results:
                    if isinstance(result, Exception):
                        self.logger.error(f"❌ Exception during evaluation fix: {result}")
                        continue

                    question_id, new_result, was_fixed = result
                    if new_result:
                        predictions[question_id] = new_result
                        if was_fixed:
                            fixed_count += 1
                            model_fixed = True
                            self.logger.debug(f"✅ Fixed evaluation for question {question_id}")
                        else:
                            self.logger.debug(f"❌ Still failed evaluation for question {question_id}")

                if fixed_count > 0:
                    self.logger.info(f"✅ Fixed {fixed_count}/{len(failed_eval_questions)} evaluation failures")
                else:
                    self.logger.warning(f"⚠️ Could not fix any evaluation failures (0/{len(failed_eval_questions)})")

                # Save updated predictions
                final_predictions_data = {
                    "evaluation_metadata": evaluation_metadata,
                    "predictions": predictions
                } if evaluation_metadata else predictions

                with open(predictions_file, "w") as f:
                    json.dump(final_predictions_data, f, indent=4)
        else:
            self.logger.info(f"📄 No prediction file found for {model_identifier}, skipping evaluation fixes")

        # Fix judge failures (empty judge responses) with concurrent processing
        if os.path.exists(judged_file):
            with open(judged_file, "r") as f:
                judged_data = json.load(f)

            judged_predictions = judged_data.get("judged_predictions", judged_data)
            judging_metadata = judged_data.get("judging_metadata", {})

            # Get original predictions file from judged metadata
            original_predictions_file = judging_metadata.get("original_predictions_file")
            if not original_predictions_file or not os.path.exists(original_predictions_file):
                self.logger.warning(f"⚠️ Original predictions file not found in judged metadata or doesn't exist: {original_predictions_file}")
                failed_judge_questions = []
            else:
                # Load original predictions to check which need judging
                with open(original_predictions_file, "r") as f:
                    original_data = json.load(f)
                original_predictions = original_data.get("predictions", original_data)

                # Find questions that need judging:
                # 1. Has non-empty response in original predictions
                # 2. Has empty judge_response in judged file
                failed_judge_questions = []
                for qid, original_pred in original_predictions.items():
                    if original_pred.get("response", "").strip():  # Has non-empty response
                        judged_pred = judged_predictions.get(qid, {})
                        if not judged_pred.get("judge_response", {}).get("reasoning", "").strip():  # Empty judge response
                            failed_judge_questions.append(qid)

            if failed_judge_questions:
                self.logger.info(f"⚖️ Re-judging {len(failed_judge_questions)} failed judgments concurrently (workers: {self.config.hle.num_workers})")

                # Get judge model and dataset info from judged file metadata
                original_judge_model = judging_metadata.get("judge_model", self.config.hle.judge_model)
                dataset_name = judging_metadata.get("dataset_name", "cais/hle")
                original_endpoint_info = judging_metadata.get("endpoint", {})

                # Create a judge instance with the original judge model settings
                from config import HLEConfig
                judge_config = HLEConfig(
                    dataset_name=dataset_name,
                    dataset_split=self.config.hle.dataset_split,
                    judge_model=original_judge_model,
                    max_completion_tokens=self.config.hle.max_completion_tokens,
                    judge_max_completion_tokens=self.config.hle.judge_max_completion_tokens,
                    temperature=self.config.hle.temperature,
                    num_workers=self.config.hle.num_workers,
                    max_concurrent_models=self.config.hle.max_concurrent_models,
                    print_streaming_output=self.config.hle.print_streaming_output
                )

                from hle.judge import HLEJudge
                fix_judge = HLEJudge(judge_config, self.config.zenmux)

                # Set the judge endpoint directly from metadata to avoid API calls
                if original_endpoint_info:
                    from zenmux.models import ZenMuxEndpoint
                    fix_judge.judge_endpoint = ZenMuxEndpoint(
                        pricing_completion="0",  # Not needed for fix mode
                        pricing_prompt="0",      # Not needed for fix mode
                        context_length=original_endpoint_info.get("context_length", 200000),
                        provider=original_endpoint_info.get("provider", "OpenAI"),
                        provider_slug=original_endpoint_info.get("provider_slug", "openai"),
                        max_completion_tokens=original_endpoint_info.get("max_completion_tokens", 4096),
                        supports_streaming=original_endpoint_info.get("supports_streaming", True),
                        supports_reasoning=original_endpoint_info.get("supports_reasoning", False),
                        supports_tool_parameters=original_endpoint_info.get("supports_tool_parameters", True),
                        supported_parameters=original_endpoint_info.get("supported_parameters", []),
                        can_abort=original_endpoint_info.get("can_abort", True),
                        visible=original_endpoint_info.get("visible", 1),
                        suitable_api=original_endpoint_info.get("suitable_api", "chat.completions")
                    )

                # Progress tracking for judge fixes
                judge_completed_count = 0
                total_judge_questions = len(failed_judge_questions)
                judge_progress_interval = max(1, total_judge_questions // 10)  # Report every 10% or at least every question

                # Concurrent judge processing
                judge_semaphore = asyncio.Semaphore(self.config.hle.num_workers)

                async def fix_single_judge_question(question_id: str):
                    """Fix a single judge question."""
                    nonlocal judge_completed_count
                    async with judge_semaphore:
                        if question_id in question_map and question_id in original_predictions:
                            question = question_map[question_id]
                            try:
                                # Use original prediction data, but ensure it has the current judged structure
                                prediction_for_judging = original_predictions[question_id].copy()
                                # Preserve any existing judge information from judged file
                                if question_id in judged_predictions:
                                    existing_judge_info = judged_predictions[question_id]
                                    prediction_for_judging.update({k: v for k, v in existing_judge_info.items()
                                                                  if k in ['judge_response', 'judge_performance', 'judge_generation_id']})

                                unique_id, updated_prediction, performance_metrics = await fix_judge.judge_single_response(
                                    question, {question_id: prediction_for_judging}
                                )
                                # Update progress
                                judge_completed_count += 1
                                if judge_completed_count % judge_progress_interval == 0 or judge_completed_count == total_judge_questions:
                                    progress_pct = (judge_completed_count / total_judge_questions) * 100
                                    self.logger.info(f"⚖️ Judge progress: {judge_completed_count}/{total_judge_questions} ({progress_pct:.0f}%) completed")

                                has_reasoning = updated_prediction.get("judge_response", {}).get("reasoning", "").strip() != ""
                                return question_id, updated_prediction, has_reasoning
                            except Exception as e:
                                judge_completed_count += 1
                                if judge_completed_count % judge_progress_interval == 0 or judge_completed_count == total_judge_questions:
                                    progress_pct = (judge_completed_count / total_judge_questions) * 100
                                    self.logger.info(f"⚖️ Judge progress: {judge_completed_count}/{total_judge_questions} ({progress_pct:.0f}%) completed")
                                self.logger.error(f"❌ Error re-judging question {question_id}: {e}")
                                return question_id, None, False

                        judge_completed_count += 1
                        if judge_completed_count % judge_progress_interval == 0 or judge_completed_count == total_judge_questions:
                            progress_pct = (judge_completed_count / total_judge_questions) * 100
                            self.logger.info(f"⚖️ Judge progress: {judge_completed_count}/{total_judge_questions} ({progress_pct:.0f}%) completed")
                        return question_id, None, False

                # Process failed judge questions concurrently
                judge_fix_tasks = [fix_single_judge_question(qid) for qid in failed_judge_questions]
                judge_fix_results = await asyncio.gather(*judge_fix_tasks, return_exceptions=True)

                # Process results
                judge_fixed_count = 0
                for result in judge_fix_results:
                    if isinstance(result, Exception):
                        self.logger.error(f"❌ Exception during judge fix: {result}")
                        continue

                    question_id, updated_prediction, was_fixed = result
                    if updated_prediction:
                        judged_predictions[question_id] = updated_prediction
                        if was_fixed:
                            judge_fixed_count += 1
                            model_fixed = True
                            self.logger.debug(f"✅ Fixed judgment for question {question_id}")
                        else:
                            self.logger.debug(f"❌ Still failed judgment for question {question_id}")

                self.logger.info(f"✅ Fixed {judge_fixed_count}/{len(failed_judge_questions)} judge failures")

                # Save updated judged data (REFACTOR: no metrics, they will be calculated in metrics_summary)
                final_judged_data = {
                    "judging_metadata": judging_metadata,
                    "judged_predictions": judged_predictions
                }

                # Handle old format fallback
                if not judging_metadata:
                    final_judged_data = judged_predictions

                with open(judged_file, "w") as f:
                    json.dump(final_judged_data, f, indent=4)
        else:
            self.logger.info(f"📄 No judged file found for {model_identifier}, skipping judge fixes")

        # Check final status after fix attempts
        final_failed_eval_questions = []
        final_failed_judge_questions = []

        # Re-check evaluation completeness after fixes
        if os.path.exists(predictions_file):
            with open(predictions_file, "r") as f:
                final_predictions_data = json.load(f)
            final_predictions = final_predictions_data.get("predictions", final_predictions_data)
            final_failed_eval_questions = [
                qid for qid, pred in final_predictions.items()
                if not pred.get("response", "").strip()
            ]

        # Re-check judge completeness after fixes
        if os.path.exists(judged_file):
            with open(judged_file, "r") as f:
                final_judged_data = json.load(f)
            final_judged_predictions = final_judged_data.get("judged_predictions", final_judged_data)
            final_failed_judge_questions = [
                qid for qid, pred in final_judged_predictions.items()
                if pred.get("response", "").strip() and not pred.get("judge_response", {}).get("reasoning", "").strip()
            ]

        # Create detailed fix info
        had_failures = bool(failed_eval_questions or failed_judge_questions)
        fix_info = {
            "had_failures": had_failures,
            "was_fixed": model_fixed,
            "initial_eval_failures": len(failed_eval_questions),
            "initial_judge_failures": len(failed_judge_questions),
            "final_eval_failures": len(final_failed_eval_questions),
            "final_judge_failures": len(final_failed_judge_questions),
            "still_has_eval_failures": len(final_failed_eval_questions) > 0,
            "still_has_judge_failures": len(final_failed_judge_questions) > 0
        }

        # Log final status for this model
        if not had_failures:
            self.logger.info(f"✅ {model_identifier}: No failures found")
        elif model_fixed:
            eval_fixed = len(failed_eval_questions) - len(final_failed_eval_questions)
            judge_fixed = len(failed_judge_questions) - len(final_failed_judge_questions)
            self.logger.info(f"🎉 {model_identifier}: Fixed {eval_fixed} eval + {judge_fixed} judge failures")
        else:
            self.logger.warning(f"⚠️ {model_identifier}: Could not fix any failures ({len(final_failed_eval_questions)} eval + {len(final_failed_judge_questions)} judge still failing)")

        return model_identifier, fix_info



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