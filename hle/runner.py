"""HLE evaluation runner with ZenMux integration."""

import asyncio
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from .evaluation import HLEEvaluator
from .judge import HLEJudge
from zenmux import ZenMuxAPI
from config import BenchmarkConfig


class HLERunner:
    """Main runner for HLE evaluations."""

    def __init__(self, config: BenchmarkConfig, batch_timestamp: str = None):
        self.config = config

        # Setup timestamped directories
        if batch_timestamp:
            self.config.setup_timestamped_directories(batch_timestamp)
        else:
            self.config.setup_timestamped_directories()

        self.batch_timestamp = self.config.batch_timestamp
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
        print(f"\n{'='*60}")
        print(f"🚀 EVALUATING: {model_identifier}")
        print(f"{'='*60}")

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

        # Run judging if requested
        if auto_judge:
            print(f"\n🏛️ JUDGING: {model_identifier}")
            judged_file = await self.judge.judge_predictions(
                predictions_file=predictions_file,
                dataset_name=self.config.hle.dataset_name,
                output_dir=self.config.get_judged_dir()
            )
            results["judged_file"] = judged_file

            # Extract metrics from judged file
            results["metrics"] = self.extract_metrics_from_judged_file(judged_file)

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
        print("🌟 Starting ZenMux Models Evaluation")
        print(f"📝 Text only: {text_only}")
        print(f"📊 Max samples: {max_samples}")
        print(f"🏛️ Auto judge: {auto_judge}")
        print(f"🔄 Max concurrent models: {self.config.hle.max_concurrent_models}")
        print(f"🔄 Workers per model: {self.config.hle.num_workers}")

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
                print(f"🚫 Excluded {excluded_count} model endpoints based on exclude patterns")
                print(f"📉 Remaining models: {len(model_endpoint_pairs)} (was {original_count})")

        print(f"🎯 Total model endpoints to evaluate: {len(model_endpoint_pairs)}")

        # Outer layer: Model-level concurrency control
        async def bound_model_evaluation(model_data):
            model_identifier, model, endpoint = model_data
            async with models_semaphore:
                print(f"🚀 Starting evaluation: {model_identifier}")
                try:
                    result = await self.run_single_model_evaluation(
                        model_identifier=model_identifier,
                        endpoint=endpoint,
                        text_only=text_only,
                        max_samples=max_samples,
                        auto_judge=auto_judge
                    )
                    print(f"✅ Completed evaluation: {model_identifier}")
                    return result

                except Exception as e:
                    print(f"❌ Error evaluating {model_identifier}: {e}")
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

        print(f"\n✅ Completed evaluation of {len(results)} model endpoints")
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

    def extract_metrics_from_judged_file(self, judged_file: str) -> Optional[Dict[str, Any]]:
        """Extract metrics from a judged file."""
        try:
            with open(judged_file, "r") as f:
                data = json.load(f)
                return data.get("metrics")
        except Exception as e:
            print(f"⚠️ Warning: Could not extract metrics from {judged_file}: {e}")
            return None

    def save_metrics_summary(self, results: List[Dict[str, Any]], run_metadata: Dict[str, Any] = None) -> str:
        """Save a unified metrics summary for all evaluations."""
        if run_metadata is None:
            run_metadata = {}

        # Create summary data structure
        summary = {
            "summary_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_evaluations": len(results),
                "successful_evaluations": len([r for r in results if r.get("metrics") is not None]),
                "failed_evaluations": len([r for r in results if r.get("error") is not None]),
                "run_metadata": run_metadata
            },
            "model_results": []
        }

        # Add results for each model
        for result in results:
            model_summary = {
                "model_identifier": result["model_identifier"],
                "predictions_file": result.get("predictions_file"),
                "judged_file": result.get("judged_file"),
                "metrics": result.get("metrics"),
                "error": result.get("error")
            }
            summary["model_results"].append(model_summary)

        # Save summary file in the timestamped run directory
        summary_file = os.path.join(self.config.run_dir, f"metrics_summary_{self.batch_timestamp}.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"📊 Metrics summary saved to: {summary_file}")
        return summary_file

    def print_summary(self, results: List[Dict[str, Any]]):
        """Print a summary of evaluation results."""
        print(f"\n{'='*60}")
        print("📊 EVALUATION SUMMARY")
        print(f"{'='*60}")

        successful = [r for r in results if r.get("predictions_file") is not None]
        failed = [r for r in results if r.get("error") is not None]

        print(f"✅ Successful evaluations: {len(successful)}")
        print(f"❌ Failed evaluations: {len(failed)}")

        if failed:
            print("\n❌ Failed models:")
            for result in failed:
                print(f"  - {result['model_identifier']}: {result['error']}")

        if successful:
            print(f"\n📁 Run directory: {self.config.run_dir}")
            print(f"📁 Prediction files saved in: {self.config.get_predictions_dir()}")
            print(f"📁 Judged files saved in: {self.config.get_judged_dir()}")

            # Print metrics for each successful model
            models_with_metrics = [r for r in successful if r.get("metrics") is not None]
            if models_with_metrics:
                print(f"\n📊 METRICS SUMMARY")
                print(f"{'='*60}")
                for result in models_with_metrics:
                    metrics = result["metrics"]
                    print(f"\n🎯 {result['model_identifier']}")
                    print(f"📊 Accuracy: {metrics['accuracy']}% +/- {metrics['confidence_interval']}% | n = {metrics['total_questions']}")
                    print(f"📏 Calibration Error: {metrics['calibration_error']}")
                    print(f"✅ Evaluated: {metrics['total_evaluated']} / {metrics['total_questions']}")