"""HLE evaluation runner with ZenMux integration."""

import asyncio
from typing import List, Optional, Dict, Any

from .evaluation import HLEEvaluator
from .judge import HLEJudge
from zenmux import ZenMuxAPI
from config import BenchmarkConfig


class HLERunner:
    """Main runner for HLE evaluations."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.zenmux_api = ZenMuxAPI(config.zenmux)
        self.evaluator = HLEEvaluator(
            config.hle,
            config.zenmux,
            config.predictions_dir
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
                output_dir=self.config.judged_dir
            )
            results["judged_file"] = judged_file

        return results

    async def run_zenmux_models_evaluation(
        self,
        text_only: bool = False,
        max_samples: Optional[int] = None,
        auto_judge: bool = True,
        model_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Run evaluation for all ZenMux models."""
        print("🌟 Starting ZenMux Models Evaluation")
        print(f"📝 Text only: {text_only}")
        print(f"📊 Max samples: {max_samples}")
        print(f"🏛️ Auto judge: {auto_judge}")

        # Get all model-endpoint pairs
        model_endpoint_pairs = self.zenmux_api.get_all_model_endpoint_pairs(text_only=text_only)

        # Apply model filter if specified
        if model_filter:
            model_endpoint_pairs = [
                (model_id, model, endpoint)
                for model_id, model, endpoint in model_endpoint_pairs
                if model_filter.lower() in model_id.lower()
            ]

        print(f"🎯 Total model endpoints to evaluate: {len(model_endpoint_pairs)}")

        results = []
        for i, (model_identifier, model, endpoint) in enumerate(model_endpoint_pairs, 1):
            print(f"\n📋 Progress: {i}/{len(model_endpoint_pairs)}")

            try:
                result = await self.run_single_model_evaluation(
                    model_identifier=model_identifier,
                    endpoint=endpoint,
                    text_only=text_only,
                    max_samples=max_samples,
                    auto_judge=auto_judge
                )
                results.append(result)

            except Exception as e:
                print(f"❌ Error evaluating {model_identifier}: {e}")
                results.append({
                    "model_identifier": model_identifier,
                    "error": str(e),
                    "predictions_file": None,
                    "judged_file": None,
                    "metrics": None
                })

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
            print(f"\n📁 Prediction files saved in: {self.config.predictions_dir}")
            print(f"📁 Judged files saved in: {self.config.judged_dir}")