"""HLE evaluation logic."""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm.asyncio import tqdm_asyncio

from .dataset import HLEDataset
from zenmux import ZenMuxOpenAIClient, ZenMuxEndpoint
from config import HLEConfig, ZenMuxConfig


class HLEEvaluator:
    """Evaluates models on HLE dataset."""

    def __init__(
        self,
        hle_config: HLEConfig,
        zenmux_config: ZenMuxConfig,
        output_dir: str = "predictions",
        batch_timestamp: str = None
    ):
        self.hle_config = hle_config
        self.zenmux_config = zenmux_config
        self.output_dir = output_dir
        self.batch_timestamp = batch_timestamp
        self.dataset = HLEDataset(hle_config.dataset_name, hle_config.dataset_split)
        self.zenmux_client = ZenMuxOpenAIClient(zenmux_config)

        # Note: Output directory will be created when evaluation starts

    async def evaluate_single_question(
        self,
        question: Dict[str, Any],
        model_name: str,
        endpoint: ZenMuxEndpoint
    ) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """Evaluate a single question with a model."""
        try:
            # Check if model is o1-based (for system prompt handling)
            is_o1_model = "o1" in model_name.lower()
            messages = self.dataset.format_message(question, for_o1=is_o1_model)

            client = self.zenmux_client.get_client(endpoint)

            # Prepare request parameters
            request_params = {
                "model": model_name,
                "messages": messages,
                "max_completion_tokens": self.hle_config.max_completion_tokens,
                "stream": False,
            }

            # Add temperature only for non-o1 models
            if not is_o1_model:
                request_params["temperature"] = self.hle_config.temperature

            response = await client.chat.completions.create(**request_params)

            content = response.choices[0].message.content
            usage = json.loads(response.usage.json()) if response.usage else {}

            if content is None:
                return None

            return question["id"], content, usage

        except Exception as e:
            print(f"Error evaluating question {question.get('id', 'unknown')}: {e}")
            return None

    async def evaluate_model(
        self,
        model_identifier: str,
        endpoint: ZenMuxEndpoint,
        text_only: bool = False,
        max_samples: Optional[int] = None
    ) -> str:
        """Evaluate a model on the HLE dataset."""
        print(f"🚀 Starting evaluation for {model_identifier}")

        # Get questions
        questions = self.dataset.get_questions(text_only=text_only, max_samples=max_samples)
        print(f"📊 Total questions: {len(questions)}")

        # Ensure output directory exists when actually starting evaluation
        os.makedirs(self.output_dir, exist_ok=True)

        # Generate output filename with batch timestamp
        safe_model_name = model_identifier.replace("/", "_").replace(":", "_")

        # Use batch timestamp if provided, otherwise generate one
        if self.batch_timestamp:
            timestamp = self.batch_timestamp
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_filepath = os.path.join(self.output_dir, f"hle_{safe_model_name}_{timestamp}.json")

        # Load existing predictions if file exists
        existing_predictions = {}
        if os.path.exists(output_filepath):
            try:
                with open(output_filepath, "r") as f:
                    data = json.load(f)
                    # Handle both old format (direct predictions) and new format (with metadata)
                    if "predictions" in data:
                        existing_predictions = data["predictions"]
                    else:
                        existing_predictions = data
                print(f"📂 Found existing file: {os.path.basename(output_filepath)}")
                print(f"📂 Loaded {len(existing_predictions)} existing predictions")
            except Exception as e:
                print(f"⚠️ Warning: Could not load existing predictions: {e}")

        # Filter out questions that already have predictions
        remaining_questions = [
            q for q in questions
            if q["id"] not in existing_predictions
        ]

        if not remaining_questions:
            print(f"✅ All questions already evaluated for {model_identifier}")
            return output_filepath

        print(f"🔄 Evaluating {len(remaining_questions)} remaining questions")

        async def bound_evaluate(question):
            async with semaphore:
                return await self.evaluate_single_question(
                    question, model_identifier, endpoint
                )

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.hle_config.num_workers)

        # Evaluate remaining questions
        async with semaphore:
            tasks = [bound_evaluate(q) for q in remaining_questions]
            results = await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {model_identifier}")

        # Process results
        new_predictions = 0
        for result in results:
            if result is None:
                continue

            unique_id, response, usage = result
            existing_predictions[unique_id] = {
                "model": model_identifier,
                "response": response,
                "usage": usage
            }
            new_predictions += 1

        # Add metadata to the predictions file
        metadata = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_identifier": model_identifier,
                "endpoint": {
                    "provider_slug": endpoint.provider_slug,
                    "provider": endpoint.provider,
                    "context_length": endpoint.context_length,
                    "max_completion_tokens": endpoint.max_completion_tokens,
                    "supports_streaming": endpoint.supports_streaming,
                    "suitable_api": endpoint.suitable_api
                },
                "dataset_config": {
                    "dataset_name": self.hle_config.dataset_name,
                    "dataset_split": self.hle_config.dataset_split,
                    "text_only": text_only,
                    "max_samples": max_samples
                },
                "evaluation_config": {
                    "max_completion_tokens": self.hle_config.max_completion_tokens,
                    "temperature": self.hle_config.temperature,
                    "num_workers": self.hle_config.num_workers,
                    "timeout": self.hle_config.timeout,
                    "max_retries": self.hle_config.max_retries
                },
                "statistics": {
                    "total_questions": len(questions),
                    "remaining_questions": len(remaining_questions),
                    "new_predictions": new_predictions,
                    "total_predictions": len(existing_predictions)
                }
            }
        }

        # Save updated predictions with metadata
        final_output = {
            **metadata,
            "predictions": existing_predictions
        }

        with open(output_filepath, "w") as f:
            json.dump(final_output, f, indent=4)

        print(f"✅ Completed evaluation for {model_identifier}")
        print(f"📝 New predictions: {new_predictions}")
        print(f"💾 Total predictions: {len(existing_predictions)}")
        print(f"📁 Saved to: {output_filepath}")

        return output_filepath