"""HLE evaluation logic."""

import os
import json
import asyncio
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
        output_dir: str = "predictions"
    ):
        self.hle_config = hle_config
        self.zenmux_config = zenmux_config
        self.output_dir = output_dir
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

        # Generate output filename
        safe_model_name = model_identifier.replace("/", "_").replace(":", "_")
        output_filepath = os.path.join(self.output_dir, f"hle_{safe_model_name}.json")

        # Load existing predictions if file exists
        existing_predictions = {}
        if os.path.exists(output_filepath):
            try:
                with open(output_filepath, "r") as f:
                    existing_predictions = json.load(f)
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

        # Save updated predictions
        with open(output_filepath, "w") as f:
            json.dump(existing_predictions, f, indent=4)

        print(f"✅ Completed evaluation for {model_identifier}")
        print(f"📝 New predictions: {new_predictions}")
        print(f"💾 Total predictions: {len(existing_predictions)}")
        print(f"📁 Saved to: {output_filepath}")

        return output_filepath