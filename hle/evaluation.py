"""HLE evaluation logic."""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm.asyncio import tqdm_asyncio

from .dataset import HLEDataset
from zenmux import ZenMuxOpenAIClient, ZenMuxEndpoint
from config import HLEConfig, ZenMuxConfig
from utils.logging import get_evaluation_logger, get_model_logger, PerformanceTimer


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
        self.logger = get_evaluation_logger()

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
                "stream": True,
            }

            # Add temperature only for non-o1 models
            if not is_o1_model:
                request_params["temperature"] = self.hle_config.temperature

            # Create streaming response and collect all content
            stream = await client.chat.completions.create(**request_params)

            # Collect all streaming content
            content_chunks = []
            usage = {}

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content_chunks.append(delta.content)

                # Capture usage information from the final chunk
                if chunk.usage:
                    usage = json.loads(chunk.usage.json())

            # Combine all content chunks
            content = "".join(content_chunks)

            if not content:
                return None

            return question["id"], content, usage

        except Exception as e:
            self.logger.error(f"Error evaluating question {question.get('id', 'unknown')}: {e}")
            return None

    async def evaluate_model(
        self,
        model_identifier: str,
        endpoint: ZenMuxEndpoint,
        text_only: bool = False,
        max_samples: Optional[int] = None
    ) -> str:
        """Evaluate a model on the HLE dataset with retry logic for incomplete evaluations."""
        # Get model-specific logger
        model_logger = get_model_logger(model_identifier)

        self.logger.info(f"🚀 Starting evaluation for {model_identifier}")
        model_logger.info(f"🚀 Starting evaluation for {model_identifier}")

        # Get questions
        questions = self.dataset.get_questions(text_only=text_only, max_samples=max_samples)
        self.logger.info(f"📊 Total questions: {len(questions)}")
        model_logger.info(f"📊 Total questions: {len(questions)} | text_only: {text_only} | max_samples: {max_samples}")

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

        # Retry logic for incomplete evaluations
        for retry_attempt in range(self.hle_config.max_evaluation_retries + 1):
            if retry_attempt > 0:
                self.logger.warning(f"🔄 Retry attempt {retry_attempt}/{self.hle_config.max_evaluation_retries}")
                model_logger.warning(f"🔄 Retry attempt {retry_attempt}/{self.hle_config.max_evaluation_retries}")

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
                    self.logger.info(f"📂 Found existing file: {os.path.basename(output_filepath)}")
                    self.logger.info(f"📂 Loaded {len(existing_predictions)} existing predictions")
                    model_logger.info(f"📂 Loaded {len(existing_predictions)} existing predictions from {os.path.basename(output_filepath)}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Warning: Could not load existing predictions: {e}")
                    model_logger.warning(f"⚠️ Warning: Could not load existing predictions: {e}")

            # Filter out questions that already have predictions
            remaining_questions = [
                q for q in questions
                if q["id"] not in existing_predictions
            ]

            if not remaining_questions:
                self.logger.info(f"✅ All questions already evaluated for {model_identifier}")
                model_logger.info(f"✅ All questions already evaluated")
                break

            self.logger.info(f"🔄 Evaluating {len(remaining_questions)} remaining questions")
            model_logger.info(f"🔄 Evaluating {len(remaining_questions)} remaining questions")

            async def bound_evaluate(question):
                async with semaphore:
                    with PerformanceTimer(model_logger, f"question {question['id']}", level=logging.DEBUG):
                        return await self.evaluate_single_question(
                            question, model_identifier, endpoint
                        )

            # Create semaphore for rate limiting
            semaphore = asyncio.Semaphore(self.hle_config.num_workers)

            # Evaluate remaining questions
            tasks = [bound_evaluate(q) for q in remaining_questions]
            results = await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {model_identifier} (attempt {retry_attempt + 1})")

            # Process results
            for result in results:
                if result is None:
                    continue

                unique_id, response, usage = result
                existing_predictions[unique_id] = {
                    "model": model_identifier,
                    "response": response,
                    "usage": usage
                }

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
                        "max_retries": self.hle_config.max_retries,
                        "max_evaluation_retries": self.hle_config.max_evaluation_retries
                    },
                    "statistics": {
                        "total_questions": len(questions),
                        "remaining_questions": len(remaining_questions),
                        "total_predictions": len(existing_predictions),
                        "retry_attempt": retry_attempt
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

            # Check if evaluation is complete
            if len(existing_predictions) == len(questions):
                self.logger.info(f"✅ Evaluation complete: {len(existing_predictions)}/{len(questions)} predictions")
                model_logger.info(f"✅ Evaluation complete: {len(existing_predictions)}/{len(questions)} predictions")
                break
            elif retry_attempt < self.hle_config.max_evaluation_retries:
                missing_count = len(questions) - len(existing_predictions)
                self.logger.warning(f"⚠️ Incomplete evaluation: {len(existing_predictions)}/{len(questions)} predictions ({missing_count} missing)")
                self.logger.info(f"🔄 Will retry missing questions (attempt {retry_attempt + 2}/{self.hle_config.max_evaluation_retries + 1})")
                model_logger.warning(f"⚠️ Incomplete evaluation: {len(existing_predictions)}/{len(questions)} predictions ({missing_count} missing)")
                model_logger.info(f"🔄 Will retry missing questions (attempt {retry_attempt + 2}/{self.hle_config.max_evaluation_retries + 1})")
            else:
                missing_count = len(questions) - len(existing_predictions)
                self.logger.error(f"❌ Evaluation incomplete after {self.hle_config.max_evaluation_retries} retries")
                self.logger.error(f"❌ Final result: {len(existing_predictions)}/{len(questions)} predictions ({missing_count} missing)")
                model_logger.error(f"❌ Evaluation incomplete after {self.hle_config.max_evaluation_retries} retries")
                model_logger.error(f"❌ Final result: {len(existing_predictions)}/{len(questions)} predictions ({missing_count} missing)")

        self.logger.info(f"✅ Completed evaluation for {model_identifier}")
        self.logger.info(f"📝 Final predictions: {len(existing_predictions)}")
        self.logger.info(f"📁 Saved to: {output_filepath}")
        model_logger.info(f"✅ Completed evaluation")
        model_logger.info(f"📝 Final predictions: {len(existing_predictions)}")
        model_logger.info(f"📁 Saved to: {output_filepath}")

        return output_filepath