"""HLE evaluation logic."""

import os
import json
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
# from tqdm.asyncio import tqdm_asyncio  # Removed for cleaner static output

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
    ) -> Tuple[str, Dict[str, Any]]:
        """Evaluate a single question with a model. Always returns a result with has_answer field."""
        question_id = question.get('id', 'unknown')

        try:
            # Check if model is o1-based (for system prompt handling)
            is_o1_model = "o1" in model_name.lower()
            messages = self.dataset.format_message(question, for_o1=is_o1_model)

            client = self.zenmux_client.get_client(endpoint)

            # Determine max_completion_tokens: use model's own value unless config overrides
            max_completion_tokens = endpoint.max_completion_tokens
            if self.hle_config.max_completion_tokens is not None:
                max_completion_tokens = self.hle_config.max_completion_tokens

            # Prepare request parameters
            request_params = {
                "model": model_name,
                "messages": messages,
                "max_completion_tokens": max_completion_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }

            # Add temperature only for non-o1 models
            if not is_o1_model:
                request_params["temperature"] = self.hle_config.temperature

            # Record request start time
            request_start_time = time.time() * 1000  # Convert to milliseconds

            # Create streaming response and collect all content
            stream = await client.chat.completions.create(**request_params)

            # Performance tracking variables
            first_token_time = None
            content_chunks = []
            usage = {}
            generation_id = None

            async for chunk in stream:
                # Capture generation ID from any chunk (usually available in first chunk)
                if chunk.id and generation_id is None:
                    generation_id = chunk.id

                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        # Record first token time if this is the first content
                        if first_token_time is None:
                            first_token_time = time.time() * 1000
                        content_chunks.append(delta.content)

                # Capture usage information from the final chunk
                if chunk.usage:
                    usage = json.loads(chunk.usage.json())

            # Record generation end time
            generation_end_time = time.time() * 1000

            # Combine all content chunks
            content = "".join(content_chunks)

            # Calculate performance metrics
            performance_metrics = {}

            if first_token_time is not None:
                # First token latency: time from request start to first token
                performance_metrics['first_token_latency_ms'] = first_token_time - request_start_time

                # Generation time: time from first token to end
                performance_metrics['generation_time_ms'] = generation_end_time - first_token_time

                # Calculate throughput (tokens per second)
                completion_tokens = usage.get('completion_tokens', 0) or 0
                if completion_tokens > 0 and performance_metrics['generation_time_ms'] > 0:
                    generation_time_seconds = performance_metrics['generation_time_ms'] / 1000
                    performance_metrics['throughput_tokens_per_second'] = completion_tokens / generation_time_seconds
                else:
                    performance_metrics['throughput_tokens_per_second'] = 0.0
            else:
                # No content received, set default values
                performance_metrics['first_token_latency_ms'] = 0.0
                performance_metrics['generation_time_ms'] = 0.0
                performance_metrics['throughput_tokens_per_second'] = 0.0

            # Return success result with has_answer=True
            result = {
                "model": model_name,
                "response": content,
                "usage": usage,
                "performance": performance_metrics,
                "generation_id": generation_id,
                "has_answer": bool(content)  # True if we got content, False if empty
            }

            return question_id, result

        except Exception as e:
            error_msg = str(e)

            # Log different error types with model information for better debugging
            if "Connection error" in error_msg or "peer closed connection" in error_msg:
                self.logger.warning(f"Connection issue for question {question_id} [{model_name}]: {error_msg}")
            elif "timeout" in error_msg.lower():
                self.logger.warning(f"Timeout for question {question_id} [{model_name}]: {error_msg}")
            elif "Too many open files" in error_msg:
                self.logger.error(f"File handle limit reached for question {question_id} [{model_name}]: {error_msg}")
            else:
                self.logger.error(f"Error evaluating question {question_id} [{model_name}]: {e}")

            # Return failure result with has_answer=False
            result = {
                "model": model_name,
                "response": "",
                "usage": {},
                "performance": {
                    'first_token_latency_ms': 0.0,
                    'generation_time_ms': 0.0,
                    'throughput_tokens_per_second': 0.0
                },
                "generation_id": None,
                "has_answer": False,
                "error": error_msg
            }

            return question_id, result

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

                # Clear client connection on retry to prevent connection issues
                try:
                    await self.zenmux_client.close()
                except:
                    pass  # Ignore errors during cleanup

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

            # Filter out questions that already have predictions (or need retry for failed ones)
            remaining_questions = []
            for q in questions:
                question_id = q["id"]
                if question_id not in existing_predictions:
                    remaining_questions.append(q)
                elif existing_predictions[question_id].get("has_answer") is False:
                    # Re-evaluate questions that previously failed
                    remaining_questions.append(q)

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
            model_logger.info(f"🔄 Starting evaluation of {len(remaining_questions)} questions for {model_identifier} (attempt {retry_attempt + 1})")
            results = await asyncio.gather(*tasks)

            # Process results and collect performance metrics
            performance_data = []
            successful_count = 0
            for question_id, result in results:
                # Always record the result, regardless of success or failure
                existing_predictions[question_id] = result

                # Collect performance data only for successful evaluations
                if result.get("has_answer"):
                    performance_data.append(result["performance"])
                    successful_count += 1

            # Early exit check: if success rate is too low, don't retry to avoid wasting time
            if len(remaining_questions) > 0:
                success_rate = successful_count / len(remaining_questions)
                if success_rate < 0.2 and retry_attempt > 0:  # Less than 20% success rate after first retry
                    model_logger.warning(f"⚠️ Low success rate ({success_rate:.1%}), skipping further retries to prevent timeout")
                    self.logger.warning(f"⚠️ Skipping retries for {model_identifier} due to low success rate ({success_rate:.1%})")
                    break

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
                        "timeout": self.zenmux_config.timeout,
                        "max_retries": self.zenmux_config.max_retries,
                        "max_evaluation_retries": self.hle_config.max_evaluation_retries
                    },
                    "statistics": {
                        "total_questions": len(questions),
                        "remaining_questions": len(questions) - len(existing_predictions),
                        "total_predictions": len(existing_predictions),
                        "retry_attempt": retry_attempt
                    }
                }
            }

            # Calculate and add average performance metrics
            if performance_data:
                avg_first_token_latency = sum(p.get('first_token_latency_ms', 0) for p in performance_data) / len(performance_data)
                avg_generation_time = sum(p.get('generation_time_ms', 0) for p in performance_data) / len(performance_data)
                avg_throughput = sum(p.get('throughput_tokens_per_second', 0) for p in performance_data) / len(performance_data)

                metadata["evaluation_metadata"]["performance_averages"] = {
                    "avg_first_token_latency_ms": round(avg_first_token_latency, 2),
                    "avg_generation_time_ms": round(avg_generation_time, 2),
                    "avg_throughput_tokens_per_second": round(avg_throughput, 2),
                    "samples_count": len(performance_data)
                }

            # Save updated predictions with metadata
            final_output = {
                **metadata,
                "predictions": existing_predictions
            }

            try:
                with open(output_filepath, "w") as f:
                    json.dump(final_output, f, indent=4)
            except OSError as e:
                self.logger.error(f"❌ Failed to write predictions file {output_filepath}: {e}")
                if "Too many open files" in str(e):
                    self.logger.error("💡 Try reducing num_workers or max_concurrent_models in config.py")
                raise

            # Check if evaluation is complete based on has_answer field
            total_predictions = len(existing_predictions)
            successful_predictions = sum(1 for pred in existing_predictions.values() if pred.get("has_answer", False))
            failed_predictions = total_predictions - successful_predictions

            if total_predictions == len(questions) and failed_predictions == 0:
                self.logger.info(f"✅ Evaluation complete: {successful_predictions}/{len(questions)} successful predictions")
                model_logger.info(f"✅ Evaluation complete: {successful_predictions}/{len(questions)} successful predictions")
                break
            elif retry_attempt < self.hle_config.max_evaluation_retries:
                self.logger.warning(f"⚠️ Evaluation incomplete: {successful_predictions}/{len(questions)} successful predictions ({failed_predictions} failed)")
                self.logger.info(f"🔄 Will retry failed questions (attempt {retry_attempt + 2}/{self.hle_config.max_evaluation_retries + 1})")
                model_logger.warning(f"⚠️ Evaluation incomplete: {successful_predictions}/{len(questions)} successful predictions ({failed_predictions} failed)")
                model_logger.info(f"🔄 Will retry failed questions (attempt {retry_attempt + 2}/{self.hle_config.max_evaluation_retries + 1})")
            else:
                self.logger.error(f"❌ Evaluation incomplete after {self.hle_config.max_evaluation_retries} retries")
                self.logger.error(f"❌ Final result: {successful_predictions}/{len(questions)} successful predictions ({failed_predictions} failed)")
                model_logger.error(f"❌ Evaluation incomplete after {self.hle_config.max_evaluation_retries} retries")
                model_logger.error(f"❌ Final result: {successful_predictions}/{len(questions)} successful predictions ({failed_predictions} failed)")

        self.logger.info(f"✅ Completed evaluation for {model_identifier}")
        self.logger.info(f"📝 Final predictions: {len(existing_predictions)}")
        self.logger.info(f"📁 Saved to: {output_filepath}")
        model_logger.info(f"✅ Completed evaluation")
        model_logger.info(f"📝 Final predictions: {len(existing_predictions)}")
        model_logger.info(f"📁 Saved to: {output_filepath}")

        return output_filepath