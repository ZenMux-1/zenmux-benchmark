"""HLE judge for scoring model predictions."""

import os
import json
import copy
import math
import asyncio
import logging
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Literal, Tuple
from pydantic import BaseModel
# from tqdm.asyncio import tqdm_asyncio  # Removed for cleaner static output
from datasets import load_dataset

from config import HLEConfig, ZenMuxConfig
from zenmux import ZenMuxOpenAIClient
from utils.logging import get_judge_logger, PerformanceTimer


class ExtractedAnswer(BaseModel):
    """Structured response from judge model."""
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] = True  # 100% reliability


class HLEJudge:
    """Judge for scoring HLE predictions."""

    JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available."""

    def __init__(self, hle_config: HLEConfig, zenmux_config: ZenMuxConfig):
        self.hle_config = hle_config
        self.zenmux_config = zenmux_config
        self.zenmux_client = ZenMuxOpenAIClient(zenmux_config)
        self.logger = get_judge_logger()

    async def extract_answer(
        self,
        question: str,
        correct_answer: str,
        response: str
    ) -> Tuple[Dict[str, Any], Dict[str, float], Optional[str], bool]:
        """Extract and judge a single answer. Always returns a result with success flag."""
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            correct_answer=correct_answer,
            response=response
        )

        try:
            # Create a dummy endpoint for the judge model - we only need it to get the client
            from zenmux.models import ZenMuxEndpoint
            dummy_endpoint = ZenMuxEndpoint(
                pricing_completion="0",
                pricing_prompt="0",
                context_length=200000,
                provider="openai",
                provider_slug="openai",
                max_completion_tokens=4096,
                supports_streaming=True,
                supports_reasoning=False,
                supports_tool_parameters=True,
                supported_parameters=[],
                can_abort=True,
                visible=1,
                suitable_api="chat.completions"
            )

            client = self.zenmux_client.get_client(dummy_endpoint)

            # Record request start time
            request_start_time = time.time() * 1000  # Convert to milliseconds

            # Note: Structured output does not support streaming, fall back to non-streaming
            response_obj = await client.beta.chat.completions.parse(
                model=self.hle_config.judge_model,
                max_completion_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                response_format=ExtractedAnswer,
            )

            # Record generation end time
            generation_end_time = time.time() * 1000

            # Get parsed content, usage information, and generation ID
            final_parsed_content = response_obj.choices[0].message.parsed
            usage = json.loads(response_obj.usage.json()) if response_obj.usage else {}
            generation_id = response_obj.id if hasattr(response_obj, 'id') else None

            # Calculate performance metrics for non-streaming judge requests
            performance_metrics = {
                'first_token_latency_ms': generation_end_time - request_start_time,  # Total time for non-streaming
                'generation_time_ms': 0.0,  # Not applicable for non-streaming
                'throughput_tokens_per_second': 0.0  # Will calculate based on total time
            }

            # Calculate throughput based on total time
            completion_tokens = usage.get('completion_tokens', 0) or 0
            if completion_tokens > 0 and performance_metrics['first_token_latency_ms'] > 0:
                total_time_seconds = performance_metrics['first_token_latency_ms'] / 1000
                performance_metrics['throughput_tokens_per_second'] = completion_tokens / total_time_seconds

            judge_result = {
                "correct_answer": correct_answer,
                "model_answer": final_parsed_content.extracted_final_answer,
                "reasoning": final_parsed_content.reasoning,
                "correct": final_parsed_content.correct,
                "confidence": final_parsed_content.confidence
            }

            return judge_result, performance_metrics, generation_id, True

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            # Extract additional info for OpenAI API errors
            additional_info = {}
            try:
                if hasattr(e, 'response'):
                    additional_info['status_code'] = getattr(e.response, 'status_code', None)
                    additional_info['response_headers'] = dict(getattr(e.response, 'headers', {}))
                    additional_info['response_text'] = getattr(e.response, 'text', None)
                if hasattr(e, 'body'):
                    additional_info['response_body'] = str(e.body)
                if hasattr(e, 'request'):
                    additional_info['request_url'] = getattr(e.request, 'url', None)
                    additional_info['request_method'] = getattr(e.request, 'method', None)
            except:
                pass

            # Create detailed error message with all available information
            detailed_error = f"""
Exception Type: {error_type}
Error Message: {error_msg if error_msg else 'No error message provided'}
Judge Model: {self.hle_config.judge_model}
Question: {question[:100] if question else 'Unknown'}...
Correct Answer: {correct_answer[:50] if correct_answer else 'Unknown'}...
Response being judged: {response[:100] if response else 'Unknown'}...
Additional Error Info: {additional_info}
"""

            self.logger.error(f"Error in judge:{detailed_error}")
            # Return error result with empty data but success=False
            error_judge_result = {
                "correct_answer": correct_answer,
                "model_answer": "",
                "reasoning": "",
                "correct": "no",
                "confidence": 0,
                "error": str(e)
            }
            error_performance = {
                'first_token_latency_ms': 0.0,
                'generation_time_ms': 0.0,
                'throughput_tokens_per_second': 0.0
            }
            return error_judge_result, error_performance, None, False

    async def judge_single_response(
        self,
        question: Dict[str, Any],
        predictions: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any], Optional[Dict[str, float]]]:
        """Judge a single prediction. Always returns a result."""
        unique_id = question["id"]
        prediction = copy.deepcopy(predictions[unique_id])
        question_text = question["question"]
        correct_answer = question["answer"]

        # If already judged successfully (has judge_response with content), skip
        if "judge_response" in prediction and prediction["judge_response"].get("reasoning", "").strip():
            return unique_id, prediction, None

        # Only judge if the prediction has an answer (non-empty response)
        if not prediction.get("response", "").strip():
            # No need to judge empty responses
            return unique_id, prediction, None

        response = prediction["response"]
        judge_result, performance_metrics, generation_id, success = await self.extract_answer(
            question_text, correct_answer, response
        )

        # Always add judge information, regardless of success
        prediction["judge_response"] = judge_result
        prediction["judge_performance"] = performance_metrics
        prediction["judge_generation_id"] = generation_id

        # Return performance metrics only if judging was successful
        return unique_id, prediction, performance_metrics if success else None

    async def judge_all_responses(
        self,
        questions: List[Dict[str, Any]],
        predictions: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any], Optional[Dict[str, float]]]]:
        """Judge all responses asynchronously."""
        async def bound_func(question):
            async with semaphore:
                with PerformanceTimer(self.logger, f"judging question {question['id']}", level=logging.DEBUG):
                    return await self.judge_single_response(question, predictions)

        semaphore = asyncio.Semaphore(self.hle_config.num_workers)
        tasks = [bound_func(q) for q in questions]
        self.logger.info(f"⚖️ Starting judgment of {len(questions)} responses")
        results = await asyncio.gather(*tasks)

        return results

    @staticmethod
    def calculate_calibration_error(confidence: np.ndarray, correct: np.ndarray, beta: int = 100) -> float:
        """Calculate calibration error using binning method."""
        # Source: https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py
        idxs = np.argsort(confidence)
        confidence = confidence[idxs]
        correct = correct[idxs]
        bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
        if bins:
            bins[-1] = [bins[-1][0], len(confidence)]

        cerr = 0
        total_examples = len(confidence)
        for i in range(len(bins) - 1):
            bin_confidence = confidence[bins[i][0]:bins[i][1]]
            bin_correct = correct[bins[i][0]:bins[i][1]]
            num_examples_in_bin = len(bin_confidence)

            if num_examples_in_bin > 0:
                difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))
                cerr += num_examples_in_bin / total_examples * np.square(difference)

        return np.sqrt(cerr)

    def calculate_metrics(self, predictions: Dict[str, Any], total_questions: int) -> Dict[str, float]:
        """Calculate accuracy and calibration metrics based on successful judgments only."""
        correct = []
        confidence = []

        # Only consider predictions with successful judgments (non-empty judge response)
        for v in predictions.values():
            if "judge_response" in v and v["judge_response"].get("reasoning", "").strip():
                judge_response = v["judge_response"]
                correct.append("yes" in judge_response["correct"])
                confidence.append(judge_response["confidence"])

        if not correct:
            return {
                "accuracy": 0.0,
                "confidence_interval": 0.0,
                "calibration_error": 0.0
            }

        correct = np.array(correct)
        confidence = np.array(confidence) / 100

        # Follow official HLE calculation: use total_questions as denominator (failed judgments count as incorrect)
        accuracy = 100 * sum(correct) / total_questions
        # Wald estimator, 95% confidence interval - use total_questions for variance calculation
        confidence_half_width = 1.96 * math.sqrt(accuracy * (100 - accuracy) / total_questions)
        calibration_error = 100 * self.calculate_calibration_error(confidence, correct, beta=100)

        return {
            "accuracy": round(accuracy, 2),
            "confidence_interval": round(confidence_half_width, 2),
            "calibration_error": round(calibration_error, 2)
        }

    async def judge_predictions(
        self,
        predictions_file: str,
        dataset_name: str = "cais/hle",
        output_dir: str = "judged"
    ) -> str:
        """Judge predictions from a file."""
        self.logger.info(f"🏛️ Starting judging for {predictions_file}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Load dataset
        dataset = load_dataset(dataset_name, split="test").to_dict()
        questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
        total_questions = len(questions)

        # Load predictions
        with open(predictions_file, "r") as f:
            predictions_data = json.load(f)

        # Handle both old format (direct predictions) and new format (with metadata)
        if "predictions" in predictions_data:
            predictions = predictions_data["predictions"]
            evaluation_metadata = predictions_data.get("evaluation_metadata", {})
        else:
            predictions = predictions_data
            evaluation_metadata = {}

        # Generate output filename using the same timestamp as predictions file
        predictions_basename = os.path.basename(predictions_file)
        # Replace "hle_" with "judged_hle_" to maintain the same timestamp
        if predictions_basename.startswith("hle_"):
            output_filename = predictions_basename.replace("hle_", "judged_hle_", 1)
        else:
            output_filename = f"judged_{predictions_basename}"
        output_filepath = os.path.join(output_dir, output_filename)

        # Load existing judged results if available
        judged_predictions = {}
        if os.path.exists(output_filepath):
            try:
                with open(output_filepath, "r") as f:
                    existing_data = json.load(f)
                    # Handle both old format (direct predictions) and new format (with metadata)
                    if "judged_predictions" in existing_data:
                        judged_predictions = existing_data["judged_predictions"]
                    else:
                        judged_predictions = existing_data
                self.logger.info(f"📂 Loaded {len(judged_predictions)} existing judgments")
            except Exception as e:
                self.logger.warning(f"⚠️ Warning: Could not load existing judgments: {e}")

        # Filter questions that need judging (including retries for failed judgments)
        questions_to_judge = []
        for q in questions:
            question_id = q["id"]
            if question_id in predictions:
                if question_id not in judged_predictions:
                    questions_to_judge.append(q)
                elif not judged_predictions[question_id].get("judge_response", {}).get("reasoning", "").strip():
                    # Re-judge questions that previously failed (empty judge response)
                    questions_to_judge.append(q)

        # Initialize performance data list
        judge_performance_data = []

        if not questions_to_judge:
            self.logger.info(f"✅ All predictions already judged")
        else:
            self.logger.info(f"🔄 Judging {len(questions_to_judge)} predictions")

            # Judge remaining questions
            results = await self.judge_all_responses(questions_to_judge, predictions)

            # Process results and collect performance metrics
            for unique_id, prediction, performance_metrics in results:
                # Always record the result, regardless of success or failure
                judged_predictions[unique_id] = prediction
                if performance_metrics is not None:
                    judge_performance_data.append(performance_metrics)

        # Calculate metrics
        metrics = self.calculate_metrics(judged_predictions, total_questions)

        # Add metadata to the judged file
        # Create endpoint info for the judge model (dummy endpoint used in extract_answer)
        judge_endpoint_info = {
            "provider_slug": "openai",
            "provider": "OpenAI",
            "context_length": 200000,
            "max_completion_tokens": 4096,
            "supports_streaming": True,
            "suitable_api": "chat.completions"
        }

        metadata = {
            "judging_metadata": {
                "timestamp": datetime.now().isoformat(),
                "judge_model": self.hle_config.judge_model,
                "dataset_name": dataset_name,
                "original_predictions_file": predictions_file,
                "endpoint": judge_endpoint_info,  # Judge model's endpoint information
                "judge_config": {
                    "num_workers": self.hle_config.num_workers,
                    "timeout": self.zenmux_config.timeout,
                    "max_retries": self.zenmux_config.max_retries
                },
                "evaluation_metadata": evaluation_metadata  # Include original evaluation metadata for the evaluated model
            },
            "metrics": metrics
        }

        # Calculate and add judge performance averages if we have performance data
        if judge_performance_data:
            avg_first_token_latency = sum(p.get('first_token_latency_ms', 0) for p in judge_performance_data) / len(judge_performance_data)
            avg_generation_time = sum(p.get('generation_time_ms', 0) for p in judge_performance_data) / len(judge_performance_data)
            avg_throughput = sum(p.get('throughput_tokens_per_second', 0) for p in judge_performance_data) / len(judge_performance_data)

            metadata["judging_metadata"]["judge_performance_averages"] = {
                "avg_first_token_latency_ms": round(avg_first_token_latency, 2),
                "avg_generation_time_ms": round(avg_generation_time, 2),
                "avg_throughput_tokens_per_second": round(avg_throughput, 2),
                "samples_count": len(judge_performance_data)
            }

        # Save judged results with metadata
        final_output = {
            **metadata,
            "judged_predictions": judged_predictions
        }

        with open(output_filepath, "w") as f:
            json.dump(final_output, f, indent=4)

        self.logger.info("🎯 *** Metrics ***")
        self.logger.info(f"📊 Accuracy: {metrics['accuracy']}% +/- {metrics['confidence_interval']}%")
        self.logger.info(f"📏 Calibration Error: {metrics['calibration_error']}")
        self.logger.info(f"💾 Saved to: {output_filepath}")

        return output_filepath