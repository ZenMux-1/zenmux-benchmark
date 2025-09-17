"""HLE dataset handling."""

from datasets import load_dataset
from typing import List, Dict, Any, Optional


class HLEDataset:
    """Handler for HLE dataset operations."""

    SYSTEM_PROMPT = (
        "Your response should be in the following format:\n"
        "Explanation: {your explanation for your answer choice}\n"
        "Answer: {your chosen answer}\n"
        "Confidence: {your confidence score between 0% and 100% for your answer}"
    )

    def __init__(self, dataset_name: str = "cais/hle", split: str = "test"):
        self.dataset_name = dataset_name
        self.split = split
        self._dataset = None

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load and return the HLE dataset."""
        if self._dataset is None:
            dataset = load_dataset(self.dataset_name, split=self.split).to_dict()
            # Convert to list of dictionaries for easier handling
            self._dataset = [
                dict(zip(dataset.keys(), values))
                for values in zip(*dataset.values())
            ]
        return self._dataset

    def filter_text_only(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter questions to only include text-only (no images)."""
        text_only_questions = []
        for question in questions:
            # Check if the question has an image
            if not question.get('image') or question['image'] == "":
                text_only_questions.append(question)
        return text_only_questions

    def get_questions(
        self,
        text_only: bool = False,
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get questions with optional filtering."""
        questions = self.load_dataset()

        if text_only:
            questions = self.filter_text_only(questions)

        if max_samples is not None:
            questions = questions[:max_samples]

        return questions

    def format_message(self, question: Dict[str, Any], for_o1: bool = False) -> List[Dict[str, Any]]:
        """Format a question into OpenAI message format."""
        question_text = question['question']

        text_content = {"type": "text", "text": question_text}
        content = [text_content]

        # Add image if present and not empty
        if question.get('image') and question['image'] != "":
            image_content = {
                "type": "image_url",
                "image_url": {"url": question['image']}
            }
            content.append(image_content)

        # o1 models don't support system prompts
        system_role = "user" if for_o1 else "system"
        messages = [
            {"role": system_role, "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]

        return messages

    def get_total_count(self) -> int:
        """Get total number of questions in dataset."""
        return len(self.load_dataset())

    def get_text_only_count(self) -> int:
        """Get number of text-only questions."""
        return len(self.filter_text_only(self.load_dataset()))