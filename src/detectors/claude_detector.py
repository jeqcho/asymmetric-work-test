"""Claude-based PII detectors (Haiku and Sonnet, zero-shot and 5-shot)."""

import time
from typing import List, Optional
from anthropic import Anthropic

from src.detectors.base import BaseDetector, DetectionResult
from src.utils.data_loader import Email, get_5_shot_examples
from src.utils.prompt_builder import build_zero_shot_prompt, build_5_shot_prompt
from src.utils.cost_calculator import calculate_haiku_cost, calculate_sonnet_cost
from src.config import (
    ANTHROPIC_API_KEY,
    HAIKU_MODEL,
    SONNET_MODEL,
    RANDOM_SEED,
    MAX_RETRIES,
    INITIAL_BACKOFF_SECONDS,
    BACKOFF_MULTIPLIER
)


class ClaudeDetector(BaseDetector):
    """Base class for Claude-based PII detectors."""

    def __init__(
        self,
        name: str,
        model: str,
        use_few_shot: bool = False,
        labeled_dataset: Optional[List[Email]] = None
    ):
        """
        Initialize Claude detector.

        Args:
            name: Detector instance name
            model: Claude model ID (haiku or sonnet)
            use_few_shot: Whether to use 5-shot examples
            labeled_dataset: Required if use_few_shot=True
        """
        super().__init__(name)
        self.model = model
        self.use_few_shot = use_few_shot
        self.labeled_dataset = labeled_dataset
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)

        # Validate parameters
        if use_few_shot and labeled_dataset is None:
            raise ValueError("labeled_dataset required for few-shot learning")

        # Determine cost calculator based on model
        if "haiku" in model.lower():
            self.cost_calculator = calculate_haiku_cost
        elif "sonnet" in model.lower():
            self.cost_calculator = calculate_sonnet_cost
        else:
            raise ValueError(f"Unknown model: {model}")

    def detect(self, email: Email) -> DetectionResult:
        """
        Detect PII in email using Claude.

        Args:
            email: Email object to analyze

        Returns:
            DetectionResult with binary prediction
        """
        start_time = time.time()

        # Build prompt
        if self.use_few_shot:
            # Get 5-shot examples (excluding current email)
            no_pii_examples, with_pii_examples = get_5_shot_examples(
                email.id,
                self.labeled_dataset,
                RANDOM_SEED
            )
            prompt = build_5_shot_prompt(email, no_pii_examples, with_pii_examples)
        else:
            prompt = build_zero_shot_prompt(email)

        # Call Claude API with exponential backoff retry logic
        has_pii = False
        input_tokens = 0
        output_tokens = 0
        cost = 0.0

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=10,  # We only need "YES" or "NO"
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                # Extract response
                response_text = response.content[0].text.strip().upper()

                # Parse response (expecting "YES" or "NO")
                has_pii = "YES" in response_text

                # Get token usage
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

                # Calculate cost
                cost = self.cost_calculator(input_tokens, output_tokens)

                # Success - break out of retry loop
                break

            except Exception as e:
                # If this is the last attempt, don't wait
                if attempt < MAX_RETRIES:
                    # Calculate exponential backoff wait time
                    wait_time = INITIAL_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER ** attempt)
                    print(f"Error calling Claude API for email {email.id} (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    print(f"Error calling Claude API for email {email.id} after {MAX_RETRIES + 1} attempts: {e}")
                    has_pii = False
                    input_tokens = 0
                    output_tokens = 0
                    cost = 0.0

        # Calculate elapsed time
        elapsed_ms = (time.time() - start_time) * 1000

        return DetectionResult(
            email_id=email.id,
            has_pii=has_pii,
            confidence=None,  # Claude doesn't return confidence scores
            time_ms=elapsed_ms,
            cost_usd=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )


def create_haiku_detectors(labeled_dataset: List[Email]) -> List[ClaudeDetector]:
    """
    Create Haiku detector instances (zero-shot and 5-shot).

    Args:
        labeled_dataset: Labeled emails for 5-shot learning

    Returns:
        List of [haiku_zeroshot, haiku_5shot]
    """
    return [
        ClaudeDetector(
            name="haiku_zeroshot",
            model=HAIKU_MODEL,
            use_few_shot=False
        ),
        ClaudeDetector(
            name="haiku_5shot",
            model=HAIKU_MODEL,
            use_few_shot=True,
            labeled_dataset=labeled_dataset
        )
    ]


def create_sonnet_detectors(labeled_dataset: List[Email]) -> List[ClaudeDetector]:
    """
    Create Sonnet detector instances (zero-shot and 5-shot).

    Args:
        labeled_dataset: Labeled emails for 5-shot learning

    Returns:
        List of [sonnet_zeroshot, sonnet_5shot]
    """
    return [
        ClaudeDetector(
            name="sonnet_zeroshot",
            model=SONNET_MODEL,
            use_few_shot=False
        ),
        ClaudeDetector(
            name="sonnet_5shot",
            model=SONNET_MODEL,
            use_few_shot=True,
            labeled_dataset=labeled_dataset
        )
    ]
