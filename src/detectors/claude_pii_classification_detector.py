"""Claude-based PII type classification detectors."""

import time
from typing import List, Optional
from anthropic import Anthropic

from src.detectors.base import BaseDetector, PIIClassificationResult
from src.utils.data_loader import Email, get_5_shot_examples
from src.utils.pii_classification_prompt_builder import (
    build_zero_shot_pii_classification_prompt,
    build_5_shot_pii_classification_prompt,
    build_zero_shot_pii_classification_prompt_with_presidio,
    build_5_shot_pii_classification_prompt_with_presidio
)
from src.utils.pii_type_normalizer import normalize_pii_types
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


class ClaudePIIClassificationDetector(BaseDetector):
    """Base class for Claude-based PII type classification detectors."""

    def __init__(
        self,
        name: str,
        model: str,
        use_few_shot: bool = False,
        labeled_dataset: Optional[List[Email]] = None
    ):
        """
        Initialize Claude PII classification detector.

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

    def detect(self, email: Email) -> PIIClassificationResult:
        """
        Classify PII types in email using Claude.

        Args:
            email: Email object to analyze

        Returns:
            PIIClassificationResult with predicted PII types
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
            prompt = build_5_shot_pii_classification_prompt(email, no_pii_examples, with_pii_examples)
        else:
            prompt = build_zero_shot_pii_classification_prompt(email)

        # Call Claude API with exponential backoff retry logic
        pii_types = []
        input_tokens = 0
        output_tokens = 0
        cost = 0.0

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=200,  # Need more tokens for comma-separated lists
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                # Extract response
                response_text = response.content[0].text.strip()

                # Parse response to extract PII types
                pii_types = normalize_pii_types(response_text)

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
                    pii_types = []
                    input_tokens = 0
                    output_tokens = 0
                    cost = 0.0

        # Calculate elapsed time
        elapsed_ms = (time.time() - start_time) * 1000

        return PIIClassificationResult(
            email_id=email.id,
            pii_types=pii_types,
            time_ms=elapsed_ms,
            cost_usd=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )


def create_haiku_classification_detectors(labeled_dataset: List[Email]) -> List[ClaudePIIClassificationDetector]:
    """
    Create Haiku PII classification detector instances (zero-shot and 5-shot).

    Args:
        labeled_dataset: Labeled emails for 5-shot learning

    Returns:
        List of [haiku_zeroshot_classification, haiku_5shot_classification]
    """
    return [
        ClaudePIIClassificationDetector(
            name="haiku_zeroshot_classification",
            model=HAIKU_MODEL,
            use_few_shot=False
        ),
        ClaudePIIClassificationDetector(
            name="haiku_5shot_classification",
            model=HAIKU_MODEL,
            use_few_shot=True,
            labeled_dataset=labeled_dataset
        )
    ]


def create_sonnet_classification_detectors(labeled_dataset: List[Email]) -> List[ClaudePIIClassificationDetector]:
    """
    Create Sonnet PII classification detector instances (zero-shot and 5-shot).

    Args:
        labeled_dataset: Labeled emails for 5-shot learning

    Returns:
        List of [sonnet_zeroshot_classification, sonnet_5shot_classification]
    """
    return [
        ClaudePIIClassificationDetector(
            name="sonnet_zeroshot_classification",
            model=SONNET_MODEL,
            use_few_shot=False
        ),
        ClaudePIIClassificationDetector(
            name="sonnet_5shot_classification",
            model=SONNET_MODEL,
            use_few_shot=True,
            labeled_dataset=labeled_dataset
        )
    ]


class ClaudePIIClassificationDetectorWithPresidio(ClaudePIIClassificationDetector):
    """Claude-based PII type classification detector that uses Presidio entity output as context."""
    
    def detect(self, email: Email) -> PIIClassificationResult:
        """
        Classify PII types in email using Claude with Presidio entities as context.
        
        This method assumes the email has presidio_entities populated.
        If not, it will work but Presidio output will be "None".
        
        Args:
            email: Email object to analyze (should have presidio_entities populated)
            
        Returns:
            PIIClassificationResult with predicted PII types
        """
        start_time = time.time()
        
        # Ensure presidio_entities is set (default to empty list if None)
        if email.presidio_entities is None:
            email.presidio_entities = []
        
        # Build prompt with Presidio output
        if self.use_few_shot:
            # Get 5-shot examples (excluding current email)
            no_pii_examples, with_pii_examples = get_5_shot_examples(
                email.id,
                self.labeled_dataset,
                RANDOM_SEED
            )
            prompt = build_5_shot_pii_classification_prompt_with_presidio(email, no_pii_examples, with_pii_examples)
        else:
            prompt = build_zero_shot_pii_classification_prompt_with_presidio(email)
        
        # Call Claude API with exponential backoff retry logic
        pii_types = []
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=200,  # Need more tokens for comma-separated lists
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Extract response
                response_text = response.content[0].text.strip()
                
                # Parse response to extract PII types
                pii_types = normalize_pii_types(response_text)
                
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
                    pii_types = []
                    input_tokens = 0
                    output_tokens = 0
                    cost = 0.0
        
        # Calculate elapsed time
        elapsed_ms = (time.time() - start_time) * 1000
        
        return PIIClassificationResult(
            email_id=email.id,
            pii_types=pii_types,
            time_ms=elapsed_ms,
            cost_usd=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )


def create_haiku_classification_detectors_with_presidio(labeled_dataset: List[Email]) -> List[ClaudePIIClassificationDetectorWithPresidio]:
    """
    Create Haiku PII classification detector instances with Presidio support (zero-shot and 5-shot).
    
    Args:
        labeled_dataset: Labeled emails for 5-shot learning (should have presidio_entities populated)
        
    Returns:
        List of [haiku_zeroshot_classification_with_presidio, haiku_5shot_classification_with_presidio]
    """
    return [
        ClaudePIIClassificationDetectorWithPresidio(
            name="haiku_zeroshot_classification_with_presidio",
            model=HAIKU_MODEL,
            use_few_shot=False
        ),
        ClaudePIIClassificationDetectorWithPresidio(
            name="haiku_5shot_classification_with_presidio",
            model=HAIKU_MODEL,
            use_few_shot=True,
            labeled_dataset=labeled_dataset
        )
    ]


def create_sonnet_classification_detectors_with_presidio(labeled_dataset: List[Email]) -> List[ClaudePIIClassificationDetectorWithPresidio]:
    """
    Create Sonnet PII classification detector instances with Presidio support (zero-shot and 5-shot).
    
    Args:
        labeled_dataset: Labeled emails for 5-shot learning (should have presidio_entities populated)
        
    Returns:
        List of [sonnet_zeroshot_classification_with_presidio, sonnet_5shot_classification_with_presidio]
    """
    return [
        ClaudePIIClassificationDetectorWithPresidio(
            name="sonnet_zeroshot_classification_with_presidio",
            model=SONNET_MODEL,
            use_few_shot=False
        ),
        ClaudePIIClassificationDetectorWithPresidio(
            name="sonnet_5shot_classification_with_presidio",
            model=SONNET_MODEL,
            use_few_shot=True,
            labeled_dataset=labeled_dataset
        )
    ]

