"""Claude-based PII type classification detector."""

import time
from typing import List, Optional, Tuple
from anthropic import Anthropic

from pipeline.utils.data_loader import Email, get_5_shot_examples
from pipeline.utils.prompt_builder import build_5_shot_pii_classification_prompt_with_presidio
from pipeline.utils.pii_type_normalizer import normalize_pii_types
from pipeline.utils.cost_calculator import calculate_haiku_cost
from pipeline.config import (
    ANTHROPIC_API_KEY,
    HAIKU_MODEL,
    RANDOM_SEED,
    MAX_RETRIES,
    INITIAL_BACKOFF_SECONDS,
    BACKOFF_MULTIPLIER
)


class ClaudePIIClassificationDetector:
    """Claude-based PII type classification detector with Presidio augmentation."""

    def __init__(
        self,
        labeled_dataset: Optional[List[Email]] = None,
        use_few_shot: bool = True
    ):
        """
        Initialize Claude PII classification detector.

        Args:
            labeled_dataset: Labeled emails for 5-shot learning (should have presidio_entities populated)
            use_few_shot: Whether to use 5-shot examples
        """
        self.labeled_dataset = labeled_dataset
        self.use_few_shot = use_few_shot
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)

        # Validate parameters
        if use_few_shot and labeled_dataset is None:
            raise ValueError("labeled_dataset required for few-shot learning")

    def classify(self, email: Email) -> Tuple[List[str], float, float, int, int]:
        """
        Classify PII types in email using Claude with Presidio entities as context.
        
        Args:
            email: Email object to analyze (should have presidio_entities populated)
            
        Returns:
            Tuple of (pii_types, time_ms, cost_usd, input_tokens, output_tokens)
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
            # Zero-shot fallback (not implemented, but structure is here)
            raise ValueError("Zero-shot not implemented. Use few-shot with labeled dataset.")
        
        # Call Claude API with exponential backoff retry logic
        pii_types = []
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.messages.create(
                    model=HAIKU_MODEL,
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
                cost = calculate_haiku_cost(input_tokens, output_tokens)
                
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
        
        return pii_types, elapsed_ms, cost, input_tokens, output_tokens

