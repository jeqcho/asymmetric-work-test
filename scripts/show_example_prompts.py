"""Script to show example prompts for haiku zero-shot and 5-shot with Presidio."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_labeled_dataset_with_presidio, get_5_shot_examples
from src.utils.prompt_builder import (
    build_zero_shot_prompt_with_presidio,
    build_5_shot_prompt_with_presidio
)
from src.config import RANDOM_SEED


def main():
    """Show example prompts."""
    # Load dataset with Presidio entities
    print("Loading dataset...")
    emails = load_labeled_dataset_with_presidio()
    
    # Find an email with PII for the example
    email_with_pii = next((e for e in emails if e.has_pii()), emails[0])
    
    print(f"\nUsing email ID {email_with_pii.id} (has_pii={email_with_pii.has_pii()})")
    print(f"Presidio found {len(email_with_pii.presidio_entities) if email_with_pii.presidio_entities else 0} entities")
    
    # Zero-shot prompt
    print("\n" + "="*80)
    print("HAIKU ZERO-SHOT PROMPT")
    print("="*80)
    zero_shot_prompt = build_zero_shot_prompt_with_presidio(email_with_pii)
    print(zero_shot_prompt)
    
    # 5-shot prompt
    print("\n" + "="*80)
    print("HAIKU 5-SHOT PROMPT")
    print("="*80)
    no_pii_examples, with_pii_examples = get_5_shot_examples(
        email_with_pii.id,
        emails,
        RANDOM_SEED
    )
    five_shot_prompt = build_5_shot_prompt_with_presidio(
        email_with_pii,
        no_pii_examples,
        with_pii_examples
    )
    print(five_shot_prompt)
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

