"""Pre-compute Presidio entity detections for all labeled emails.

This script runs Presidio V2 with confidence threshold 0.0 on all labeled emails
and saves the results to a JSON cache for use in hybrid LLM+Presidio detectors.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_labeled_dataset, Email
from src.detectors.presidio_detector import PresidioDetector
from src.config import RESULTS_DIR


def extract_presidio_entities(email: Email, detector: PresidioDetector) -> List[Dict[str, Any]]:
    """
    Extract Presidio entities from an email with text snippets.
    
    Args:
        email: Email to analyze
        detector: Presidio detector instance
        
    Returns:
        List of entity dictionaries with entity_type, score, start, end, and text
    """
    # Analyze message body for PII (using threshold 0.0, so we get all results)
    results = detector.analyzer.analyze(
        text=email.message_body,
        entities=detector.entities_to_detect,
        language="en"
    )
    
    # Convert to list of dictionaries with extracted text snippets
    entities = []
    for result in results:
        # Extract the text snippet from the message body
        text_snippet = email.message_body[result.start:result.end]
        
        entities.append({
            "entity_type": result.entity_type,
            "score": result.score,
            "start": result.start,
            "end": result.end,
            "text": text_snippet
        })
    
    return entities


def main():
    """Main execution function."""
    print("="*60)
    print("Pre-computing Presidio Entity Detections")
    print("="*60)
    
    # Load labeled dataset
    print("\nLoading labeled dataset...")
    labeled_emails = load_labeled_dataset()
    print(f"Loaded {len(labeled_emails)} emails")
    
    # Create Presidio detector with confidence threshold 0.0
    print("\nInitializing Presidio detector (confidence threshold 0.0)...")
    detector = PresidioDetector("presidio_v2_all", confidence_threshold=0.0)
    
    # Process all emails
    print(f"\nProcessing {len(labeled_emails)} emails...")
    presidio_cache: Dict[int, List[Dict[str, Any]]] = {}
    
    for i, email in enumerate(labeled_emails, 1):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(labeled_emails)} emails...")
        
        entities = extract_presidio_entities(email, detector)
        presidio_cache[email.id] = entities
    
    print(f"  ✓ Processed all {len(labeled_emails)} emails")
    
    # Save to JSON
    output_path = RESULTS_DIR / "presidio_entities_cache.json"
    print(f"\nSaving results to {output_path}...")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(presidio_cache, f, indent=2)
    
    # Print summary statistics
    total_entities = sum(len(entities) for entities in presidio_cache.values())
    emails_with_entities = sum(1 for entities in presidio_cache.values() if len(entities) > 0)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total emails processed: {len(labeled_emails)}")
    print(f"Total entities detected: {total_entities}")
    print(f"Emails with entities: {emails_with_entities}")
    print(f"Emails without entities: {len(labeled_emails) - emails_with_entities}")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

