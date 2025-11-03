"""Analyze missed PII types and show representative examples."""

import sys
import json
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_labeled_dataset
from src.config import RESULTS_DIR


def load_all_classification_results():
    """Load all classification result files."""
    detector_names = [
        "haiku_zeroshot_classification",
        "haiku_5shot_classification",
        "sonnet_zeroshot_classification",
        "sonnet_5shot_classification",
        "haiku_zeroshot_classification_with_presidio",
        "haiku_5shot_classification_with_presidio",
        "sonnet_zeroshot_classification_with_presidio",
        "sonnet_5shot_classification_with_presidio"
    ]
    
    results = {}
    for detector_name in detector_names:
        result_path = RESULTS_DIR / f"{detector_name}_classification_results.json"
        if result_path.exists():
            with open(result_path, 'r') as f:
                results[detector_name] = json.load(f)
        else:
            print(f"  ⚠ Missing {result_path}")
    
    return results


def find_missed_cases(target_pii_types, emails, all_results):
    """
    Find emails where target PII types are in ground truth but missed by all classifiers.
    
    Args:
        target_pii_types: List of PII types to check (e.g., ['address', 'drivers license', 'full name'])
        emails: List of Email objects
        all_results: Dictionary of classification results keyed by detector name
        
    Returns:
        Dictionary mapping PII type -> list of email examples
    """
    # Create email lookup by ID
    email_dict = {email.id: email for email in emails}
    
    # Find missed cases for each target PII type
    missed_cases = {pii_type: [] for pii_type in target_pii_types}
    
    for email in emails:
        gt_pii_types = [p.lower().strip() for p in email.get_pii_types()]
        
        # Check each target PII type
        for target_type in target_pii_types:
            target_type_lower = target_type.lower().strip()
            
            # If this PII type is in ground truth
            if target_type_lower in gt_pii_types:
                # Check if it was missed by ALL classifiers
                missed_by_all = True
                
                for detector_name, result in all_results.items():
                    # Find prediction for this email
                    for pred in result['predictions']:
                        if pred['email_id'] == email.id:
                            predicted_types = [p.lower().strip() for p in pred['predicted_pii_types']]
                            
                            # If any detector got it, it's not missed by all
                            if target_type_lower in predicted_types:
                                missed_by_all = False
                                break
                    
                    if not missed_by_all:
                        break
                
                # If missed by all, add to examples
                if missed_by_all:
                    missed_cases[target_type].append(email)
    
    return missed_cases


def format_email_for_display(email):
    """Format email for readable display."""
    return f"""
Email ID: {email.id}
Subject: {email.subject}
From: {email.from_addr}
To: {email.to_addr}
Ground Truth PII Types: {email.data_elements if email.data_elements else 'None'}
Message Body:
{email.message_body}
{'='*80}
"""


def main():
    """Main execution function."""
    print("="*80)
    print("Analyzing Missed PII Types")
    print("="*80)
    
    # Target PII types to analyze
    target_pii_types = ['address', 'drivers license', 'full name']
    
    # Load data
    print("\nLoading labeled dataset...")
    emails = load_labeled_dataset()
    print(f"  ✓ Loaded {len(emails)} emails")
    
    print("\nLoading classification results...")
    all_results = load_all_classification_results()
    print(f"  ✓ Loaded {len(all_results)} result files")
    
    # Find missed cases
    print(f"\nFinding emails where {', '.join(target_pii_types)} are missed by ALL classifiers...")
    missed_cases = find_missed_cases(target_pii_types, emails, all_results)
    
    # Display results
    print("\n" + "="*80)
    print("REPRESENTATIVE EXAMPLES")
    print("="*80)
    
    for pii_type in target_pii_types:
        cases = missed_cases[pii_type]
        print(f"\n\n{'='*80}")
        print(f"PII TYPE: {pii_type.upper()}")
        print(f"Total emails missed by all classifiers: {len(cases)}")
        print(f"{'='*80}")
        
        if len(cases) == 0:
            print(f"\n✓ No cases found where '{pii_type}' was missed by all classifiers.")
        else:
            # Show up to 5 representative examples
            examples_to_show = min(5, len(cases))
            print(f"\nShowing {examples_to_show} representative example(s):\n")
            
            for i, email in enumerate(cases[:examples_to_show], 1):
                print(f"\n--- Example {i} of {examples_to_show} ---")
                print(format_email_for_display(email))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for pii_type in target_pii_types:
        count = len(missed_cases[pii_type])
        print(f"  {pii_type}: {count} email(s) missed by all classifiers")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

