"""Run PII type classification experiment on all 8 LLM variants."""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_labeled_dataset, load_labeled_dataset_with_presidio
from src.detectors.claude_pii_classification_detector import (
    create_haiku_classification_detectors,
    create_sonnet_classification_detectors,
    create_haiku_classification_detectors_with_presidio,
    create_sonnet_classification_detectors_with_presidio
)
from src.evaluation.classification_test_harness import ClassificationTestHarness
from src.config import RESULTS_DIR


def generate_accuracy_summary(all_results):
    """
    Generate summary CSV with classification accuracies.
    
    Args:
        all_results: List of result dictionaries from evaluations
        
    Returns:
        Path to generated CSV file
    """
    # Collect exact match accuracies
    summary_rows = []
    
    for result in all_results:
        detector_name = result['detector_name']
        exact_match_acc = result['exact_match_accuracy']
        per_type_metrics = result['per_type_metrics']
        
        # Add row for exact match accuracy
        summary_rows.append({
            'detector_name': detector_name,
            'pii_type': 'OVERALL_EXACT_MATCH',
            'accuracy': exact_match_acc,
            'precision': None,
            'recall': None,
            'f1': None
        })
        
        # Add rows for each PII type
        for pii_type, metrics in per_type_metrics.items():
            summary_rows.append({
                'detector_name': detector_name,
                'pii_type': pii_type,
                'accuracy': None,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            })
    
    # Create DataFrame
    df = pd.DataFrame(summary_rows)
    
    # Save to CSV
    output_path = RESULTS_DIR / "pii_classification_accuracies.csv"
    df.to_csv(output_path, index=False)
    
    return output_path


def print_accuracy_summary(all_results):
    """Print summary of classification accuracies to console."""
    print("\n" + "="*80)
    print("PII TYPE CLASSIFICATION ACCURACY SUMMARY")
    print("="*80)
    
    for result in all_results:
        detector_name = result['detector_name']
        exact_match_acc = result['exact_match_accuracy']
        
        print(f"\n{detector_name}:")
        print(f"  Exact Match Accuracy: {exact_match_acc:.3f}")
        
        # Show top 5 PII types by F1 score
        per_type_metrics = result['per_type_metrics']
        sorted_types = sorted(
            per_type_metrics.items(),
            key=lambda x: x[1]['f1'],
            reverse=True
        )
        
        print(f"  Top PII Types by F1 Score:")
        for pii_type, metrics in sorted_types[:5]:
            if metrics['f1'] > 0:
                print(f"    {pii_type}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    
    print("\n" + "="*80)


def main():
    """Main execution function."""
    print("="*80)
    print("PII Type Classification Experiment")
    print("="*80)

    # Load labeled dataset
    print("\n[STEP 1/4] Loading 250-sample labeled dataset...")
    labeled_emails = load_labeled_dataset()
    print(f"  ✓ Loaded {len(labeled_emails)} emails")

    # Load labeled dataset with Presidio entities for Presidio variants
    print("\n[STEP 2/4] Loading Presidio entities cache...")
    try:
        labeled_emails_with_presidio = load_labeled_dataset_with_presidio()
        print(f"  ✓ Loaded {len(labeled_emails_with_presidio)} emails with Presidio entities")
    except FileNotFoundError as e:
        print(f"  ⚠ Warning: {e}")
        print("  Presidio variants will still run but without Presidio context")
        labeled_emails_with_presidio = labeled_emails

    # Create all detector instances
    print("\n[STEP 3/4] Initializing PII classification detectors...")

    print("  - Creating Haiku detectors (2 variants)...")
    haiku_detectors = create_haiku_classification_detectors(labeled_emails)

    print("  - Creating Sonnet detectors (2 variants)...")
    sonnet_detectors = create_sonnet_classification_detectors(labeled_emails)

    print("  - Creating Haiku detectors with Presidio (2 variants)...")
    haiku_detectors_presidio = create_haiku_classification_detectors_with_presidio(labeled_emails_with_presidio)

    print("  - Creating Sonnet detectors with Presidio (2 variants)...")
    sonnet_detectors_presidio = create_sonnet_classification_detectors_with_presidio(labeled_emails_with_presidio)

    # Combine all detectors
    all_detectors = (
        haiku_detectors +
        sonnet_detectors +
        haiku_detectors_presidio +
        sonnet_detectors_presidio
    )
    
    print(f"\n  ✓ Total detectors: {len(all_detectors)}")
    for detector in all_detectors:
        print(f"    - {detector.name}")

    # Create test harness
    print("\n[STEP 4/4] Initializing classification test harness...")
    harness = ClassificationTestHarness(all_detectors, labeled_emails)

    # Run evaluation
    print("\nStarting PII type classification evaluation...")
    all_results = harness.run_all_evaluations()

    # Generate summary
    print("\nGenerating accuracy summary...")
    summary_path = generate_accuracy_summary(all_results)
    
    # Print summary
    print_accuracy_summary(all_results)

    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"\n  Accuracy Summary CSV:")
    print(f"    {summary_path}")
    print(f"\n  Individual detector results:")
    for result in all_results:
        print(f"    results/{result['detector_name']}_classification_results.json")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

