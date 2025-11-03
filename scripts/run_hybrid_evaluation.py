"""Main script to run the hybrid LLM+Presidio detector evaluation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_labeled_dataset_with_presidio
from src.detectors.claude_detector import (
    create_haiku_detectors_with_presidio,
    create_sonnet_detectors_with_presidio
)
from src.evaluation.test_harness import TestHarness
from src.evaluation.visualizations import generate_all_visualizations


def main():
    """Main execution function."""
    print("="*60)
    print("Hybrid LLM + Presidio Detector Evaluation")
    print("="*60)
    
    # Load labeled dataset with Presidio entities attached
    print("\nLoading labeled dataset with Presidio entities...")
    try:
        labeled_emails = load_labeled_dataset_with_presidio()
        print(f"Loaded {len(labeled_emails)} emails with Presidio entities")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease run scripts/precompute_presidio_outputs.py first to generate the Presidio cache.")
        return
    
    # Create all detector instances
    print("\nInitializing hybrid detectors...")
    
    print("  - Creating Haiku detectors (2 variants)...")
    haiku_detectors = create_haiku_detectors_with_presidio(labeled_emails)
    
    print("  - Creating Sonnet detectors (2 variants)...")
    sonnet_detectors = create_sonnet_detectors_with_presidio(labeled_emails)
    
    # Combine all detectors
    all_detectors = haiku_detectors + sonnet_detectors
    print(f"\nTotal detectors: {len(all_detectors)}")
    for detector in all_detectors:
        print(f"  - {detector.name}")
    
    # Create test harness
    print("\nInitializing test harness...")
    harness = TestHarness(all_detectors, labeled_emails)
    
    # Run evaluation
    print("\nStarting evaluation...")
    results = harness.run_all_evaluations()
    
    # Note: Visualizations generation is optional - the comparison CSV will be generated
    # but it will include all detectors. We can filter later if needed.
    print("\nGenerating comparison CSV...")
    viz_paths = generate_all_visualizations(results)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print(f"\n  Comparison CSV:")
    print(f"    {viz_paths['comparison_csv']}")
    print(f"\n  Visualizations:")
    print(f"    {viz_paths['f1_scores']}")
    print(f"    {viz_paths['cost_vs_f1']}")
    print(f"    {viz_paths['time_vs_f1']}")
    print(f"    {viz_paths['confusion_matrices']}")
    print(f"\n  Individual detector results:")
    for detector in all_detectors:
        print(f"    results/{detector.name}_results.json")
    print(f"\n  False positive/negative CSVs:")
    print(f"    results/error_analysis/")
    
    print("\n" + "="*60)
    print("Next steps:")
    print("  1. Review comparison CSV for overall metrics")
    print("  2. Check visualizations for insights")
    print("  3. Compare with baseline detectors (run scripts/run_evaluation.py)")
    print("  4. Review false positives/negatives for error analysis")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

