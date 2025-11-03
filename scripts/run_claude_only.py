"""Run evaluation for Claude detectors only (skipping Presidio)."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_labeled_dataset
from src.detectors.claude_detector import create_haiku_detectors, create_sonnet_detectors
from src.evaluation.test_harness import TestHarness
from src.evaluation.visualizations import generate_all_visualizations
import json


def load_existing_results(detector_names):
    """Load existing results for any detectors that have completed."""
    from src.config import RESULTS_DIR

    existing_results = []

    for name in detector_names:
        result_path = RESULTS_DIR / f"{name}_results.json"
        if result_path.exists():
            with open(result_path, 'r') as f:
                existing_results.append(json.load(f))
            print(f"  ✓ Loaded existing results: {name}")

    return existing_results


def main():
    """Main execution function."""
    from src.config import RESULTS_DIR

    print("="*60)
    print("PII Detection Evaluation - Claude Detectors Only")
    print("="*60)

    # Load labeled dataset
    print("\n[STEP 1/5] Loading 250-sample labeled dataset...")
    labeled_emails = load_labeled_dataset()
    print(f"  ✓ Loaded {len(labeled_emails)} emails")

    # Create Claude detector instances
    print("\n[STEP 2/5] Initializing Claude detectors...")

    print("  - Creating Haiku detectors (2 variants)...")
    haiku_detectors = create_haiku_detectors(labeled_emails)

    print("  - Creating Sonnet detectors (2 variants)...")
    sonnet_detectors = create_sonnet_detectors(labeled_emails)

    # Combine Claude detectors
    claude_detectors = haiku_detectors + sonnet_detectors
    print(f"  ✓ Total Claude detectors: {len(claude_detectors)}")
    for detector in claude_detectors:
        print(f"    - {detector.name}")

    # Check which detectors already have results
    print("\n[STEP 3/5] Checking for existing results...")
    all_detector_names = [d.name for d in claude_detectors]
    claude_results = []
    detectors_to_run = []

    for detector in claude_detectors:
        result_path = RESULTS_DIR / f"{detector.name}_results.json"
        if result_path.exists():
            print(f"  ✓ Skipping {detector.name} (already completed)")
            with open(result_path, 'r') as f:
                claude_results.append(json.load(f))
        else:
            print(f"  ⏳ Will run {detector.name}")
            detectors_to_run.append(detector)

    # Run evaluation for detectors that need it
    if detectors_to_run:
        print(f"\n[STEP 4/5] Running evaluation for {len(detectors_to_run)} detector(s)...")
        harness = TestHarness(detectors_to_run, labeled_emails)
        new_results = harness.run_all_evaluations()
        claude_results.extend(new_results)
    else:
        print(f"\n[STEP 4/5] All Claude detectors already completed - skipping evaluation")

    # Load existing Presidio results
    print("\n[STEP 5/5] Loading existing Presidio results...")
    presidio_names = ['presidio_lax', 'presidio_moderate', 'presidio_strict']
    presidio_results = load_existing_results(presidio_names)

    # Combine all results
    all_results = presidio_results + claude_results

    # Generate visualizations
    print("\n[FINAL] Generating comparison CSV and visualizations...")
    viz_paths = generate_all_visualizations(all_results)

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
    print(f"    {viz_paths['classification_rates']}")
    print(f"    {viz_paths['confusion_matrices']}")
    print(f"\n  Individual detector results:")
    for detector in claude_detectors:
        print(f"    results/{detector.name}_results.json")
    print(f"\n  False positive/negative CSVs:")
    print(f"    results/error_analysis/")

    print("\n" + "="*60)
    print("Next steps:")
    print("  1. Review comparison CSV for overall metrics")
    print("  2. Check visualizations for insights")
    print("  3. Compare Claude detectors with Presidio")
    print("  4. Consider the cost/accuracy/speed trade-offs")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
