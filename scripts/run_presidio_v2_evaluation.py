"""Run evaluation for Presidio V2 detectors only."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_labeled_dataset
from src.detectors.presidio_detector import create_presidio_detectors
from src.evaluation.test_harness import TestHarness


def main():
    """Main execution function for Presidio V2 only."""
    print("="*60)
    print("Presidio V2 Evaluation")
    print("="*60)

    # Load labeled dataset
    print("\nLoading 250-sample labeled dataset...")
    labeled_emails = load_labeled_dataset()
    print(f"Loaded {len(labeled_emails)} emails")

    # Create only Presidio V2 detector instances
    print("\nInitializing Presidio V2 detectors...")
    presidio_v2_detectors = create_presidio_detectors()

    print(f"\nTotal detectors: {len(presidio_v2_detectors)}")
    for detector in presidio_v2_detectors:
        print(f"  - {detector.name}")

    # Create test harness
    print("\nInitializing test harness...")
    harness = TestHarness(presidio_v2_detectors, labeled_emails)

    # Run evaluation
    print("\nStarting evaluation...")
    results = harness.run_all_evaluations()

    # Print summary
    print("\n" + "="*60)
    print("PRESIDIO V2 EVALUATION COMPLETE!")
    print("="*60)
    print(f"\n  Individual detector results:")
    for detector in presidio_v2_detectors:
        print(f"    results/{detector.name}_results.json")

    print("\n" + "="*60)
    print("Next steps:")
    print("  1. Run visualization script to update comparison CSV and plots")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
