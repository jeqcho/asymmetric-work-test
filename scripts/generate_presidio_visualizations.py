"""Generate visualizations combining all results (V1, V2, Claude)."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR
from src.utils.data_loader import load_labeled_dataset
from src.evaluation.visualizations import (
    generate_comparison_csv,
    plot_tnr_fnr_comparison,
    plot_presidio_v2_threshold_analysis,
    generate_all_visualizations
)


def load_all_results():
    """Load all result JSON files (V1, V2, and Claude)."""
    result_files = [
        # Presidio V1
        "presidio_lax_results.json",
        "presidio_moderate_results.json",
        "presidio_strict_results.json",
        # Presidio V2
        "presidio_v2_lax_results.json",
        "presidio_v2_moderate_results.json",
        "presidio_v2_strict_results.json",
        # Claude models
        "haiku_zeroshot_results.json",
        "haiku_5shot_results.json",
        "sonnet_zeroshot_results.json",
        "sonnet_5shot_results.json",
    ]

    results = []
    for filename in result_files:
        filepath = RESULTS_DIR / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                results.append(json.load(f))
            print(f"  ✓ Loaded {filename}")
        else:
            print(f"  ⚠ Missing {filename}")

    return results


def main():
    """Main execution function."""
    print("="*60)
    print("Generating Presidio V2 Visualizations")
    print("="*60)

    # Load all results
    print("\nLoading all result files...")
    all_results = load_all_results()
    print(f"Loaded {len(all_results)} result files")

    # Load labeled dataset for threshold analysis
    print("\nLoading labeled dataset...")
    labeled_emails = load_labeled_dataset()

    # Generate comparison CSV with all results
    print("\n1. Generating comparison CSV...")
    comparison_csv = generate_comparison_csv(all_results)
    print(f"  ✓ Saved to: {comparison_csv}")

    # Generate all standard visualizations
    print("\n2. Generating standard visualizations...")
    viz_paths = generate_all_visualizations(all_results)

    # Generate TNR/FNR comparison plot
    print("\n3. Generating TNR/FNR comparison plot...")
    tnr_fnr_path = plot_tnr_fnr_comparison(all_results)
    print(f"  ✓ Saved to: {tnr_fnr_path}")

    # Generate threshold sensitivity analysis
    print("\n4. Generating Presidio V2 threshold sensitivity analysis...")
    threshold_path = plot_presidio_v2_threshold_analysis(all_results, labeled_emails)
    print(f"  ✓ Saved to: {threshold_path}")

    # Print summary
    print("\n" + "="*60)
    print("VISUALIZATION GENERATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print(f"\n  Comparison CSV:")
    print(f"    {comparison_csv}")
    print(f"\n  New Visualizations:")
    print(f"    {tnr_fnr_path}")
    print(f"    {threshold_path}")
    print(f"\n  Standard Visualizations:")
    for name, path in viz_paths.items():
        if name != 'comparison_csv':
            print(f"    {path}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
