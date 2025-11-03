"""Generate visualizations for PII type classification results."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.classification_visualizations import generate_all_classification_visualizations


def main():
    """Main execution function."""
    print("="*80)
    print("PII Type Classification Visualization Generation")
    print("="*80)
    
    print("\nGenerating all classification visualizations...")
    paths = generate_all_classification_visualizations()
    
    print("\n" + "="*80)
    print("VISUALIZATION GENERATION COMPLETE!")
    print("="*80)
    print("\nGenerated visualizations:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

