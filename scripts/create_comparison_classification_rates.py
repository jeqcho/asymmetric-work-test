"""Create a classification rates comparison visualization for original, conservative, and hybrid variants."""

import sys
import json
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR, VISUALIZATIONS_DIR


def load_results(detector_names: List[str]) -> List[Dict]:
    """Load results from JSON files. Returns (results, result_variant_map)."""
    results = []
    result_variant_map = {}  # Maps (detector_name, filename) -> variant_type
    
    for name in detector_names:
        # Handle different naming patterns
        if '_conservative' in name:
            # Conservative files are named: {base}_results_conservative.json
            base = name.replace('_conservative', '')
            json_path = RESULTS_DIR / f"{base}_results_conservative.json"
            variant = 'conservative'
        elif '_with_presidio' in name:
            # Hybrid files are named: {name}_results.json
            json_path = RESULTS_DIR / f"{name}_results.json"
            variant = 'hybrid'
        else:
            # Regular files are named: {name}_results.json
            json_path = RESULTS_DIR / f"{name}_results.json"
            variant = 'original'
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                detector_name = data['detector_name']
                results.append(data)
                # Use filename as part of the key to handle duplicate detector_names
                # Store variant info in the result dict itself
                data['_variant_type'] = variant
                data['_filename'] = json_path.name
        else:
            print(f"Warning: {json_path} not found, skipping...")
    
    return results


def calculate_classification_rates(result: Dict) -> Dict[str, float]:
    """Calculate FPR, FNR, TPR, TNR from result metrics."""
    metrics = result['metrics']
    tp = metrics['TP']
    fp = metrics['FP']
    tn = metrics['TN']
    fn = metrics['FN']
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'FPR': fpr,
        'FNR': fnr,
        'TPR': tpr,
        'TNR': tnr
    }


def plot_comparison_classification_rates():
    """Create grouped comparison visualization."""
    # Define detector variants to compare
    detectors = [
        'haiku_zeroshot',
        'haiku_zeroshot_conservative',
        'haiku_zeroshot_with_presidio',
        'haiku_5shot',
        'haiku_5shot_conservative',
        'haiku_5shot_with_presidio',
        'sonnet_zeroshot',
        'sonnet_zeroshot_conservative',
        'sonnet_zeroshot_with_presidio',
        'sonnet_5shot',
        'sonnet_5shot_conservative',
        'sonnet_5shot_with_presidio',
    ]
    
    # Load all results
    results = load_results(detectors)
    
    if not results:
        print("Error: No results found!")
        return
    
    # Group results by base detector name using the variant type stored in result
    grouped = {}
    for result in results:
        name = result['detector_name']
        
        # Get variant type from the metadata we stored (this is the key!)
        variant = result.get('_variant_type', 'original')
        
        # Determine base name
        if '_with_presidio' in name:
            base = name.replace('_with_presidio', '')
            variant = 'hybrid'  # Override to ensure hybrid is correct
        else:
            base = name
            # Keep the variant from metadata (conservative vs original)
            # Don't override it here since they have the same base name
        
        if base not in grouped:
            grouped[base] = {}
        
        # Store the result with its variant type
        # Multiple results can share the same base but have different variants
        grouped[base][variant] = result
    
    # Prepare data for plotting
    base_names = sorted(grouped.keys())
    # Order: conservative, original, hybrid
    variants = ['conservative', 'original', 'hybrid']
    variant_labels = ['Conservative', 'Original', 'Hybrid (with Presidio)']
    rate_types = ['FPR', 'FNR', 'TPR', 'TNR']
    rate_labels = ['False Positive Rate', 'False Negative Rate', 
                   'True Positive Rate (Recall)', 'True Negative Rate (Specificity)']
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    # Color palette for variants
    colors = {
        'original': '#2ca02c',      # Green
        'conservative': '#ff7f0e',   # Orange
        'hybrid': '#1f77b4'          # Blue
    }
    
    # Plot each rate type
    for rate_idx, (rate_type, rate_label) in enumerate(zip(rate_types, rate_labels)):
        ax = axes[rate_idx]
        
        # Calculate positions for grouped bars
        x = np.arange(len(base_names))
        width = 0.25  # Width of each bar group
        
        # Collect data for each variant
        variant_data = {variant: [] for variant in variants}
        
        for base in base_names:
            for variant in variants:
                if variant in grouped[base]:
                    rates = calculate_classification_rates(grouped[base][variant])
                    variant_data[variant].append(rates[rate_type])
                else:
                    variant_data[variant].append(None)
        
        # Plot bars for each variant (only for variants that have data)
        available_variants = []
        available_labels_list = []
        available_colors_list = []
        
        for i, variant in enumerate(variants):
            values = variant_data[variant]
            # Check if this variant has any data
            if any(v is not None for v in values):
                available_variants.append(variant)
                available_labels_list.append(variant_labels[i])
                available_colors_list.append(colors[variant])
        
        # Calculate width based on number of available variants
        n_variants = len(available_variants)
        if n_variants > 0:
            bar_width = 0.25 if n_variants <= 3 else 0.2
            total_width = bar_width * n_variants
            start_offset = -total_width / 2 + bar_width / 2
            
            for i, variant in enumerate(available_variants):
                offset = start_offset + i * bar_width
                values = variant_data[variant]
                
                # Convert None to 0 for plotting (bars won't show but won't error)
                plot_values = [v if v is not None else 0 for v in values]
                
                bars = ax.bar(x + offset, plot_values, bar_width, 
                             label=available_labels_list[i], 
                             color=available_colors_list[i], alpha=0.8, 
                             edgecolor='black', linewidth=1)
                
                # Add value labels on bars (only for non-zero values that aren't None)
                for j, (bar, val) in enumerate(zip(bars, values)):
                    if val is not None and val > 0:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Formatting
        ax.set_xlabel('Detector', fontsize=11, fontweight='bold')
        ax.set_ylabel(rate_label, fontsize=11, fontweight='bold')
        
        # Determine title based on rate type
        if rate_type in ['FPR', 'FNR']:
            title = f'{rate_label} (Lower is Better)'
        else:
            title = f'{rate_label} (Higher is Better)'
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace('_', '\n') for name in base_names], 
                          rotation=0, ha='center', fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('Classification Rates Comparison: Original vs Conservative vs Hybrid (with Presidio)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = VISUALIZATIONS_DIR / "classification_rates_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nClassification rates comparison plot saved to: {output_path}")
    return str(output_path)


def main():
    """Main execution function."""
    print("="*60)
    print("Creating Classification Rates Comparison Visualization")
    print("="*60)
    
    output_path = plot_comparison_classification_rates()
    
    print(f"\n✓ Visualization complete!")
    print(f"  Saved to: {output_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

