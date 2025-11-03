"""Generate visualizations for PII type classification results."""

import json
from typing import List, Dict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, VISUALIZATIONS_DIR


def load_classification_results(detector_names: List[str]) -> List[Dict]:
    """
    Load classification results from JSON files.
    
    Args:
        detector_names: List of detector names to load
        
    Returns:
        List of result dictionaries
    """
    results = []
    for detector_name in detector_names:
        result_path = RESULTS_DIR / f"{detector_name}_classification_results.json"
        if result_path.exists():
            with open(result_path, 'r') as f:
                results.append(json.load(f))
        else:
            print(f"  ⚠ Missing {result_path}")
    return results


def plot_exact_match_accuracy(results: List[Dict]) -> str:
    """
    Plot exact match accuracy for all detectors.
    
    Args:
        results: List of classification result dictionaries
        
    Returns:
        Path to saved plot
    """
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    detector_names = [r['detector_name'] for r in results]
    accuracies = [r['exact_match_accuracy'] for r in results]
    
    # Sort by accuracy for better visualization
    sorted_data = sorted(zip(detector_names, accuracies), key=lambda x: x[1], reverse=True)
    detector_names = [d[0] for d in sorted_data]
    accuracies = [d[1] for d in sorted_data]
    
    plt.figure(figsize=(14, 7))
    colors = sns.color_palette("viridis", len(detector_names))
    bars = plt.barh(detector_names, accuracies, color=colors)
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('Exact Match Accuracy', fontsize=12)
    plt.ylabel('Detector', fontsize=12)
    plt.title('PII Type Classification - Exact Match Accuracy', fontsize=14, fontweight='bold')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path = VISUALIZATIONS_DIR / "classification_exact_match_accuracy.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Exact match accuracy plot saved to: {output_path}")
    return str(output_path)


def plot_per_type_f1_heatmap(results: List[Dict]) -> str:
    """
    Plot heatmap of F1 scores per PII type for all detectors.
    
    Args:
        results: List of classification result dictionaries
        
    Returns:
        Path to saved plot
    """
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect all PII types from all results
    all_pii_types = set()
    for result in results:
        all_pii_types.update(result['per_type_metrics'].keys())
    
    all_pii_types = sorted(list(all_pii_types))
    detector_names = [r['detector_name'] for r in results]
    
    # Build matrix of F1 scores
    f1_matrix = []
    for detector_name in detector_names:
        result = next(r for r in results if r['detector_name'] == detector_name)
        row = []
        for pii_type in all_pii_types:
            if pii_type in result['per_type_metrics']:
                f1 = result['per_type_metrics'][pii_type]['f1']
                row.append(f1)
            else:
                row.append(0.0)
        f1_matrix.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(f1_matrix, index=detector_names, columns=all_pii_types)
    
    # Create heatmap
    plt.figure(figsize=(max(12, len(all_pii_types) * 1.5), max(8, len(detector_names) * 0.8)))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', vmin=0, vmax=1,
                cbar_kws={'label': 'F1 Score'}, linewidths=0.5)
    plt.xlabel('PII Type', fontsize=12)
    plt.ylabel('Detector', fontsize=12)
    plt.title('PII Type Classification - F1 Score Heatmap', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = VISUALIZATIONS_DIR / "classification_per_type_f1_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Per-type F1 heatmap saved to: {output_path}")
    return str(output_path)


def plot_per_type_metrics_comparison(results: List[Dict], metric: str = 'f1') -> str:
    """
    Plot comparison of a specific metric (precision/recall/f1) across PII types.
    
    Args:
        results: List of classification result dictionaries
        metric: Metric to plot ('precision', 'recall', or 'f1')
        
    Returns:
        Path to saved plot
    """
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect all PII types
    all_pii_types = set()
    for result in results:
        all_pii_types.update(result['per_type_metrics'].keys())
    
    all_pii_types = sorted(list(all_pii_types))
    detector_names = [r['detector_name'] for r in results]
    
    # Build data for grouped bar chart
    x = np.arange(len(all_pii_types))
    width = 0.12  # Width of each bar
    colors = sns.color_palette("husl", len(detector_names))
    
    fig, ax = plt.subplots(figsize=(max(16, len(all_pii_types) * 1.5), 8))
    
    for i, detector_name in enumerate(detector_names):
        result = next(r for r in results if r['detector_name'] == detector_name)
        values = []
        for pii_type in all_pii_types:
            if pii_type in result['per_type_metrics']:
                values.append(result['per_type_metrics'][pii_type][metric])
            else:
                values.append(0.0)
        
        offset = (i - len(detector_names) / 2) * width
        ax.bar(x + offset, values, width, label=detector_name, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('PII Type', fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f'PII Type Classification - {metric.capitalize()} by PII Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_pii_types, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = VISUALIZATIONS_DIR / f"classification_per_type_{metric}_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Per-type {metric} comparison plot saved to: {output_path}")
    return str(output_path)


def plot_cost_vs_accuracy(results: List[Dict]) -> str:
    """
    Plot cost vs exact match accuracy scatter plot.
    
    Args:
        results: List of classification result dictionaries
        
    Returns:
        Path to saved plot
    """
    detector_names = [r['detector_name'] for r in results]
    accuracies = [r['exact_match_accuracy'] for r in results]
    costs = [r['benchmarks']['projected_cost_50k_usd'] for r in results]
    
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("husl", len(detector_names))
    
    for i, (name, acc, cost) in enumerate(zip(detector_names, accuracies, costs)):
        plt.scatter(cost, acc, s=300, c=[colors[i]], alpha=0.7, edgecolors='black', linewidth=2)
        plt.annotate(name, (cost, acc), xytext=(8, 8), textcoords='offset points', 
                    fontsize=9, fontweight='bold')
    
    plt.xlabel('Projected Cost for 50k Emails (USD)', fontsize=12)
    plt.ylabel('Exact Match Accuracy', fontsize=12)
    plt.title('Cost vs Exact Match Accuracy Trade-off', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    output_path = VISUALIZATIONS_DIR / "classification_cost_vs_accuracy.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cost vs accuracy plot saved to: {output_path}")
    return str(output_path)


def plot_time_vs_accuracy(results: List[Dict]) -> str:
    """
    Plot time vs exact match accuracy scatter plot.
    
    Args:
        results: List of classification result dictionaries
        
    Returns:
        Path to saved plot
    """
    detector_names = [r['detector_name'] for r in results]
    accuracies = [r['exact_match_accuracy'] for r in results]
    times = [r['benchmarks']['projected_time_50k_minutes'] for r in results]
    
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("husl", len(detector_names))
    
    for i, (name, acc, time) in enumerate(zip(detector_names, accuracies, times)):
        plt.scatter(time, acc, s=300, c=[colors[i]], alpha=0.7, edgecolors='black', linewidth=2)
        plt.annotate(name, (time, acc), xytext=(8, 8), textcoords='offset points', 
                    fontsize=9, fontweight='bold')
    
    plt.xlabel('Projected Time for 50k Emails (minutes)', fontsize=12)
    plt.ylabel('Exact Match Accuracy', fontsize=12)
    plt.title('Time vs Exact Match Accuracy Trade-off', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    output_path = VISUALIZATIONS_DIR / "classification_time_vs_accuracy.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Time vs accuracy plot saved to: {output_path}")
    return str(output_path)


def plot_fnr_by_pii_type(results: List[Dict]) -> str:
    """
    Plot False Negative Rate (FNR) heatmap by PII type for each detector variant.
    
    FNR = FN / (FN + TP) = 1 - Recall
    Lower FNR is better (means fewer missed detections).
    
    Args:
        results: List of classification result dictionaries
        
    Returns:
        Path to saved plot
    """
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect all PII types from all results
    all_pii_types = set()
    for result in results:
        all_pii_types.update(result['per_type_metrics'].keys())
    
    all_pii_types = sorted(list(all_pii_types))
    detector_names = [r['detector_name'] for r in results]
    
    # Build matrix of FNR scores
    fnr_matrix = []
    for detector_name in detector_names:
        result = next(r for r in results if r['detector_name'] == detector_name)
        row = []
        for pii_type in all_pii_types:
            if pii_type in result['per_type_metrics']:
                metrics = result['per_type_metrics'][pii_type]
                tp = metrics['TP']
                fn = metrics['FN']
                # FNR = FN / (FN + TP)
                if (fn + tp) > 0:
                    fnr = fn / (fn + tp)
                else:
                    fnr = 0.0  # No ground truth examples
                row.append(fnr)
            else:
                row.append(0.0)  # No data for this PII type
        fnr_matrix.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(fnr_matrix, index=detector_names, columns=all_pii_types)
    
    # Create heatmap with color scale where lower is better (darker = lower FNR = better)
    plt.figure(figsize=(max(14, len(all_pii_types) * 1.5), max(8, len(detector_names) * 0.8)))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn_r', vmin=0, vmax=1,
                cbar_kws={'label': 'False Negative Rate (Lower is Better)'}, 
                linewidths=0.5, center=0.5)
    plt.xlabel('PII Type', fontsize=12)
    plt.ylabel('Detector Variant', fontsize=12)
    plt.title('False Negative Rate (FNR) by PII Type and Detector Variant', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = VISUALIZATIONS_DIR / "classification_fnr_by_pii_type.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FNR by PII type heatmap saved to: {output_path}")
    return str(output_path)


def plot_top_pii_types_performance(results: List[Dict], top_n: int = 5) -> str:
    """
    Plot bar chart showing top N PII types by average F1 score across all detectors.
    
    Args:
        results: List of classification result dictionaries
        top_n: Number of top PII types to show
        
    Returns:
        Path to saved plot
    """
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Calculate average F1 for each PII type across all detectors
    pii_type_f1_scores = {}
    
    for result in results:
        for pii_type, metrics in result['per_type_metrics'].items():
            if pii_type not in pii_type_f1_scores:
                pii_type_f1_scores[pii_type] = []
            pii_type_f1_scores[pii_type].append(metrics['f1'])
    
    # Calculate averages
    pii_type_avg_f1 = {
        pii_type: np.mean(scores) 
        for pii_type, scores in pii_type_f1_scores.items()
    }
    
    # Get top N
    top_pii_types = sorted(pii_type_avg_f1.items(), key=lambda x: x[1], reverse=True)[:top_n]
    pii_types = [t[0] for t in top_pii_types]
    avg_f1s = [t[1] for t in top_pii_types]
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(pii_types))
    bars = plt.barh(pii_types, avg_f1s, color=colors)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}',
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.xlabel('Average F1 Score (Across All Detectors)', fontsize=12)
    plt.ylabel('PII Type', fontsize=12)
    plt.title(f'Top {top_n} PII Types by Average F1 Score', fontsize=14, fontweight='bold')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path = VISUALIZATIONS_DIR / f"classification_top_{top_n}_pii_types.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Top {top_n} PII types plot saved to: {output_path}")
    return str(output_path)


def generate_all_classification_visualizations() -> Dict[str, str]:
    """
    Generate all classification visualizations.
    
    Returns:
        Dictionary mapping visualization names to file paths
    """
    # Load all classification results
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
    
    print("Loading classification results...")
    results = load_classification_results(detector_names)
    print(f"  ✓ Loaded {len(results)} result files")
    
    paths = {}
    
    print("\n⏳ Generating exact match accuracy chart...")
    paths['exact_match_accuracy'] = plot_exact_match_accuracy(results)
    
    print("\n⏳ Generating per-type F1 heatmap...")
    paths['per_type_f1_heatmap'] = plot_per_type_f1_heatmap(results)
    
    print("\n⏳ Generating per-type F1 comparison...")
    paths['per_type_f1_comparison'] = plot_per_type_metrics_comparison(results, 'f1')
    
    print("\n⏳ Generating per-type precision comparison...")
    paths['per_type_precision_comparison'] = plot_per_type_metrics_comparison(results, 'precision')
    
    print("\n⏳ Generating per-type recall comparison...")
    paths['per_type_recall_comparison'] = plot_per_type_metrics_comparison(results, 'recall')
    
    print("\n⏳ Generating cost vs accuracy chart...")
    paths['cost_vs_accuracy'] = plot_cost_vs_accuracy(results)
    
    print("\n⏳ Generating time vs accuracy chart...")
    paths['time_vs_accuracy'] = plot_time_vs_accuracy(results)
    
    print("\n⏳ Generating top PII types performance chart...")
    paths['top_pii_types'] = plot_top_pii_types_performance(results, top_n=8)
    
    print("\n⏳ Generating FNR by PII type heatmap...")
    paths['fnr_by_pii_type'] = plot_fnr_by_pii_type(results)
    
    print("\n✓ All classification visualizations generated successfully!")
    return paths

