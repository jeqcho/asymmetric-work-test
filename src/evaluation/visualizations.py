"""Generate comparison CSV and visualizations from evaluation results."""

import csv
import json
from typing import List, Dict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.config import RESULTS_DIR, VISUALIZATIONS_DIR


def generate_comparison_csv(results: List[Dict]) -> str:
    """
    Generate comparison CSV from all detector results.

    Args:
        results: List of result dictionaries from test harness

    Returns:
        Path to comparison CSV file
    """
    output_path = RESULTS_DIR / "comparison_metrics.csv"

    # Extract key metrics from each result
    rows = []
    for result in results:
        metrics = result['metrics']
        benchmarks = result['benchmarks']

        row = {
            'Detector': result['detector_name'],
            'TP': metrics['TP'],
            'FP': metrics['FP'],
            'TN': metrics['TN'],
            'FN': metrics['FN'],
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1': f"{metrics['f1']:.3f}",
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Avg Time (ms)': f"{benchmarks['avg_time_per_email_ms']:.1f}",
            'Avg Cost (USD)': f"{benchmarks['avg_cost_per_email_usd']:.6f}",
            'Total Cost (USD)': f"{benchmarks['total_cost_usd']:.4f}",
            'Projected Time 50k (min)': f"{benchmarks['projected_time_50k_minutes']:.1f}",
            'Projected Cost 50k (USD)': f"{benchmarks['projected_cost_50k_usd']:.2f}"
        }
        rows.append(row)

    # Write to CSV
    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            'Detector', 'TP', 'FP', 'TN', 'FN', 'Precision', 'Recall', 'F1', 'Accuracy',
            'Avg Time (ms)', 'Avg Cost (USD)', 'Total Cost (USD)',
            'Projected Time 50k (min)', 'Projected Cost 50k (USD)'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Comparison CSV saved to: {output_path}")
    return str(output_path)


def plot_f1_scores(results: List[Dict]) -> str:
    """
    Plot F1 scores for all detectors.

    Args:
        results: List of result dictionaries

    Returns:
        Path to saved plot
    """
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    detector_names = [r['detector_name'] for r in results]
    f1_scores = [r['metrics']['f1'] for r in results]

    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("husl", len(detector_names))
    bars = plt.bar(detector_names, f1_scores, color=colors)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

    plt.xlabel('Detector', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Scores by Detector', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    output_path = VISUALIZATIONS_DIR / "f1_scores.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"F1 scores plot saved to: {output_path}")
    return str(output_path)


def plot_cost_vs_f1(results: List[Dict]) -> str:
    """
    Plot cost vs F1 score scatter plot.

    Args:
        results: List of result dictionaries

    Returns:
        Path to saved plot
    """
    detector_names = [r['detector_name'] for r in results]
    f1_scores = [r['metrics']['f1'] for r in results]
    costs = [r['benchmarks']['projected_cost_50k_usd'] for r in results]

    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", len(detector_names))

    for i, (name, f1, cost) in enumerate(zip(detector_names, f1_scores, costs)):
        plt.scatter(cost, f1, s=200, c=[colors[i]], alpha=0.7, edgecolors='black', linewidth=1.5)
        plt.annotate(name, (cost, f1), xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.xlabel('Projected Cost for 50k Emails (USD)', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Cost vs F1 Score Trade-off', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = VISUALIZATIONS_DIR / "cost_vs_f1.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Cost vs F1 plot saved to: {output_path}")
    return str(output_path)


def plot_time_vs_f1(results: List[Dict]) -> str:
    """
    Plot time vs F1 score scatter plot.

    Args:
        results: List of result dictionaries

    Returns:
        Path to saved plot
    """
    detector_names = [r['detector_name'] for r in results]
    f1_scores = [r['metrics']['f1'] for r in results]
    times = [r['benchmarks']['projected_time_50k_minutes'] for r in results]

    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", len(detector_names))

    for i, (name, f1, time) in enumerate(zip(detector_names, f1_scores, times)):
        plt.scatter(time, f1, s=200, c=[colors[i]], alpha=0.7, edgecolors='black', linewidth=1.5)
        plt.annotate(name, (time, f1), xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.xlabel('Projected Time for 50k Emails (minutes)', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Time vs F1 Score Trade-off', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = VISUALIZATIONS_DIR / "time_vs_f1.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Time vs F1 plot saved to: {output_path}")
    return str(output_path)


def plot_classification_rates(results: List[Dict]) -> str:
    """
    Plot all classification rates: FPR, FNR, TPR, TNR for all detectors.

    Args:
        results: List of result dictionaries

    Returns:
        Path to saved plot
    """
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    detector_names = [r['detector_name'] for r in results]

    # Calculate all rates
    fpr_values = []  # False Positive Rate = FP / (FP + TN)
    fnr_values = []  # False Negative Rate = FN / (FN + TP)
    tpr_values = []  # True Positive Rate (Recall) = TP / (TP + FN)
    tnr_values = []  # True Negative Rate (Specificity) = TN / (TN + FP)

    for r in results:
        metrics = r['metrics']
        tp = metrics['TP']
        fp = metrics['FP']
        tn = metrics['TN']
        fn = metrics['FN']

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

        fpr_values.append(fpr)
        fnr_values.append(fnr)
        tpr_values.append(tpr)
        tnr_values.append(tnr)

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = sns.color_palette("husl", len(detector_names))

    # Plot 1: False Positive Rate (Lower is Better)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(detector_names, fpr_values, color=colors)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    ax1.set_xlabel('Detector', fontsize=11)
    ax1.set_ylabel('False Positive Rate', fontsize=11)
    ax1.set_title('False Positive Rate (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: False Negative Rate (Lower is Better)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(detector_names, fnr_values, color=colors)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    ax2.set_xlabel('Detector', fontsize=11)
    ax2.set_ylabel('False Negative Rate', fontsize=11)
    ax2.set_title('False Negative Rate (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: True Positive Rate / Recall (Higher is Better)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(detector_names, tpr_values, color=colors)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    ax3.set_xlabel('Detector', fontsize=11)
    ax3.set_ylabel('True Positive Rate (Recall)', fontsize=11)
    ax3.set_title('True Positive Rate / Recall (Higher is Better)', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, 1.0)
    ax3.grid(axis='y', alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 4: True Negative Rate / Specificity (Higher is Better)
    ax4 = axes[1, 1]
    bars4 = ax4.bar(detector_names, tnr_values, color=colors)
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    ax4.set_xlabel('Detector', fontsize=11)
    ax4.set_ylabel('True Negative Rate (Specificity)', fontsize=11)
    ax4.set_title('True Negative Rate / Specificity (Higher is Better)', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0, 1.0)
    ax4.grid(axis='y', alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    output_path = VISUALIZATIONS_DIR / "classification_rates.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Classification rates plot saved to: {output_path}")
    return str(output_path)


def plot_confusion_matrices(results: List[Dict]) -> str:
    """
    Plot confusion matrices for all detectors.

    Args:
        results: List of result dictionaries

    Returns:
        Path to saved plot
    """
    n_detectors = len(results)
    n_cols = 4
    n_rows = (n_detectors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_detectors > 1 else [axes]

    for i, result in enumerate(results):
        metrics = result['metrics']
        cm = np.array([[metrics['TP'], metrics['FP']],
                       [metrics['FN'], metrics['TN']]])

        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Pred PII', 'Pred No PII'],
                   yticklabels=['True PII', 'True No PII'],
                   cbar=False)
        ax.set_title(f"{result['detector_name']}\nF1: {metrics['f1']:.3f}",
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Actual', fontsize=9)

    # Hide unused subplots
    for i in range(n_detectors, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    output_path = VISUALIZATIONS_DIR / "confusion_matrices.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrices plot saved to: {output_path}")
    return str(output_path)


def plot_tnr_fnr_comparison(results: List[Dict]) -> str:
    """
    Plot TNR and FNR comparison for all detectors.

    Args:
        results: List of result dictionaries

    Returns:
        Path to saved plot
    """
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    detector_names = [r['detector_name'] for r in results]

    # Calculate TNR and FNR for each detector
    tnr_values = []
    fnr_values = []

    for r in results:
        metrics = r['metrics']
        tp = metrics['TP']
        fp = metrics['FP']
        tn = metrics['TN']
        fn = metrics['FN']

        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

        tnr_values.append(tnr)
        fnr_values.append(fnr)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(detector_names))
    width = 0.6

    # Assign colors: group Claude models together
    colors = []
    for name in detector_names:
        if 'presidio_v2' in name:
            colors.append('#1f77b4')  # Blue for Presidio V2
        elif 'presidio' in name:
            colors.append('#ff7f0e')  # Orange for Presidio V1
        elif 'haiku' in name or 'sonnet' in name:
            colors.append('#2ca02c')  # Green for all Claude models
        else:
            colors.append('#d62728')  # Red for others

    # Plot 1: TNR (Higher is Better)
    bars1 = ax1.bar(x, tnr_values, width, color=colors)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Detector', fontsize=12)
    ax1.set_ylabel('True Negative Rate (TNR)', fontsize=12)
    ax1.set_title('True Negative Rate - Specificity (Higher is Better)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(detector_names, rotation=45, ha='right')
    # Color presidio_v2_strict label text
    for i, label in enumerate(ax1.get_xticklabels()):
        if detector_names[i] == 'presidio_v2_strict':
            label.set_color('red')
            label.set_weight('bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: FNR (Lower is Better)
    bars2 = ax2.bar(x, fnr_values, width, color=colors)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel('Detector', fontsize=12)
    ax2.set_ylabel('False Negative Rate (FNR)', fontsize=12)
    ax2.set_title('False Negative Rate (Lower is Better)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(detector_names, rotation=45, ha='right')
    # Color presidio_v2_strict label text
    for i, label in enumerate(ax2.get_xticklabels()):
        if detector_names[i] == 'presidio_v2_strict':
            label.set_color('red')
            label.set_weight('bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = VISUALIZATIONS_DIR / "tnr_fnr_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"TNR/FNR comparison plot saved to: {output_path}")
    return str(output_path)


def plot_presidio_v2_threshold_analysis(v2_results: List[Dict], labeled_emails) -> str:
    """
    Plot TNR and FNR for various presidio v2 thresholds.

    Args:
        v2_results: List of presidio v2 result dictionaries
        labeled_emails: List of labeled email objects for re-evaluation

    Returns:
        Path to saved plot
    """
    from src.detectors.presidio_detector import PresidioDetector
    from src.evaluation.metrics import calculate_confusion_matrix

    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Test thresholds from 0.0 to 1.0 at 0.1 increments
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tnr_values = []
    fnr_values = []

    print(f"  Analyzing presidio_v2 across {len(thresholds)} thresholds...")

    for threshold in thresholds:
        # Create a temporary detector with this threshold
        detector = PresidioDetector(f"presidio_v2_temp_{threshold}", threshold)

        # Run detection on all emails
        predictions = []
        ground_truths = []

        for email in labeled_emails:
            result = detector.detect(email)
            predictions.append(result.has_pii)
            ground_truths.append(email.has_pii())

        # Calculate metrics
        cm = calculate_confusion_matrix(predictions, ground_truths)

        tp = cm.TP
        fp = cm.FP
        tn = cm.TN
        fn = cm.FN

        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        tnr_values.append(tnr)
        fnr_values.append(fnr)

        pii_count = sum(predictions)
        print(f"    Threshold {threshold:.1f}: TP={tp}, FP={fp}, TN={tn}, FN={fn}, PII detected={pii_count}, TNR={tnr:.3f}, FNR={fnr:.3f}")

    # Create line plot
    plt.figure(figsize=(12, 7))

    plt.plot(thresholds, tnr_values, marker='o', linewidth=2, markersize=8,
             color='#2E86AB', label='TNR (True Negative Rate) - Higher is Better', zorder=3)
    plt.plot(thresholds, fnr_values, marker='s', linewidth=2, markersize=8,
             color='#A23B72', label='FNR (False Negative Rate) - Lower is Better', zorder=3)

    # Add vertical dotted lines for the 3 variants
    plt.axvline(x=0.3, color='gray', linestyle=':', linewidth=1.5, alpha=0.5,
                label='Strict (0.3)', zorder=1)
    plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5,
                label='Moderate (0.5)', zorder=1)
    plt.axvline(x=0.8, color='gray', linestyle=':', linewidth=1.5, alpha=0.5,
                label='Lax (0.8)', zorder=1)

    plt.xlabel('Confidence Threshold', fontsize=13)
    plt.ylabel('Rate', fontsize=13)
    plt.title('Presidio V2 Threshold Sensitivity Analysis', fontsize=14, fontweight='bold')
    plt.xticks(thresholds)
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3, zorder=0)
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()

    output_path = VISUALIZATIONS_DIR / "presidio_v2_threshold_sensitivity.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Presidio V2 threshold sensitivity plot saved to: {output_path}")
    return str(output_path)


def generate_all_visualizations(results: List[Dict]) -> Dict[str, str]:
    """
    Generate all visualizations and comparison CSV.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary mapping visualization names to file paths
    """
    print("  ⏳ Generating comparison CSV...")
    comparison_csv = generate_comparison_csv(results)
    print(f"  ✓ Comparison CSV: {comparison_csv}")

    print("  ⏳ Generating F1 scores chart...")
    f1_scores = plot_f1_scores(results)
    print(f"  ✓ F1 scores chart: {f1_scores}")

    print("  ⏳ Generating cost vs F1 chart...")
    cost_vs_f1 = plot_cost_vs_f1(results)
    print(f"  ✓ Cost vs F1 chart: {cost_vs_f1}")

    print("  ⏳ Generating time vs F1 chart...")
    time_vs_f1 = plot_time_vs_f1(results)
    print(f"  ✓ Time vs F1 chart: {time_vs_f1}")

    print("  ⏳ Generating classification rates chart (FPR, FNR, TPR, TNR)...")
    classification_rates = plot_classification_rates(results)
    print(f"  ✓ Classification rates chart: {classification_rates}")

    print("  ⏳ Generating confusion matrices...")
    confusion_matrices = plot_confusion_matrices(results)
    print(f"  ✓ Confusion matrices: {confusion_matrices}")

    paths = {
        'comparison_csv': comparison_csv,
        'f1_scores': f1_scores,
        'cost_vs_f1': cost_vs_f1,
        'time_vs_f1': time_vs_f1,
        'classification_rates': classification_rates,
        'confusion_matrices': confusion_matrices
    }

    print("  ✓ All visualizations generated successfully!")
    return paths
