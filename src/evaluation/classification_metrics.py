"""Metrics calculation for PII type classification evaluation."""

from typing import List, Dict, Set
from dataclasses import dataclass


@dataclass
class PerTypeMetrics:
    """Metrics for a single PII type in multi-label classification."""
    pii_type: str
    TP: int  # True Positives
    FP: int  # False Positives
    FN: int  # False Negatives
    
    def precision(self) -> float:
        """Calculate precision: TP / (TP + FP)"""
        denominator = self.TP + self.FP
        return self.TP / denominator if denominator > 0 else 0.0
    
    def recall(self) -> float:
        """Calculate recall: TP / (TP + FN)"""
        denominator = self.TP + self.FN
        return self.TP / denominator if denominator > 0 else 0.0
    
    def f1_score(self) -> float:
        """Calculate F1 score: 2 * (precision * recall) / (precision + recall)"""
        p = self.precision()
        r = self.recall()
        denominator = p + r
        return 2 * p * r / denominator if denominator > 0 else 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'pii_type': self.pii_type,
            'TP': self.TP,
            'FP': self.FP,
            'FN': self.FN,
            'precision': self.precision(),
            'recall': self.recall(),
            'f1': self.f1_score()
        }


def calculate_exact_match_accuracy(
    predictions: List[List[str]],
    ground_truth: List[List[str]]
) -> float:
    """
    Calculate exact match accuracy.
    
    Exact match means the predicted set of PII types exactly matches
    the ground truth set (order doesn't matter, case-insensitive).
    
    Args:
        predictions: List of predicted PII type lists
        ground_truth: List of ground truth PII type lists
        
    Returns:
        Exact match accuracy (0.0 to 1.0)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Predictions and ground truth must have same length. "
            f"Got {len(predictions)} vs {len(ground_truth)}"
        )
    
    matches = 0
    for pred_set, gt_set in zip(predictions, ground_truth):
        # Convert to sets for comparison (case-insensitive)
        pred_set_normalized = {p.lower().strip() for p in pred_set}
        gt_set_normalized = {g.lower().strip() for g in gt_set}
        
        if pred_set_normalized == gt_set_normalized:
            matches += 1
    
    return matches / len(predictions) if predictions else 0.0


def calculate_per_type_metrics(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    all_pii_types: List[str]
) -> Dict[str, PerTypeMetrics]:
    """
    Calculate precision, recall, and F1 for each PII type.
    
    Treats this as a multi-label classification problem.
    For each PII type:
    - TP: predicted and in ground truth
    - FP: predicted but not in ground truth
    - FN: in ground truth but not predicted
    
    Args:
        predictions: List of predicted PII type lists
        ground_truth: List of ground truth PII type lists
        all_pii_types: List of all possible PII types to evaluate
        
    Returns:
        Dictionary mapping PII type -> PerTypeMetrics
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Predictions and ground truth must have same length. "
            f"Got {len(predictions)} vs {len(ground_truth)}"
        )
    
    # Normalize all PII types to lowercase
    all_pii_types_normalized = {pii_type.lower().strip() for pii_type in all_pii_types}
    
    # Initialize metrics for each PII type
    metrics_dict = {}
    for pii_type in all_pii_types:
        metrics_dict[pii_type] = PerTypeMetrics(
            pii_type=pii_type,
            TP=0,
            FP=0,
            FN=0
        )
    
    # Calculate TP, FP, FN for each PII type
    for pred_list, gt_list in zip(predictions, ground_truth):
        # Normalize to sets (case-insensitive)
        pred_set = {p.lower().strip() for p in pred_list}
        gt_set = {g.lower().strip() for g in gt_list}
        
        # Check each PII type
        for pii_type in all_pii_types:
            pii_type_normalized = pii_type.lower().strip()
            
            pred_has = pii_type_normalized in pred_set
            gt_has = pii_type_normalized in gt_set
            
            if pred_has and gt_has:
                metrics_dict[pii_type].TP += 1
            elif pred_has and not gt_has:
                metrics_dict[pii_type].FP += 1
            elif not pred_has and gt_has:
                metrics_dict[pii_type].FN += 1
            # else: TN (true negative) - both don't have it, which we don't track
    
    return metrics_dict


def get_all_pii_types_from_dataset(ground_truth: List[List[str]]) -> List[str]:
    """
    Extract all unique PII types from the ground truth dataset.
    
    Args:
        ground_truth: List of ground truth PII type lists
        
    Returns:
        Sorted list of unique PII types
    """
    all_types = set()
    for gt_list in ground_truth:
        for pii_type in gt_list:
            all_types.add(pii_type.strip().lower())
    
    return sorted(all_types)

