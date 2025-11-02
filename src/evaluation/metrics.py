"""Metrics calculation for PII detection evaluation."""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class ConfusionMatrix:
    """Confusion matrix for binary classification."""
    TP: int  # True Positives
    FP: int  # False Positives
    TN: int  # True Negatives
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

    def accuracy(self) -> float:
        """Calculate accuracy: (TP + TN) / (TP + FP + TN + FN)"""
        total = self.TP + self.FP + self.TN + self.FN
        return (self.TP + self.TN) / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with all metrics."""
        return {
            'TP': self.TP,
            'FP': self.FP,
            'TN': self.TN,
            'FN': self.FN,
            'precision': self.precision(),
            'recall': self.recall(),
            'f1': self.f1_score(),
            'accuracy': self.accuracy()
        }


def calculate_confusion_matrix(
    predictions: List[bool],
    ground_truth: List[bool]
) -> ConfusionMatrix:
    """
    Calculate confusion matrix from predictions and ground truth.

    Args:
        predictions: List of boolean predictions (True = has PII, False = no PII)
        ground_truth: List of boolean ground truth labels

    Returns:
        ConfusionMatrix object
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Predictions and ground truth must have same length. "
            f"Got {len(predictions)} vs {len(ground_truth)}"
        )

    TP = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    FP = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    TN = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)
    FN = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)

    return ConfusionMatrix(TP=TP, FP=FP, TN=TN, FN=FN)


def get_false_positive_indices(
    predictions: List[bool],
    ground_truth: List[bool]
) -> List[int]:
    """
    Get indices of false positive predictions.

    Args:
        predictions: List of boolean predictions
        ground_truth: List of boolean ground truth labels

    Returns:
        List of indices where prediction is True but ground truth is False
    """
    return [
        i for i, (p, g) in enumerate(zip(predictions, ground_truth))
        if p and not g
    ]


def get_false_negative_indices(
    predictions: List[bool],
    ground_truth: List[bool]
) -> List[int]:
    """
    Get indices of false negative predictions.

    Args:
        predictions: List of boolean predictions
        ground_truth: List of boolean ground truth labels

    Returns:
        List of indices where prediction is False but ground truth is True
    """
    return [
        i for i, (p, g) in enumerate(zip(predictions, ground_truth))
        if not p and g
    ]
