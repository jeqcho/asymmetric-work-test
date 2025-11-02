"""False positive tracking and CSV export utilities."""

import csv
from typing import List
from pathlib import Path

from src.detectors.base import DetectionResult
from src.utils.data_loader import Email
from src.config import FALSE_POSITIVES_DIR


def export_false_positives(
    detector_name: str,
    results: List[DetectionResult],
    emails: List[Email],
    ground_truth: List[bool]
) -> str:
    """
    Export false positives to CSV for manual review.

    Args:
        detector_name: Name of the detector
        results: List of detection results
        emails: List of emails (corresponding to results)
        ground_truth: List of ground truth labels

    Returns:
        Path to the exported CSV file
    """
    # Ensure output directory exists
    FALSE_POSITIVES_DIR.mkdir(parents=True, exist_ok=True)

    # Build output path
    output_path = FALSE_POSITIVES_DIR / f"{detector_name}_fp.csv"

    # Find false positives (predicted PII but ground truth is no PII)
    false_positives = []
    for result, email, gt in zip(results, emails, ground_truth):
        if result.has_pii and not gt:
            # This is a false positive
            false_positives.append({
                'email_id': email.id,
                'detector': detector_name,
                'ground_truth': 'NO',
                'prediction': 'YES',
                'subject': email.subject,
                'from': email.from_addr,
                'to': email.to_addr,
                'message_body_preview': email.message_body[:200] + '...' if len(email.message_body) > 200 else email.message_body,
                'confidence': result.confidence if result.confidence is not None else 'N/A'
            })

    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if false_positives:
            fieldnames = [
                'email_id',
                'detector',
                'ground_truth',
                'prediction',
                'subject',
                'from',
                'to',
                'message_body_preview',
                'confidence'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(false_positives)
        else:
            # Write empty file with header
            writer = csv.writer(f)
            writer.writerow(['No false positives detected'])

    return str(output_path)


def export_false_negatives(
    detector_name: str,
    results: List[DetectionResult],
    emails: List[Email],
    ground_truth: List[bool]
) -> str:
    """
    Export false negatives to CSV for manual review.

    Args:
        detector_name: Name of the detector
        results: List of detection results
        emails: List of emails (corresponding to results)
        ground_truth: List of ground truth labels

    Returns:
        Path to the exported CSV file
    """
    # Ensure output directory exists
    FALSE_POSITIVES_DIR.mkdir(parents=True, exist_ok=True)

    # Build output path
    output_path = FALSE_POSITIVES_DIR / f"{detector_name}_fn.csv"

    # Find false negatives (predicted no PII but ground truth is PII)
    false_negatives = []
    for result, email, gt in zip(results, emails, ground_truth):
        if not result.has_pii and gt:
            # This is a false negative
            false_negatives.append({
                'email_id': email.id,
                'detector': detector_name,
                'ground_truth': 'YES',
                'prediction': 'NO',
                'subject': email.subject,
                'pii_types': email.data_elements,
                'message_body_preview': email.message_body[:200] + '...' if len(email.message_body) > 200 else email.message_body,
                'confidence': result.confidence if result.confidence is not None else 'N/A'
            })

    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if false_negatives:
            fieldnames = [
                'email_id',
                'detector',
                'ground_truth',
                'prediction',
                'subject',
                'pii_types',
                'message_body_preview',
                'confidence'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(false_negatives)
        else:
            # Write empty file with header
            writer = csv.writer(f)
            writer.writerow(['No false negatives detected'])

    return str(output_path)
