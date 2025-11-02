"""Main test harness to orchestrate detector evaluation."""

import json
from typing import List, Dict
from dataclasses import asdict
from pathlib import Path

from src.detectors.base import BaseDetector, DetectionResult
from src.utils.data_loader import Email, parse_ground_truth
from src.evaluation.metrics import calculate_confusion_matrix
from src.evaluation.benchmark import calculate_benchmark_stats
from src.evaluation.false_positive_tracker import export_false_positives, export_false_negatives
from src.config import RESULTS_DIR


class TestHarness:
    """Orchestrates evaluation of PII detectors."""

    def __init__(self, detectors: List[BaseDetector], emails: List[Email]):
        """
        Initialize test harness.

        Args:
            detectors: List of detector instances to evaluate
            emails: List of emails to process
        """
        self.detectors = detectors
        self.emails = emails
        self.ground_truth = parse_ground_truth(emails)

        # Ensure results directory exists
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def run_evaluation(self, detector: BaseDetector) -> Dict:
        """
        Run evaluation for a single detector.

        Args:
            detector: Detector instance to evaluate

        Returns:
            Dictionary with evaluation results
        """
        print(f"\n  Evaluating {detector.name}...")
        print(f"  ⏳ Processing {len(self.emails)} emails...")

        # Run detection on all emails
        results: List[DetectionResult] = []
        for i, email in enumerate(self.emails, 1):
            if i % 50 == 0:
                print(f"    Progress: {i}/{len(self.emails)} emails processed ({i/len(self.emails)*100:.1f}%)")

            result = detector.detect(email)
            results.append(result)

        print(f"  ✓ Completed {len(results)} detections")

        # Extract predictions
        predictions = [r.has_pii for r in results]

        # Calculate metrics
        print(f"  ⏳ Calculating metrics...")
        confusion_matrix = calculate_confusion_matrix(predictions, self.ground_truth)
        benchmark_stats = calculate_benchmark_stats(results)

        # Export false positives and false negatives
        print(f"  ⏳ Exporting false positives/negatives...")
        fp_path = export_false_positives(
            detector.name,
            results,
            self.emails,
            self.ground_truth
        )
        fn_path = export_false_negatives(
            detector.name,
            results,
            self.emails,
            self.ground_truth
        )

        print(f"  ✓ Metrics: TP={confusion_matrix.TP}, FP={confusion_matrix.FP}, "
              f"TN={confusion_matrix.TN}, FN={confusion_matrix.FN}")
        print(f"  ✓ F1 Score: {confusion_matrix.f1_score():.3f}")
        print(f"  ✓ Avg Time: {benchmark_stats.avg_time_per_email_ms:.1f}ms")
        print(f"  ✓ Avg Cost: ${benchmark_stats.avg_cost_per_email_usd:.6f}")

        # Build result dictionary
        result_dict = {
            'detector_name': detector.name,
            'total_emails': len(self.emails),
            'predictions': [
                {
                    'email_id': r.email_id,
                    'prediction': r.has_pii,
                    'ground_truth': gt,
                    'time_ms': r.time_ms,
                    'cost_usd': r.cost_usd,
                    'confidence': r.confidence,
                    'input_tokens': r.input_tokens,
                    'output_tokens': r.output_tokens
                }
                for r, gt in zip(results, self.ground_truth)
            ],
            'metrics': confusion_matrix.to_dict(),
            'benchmarks': benchmark_stats.to_dict(),
            'false_positives_file': fp_path,
            'false_negatives_file': fn_path
        }

        # Save results to JSON
        print(f"  ⏳ Saving results to JSON...")
        output_path = RESULTS_DIR / f"{detector.name}_results.json"
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)

        print(f"  ✓ Results saved to: {output_path}")

        return result_dict

    def run_all_evaluations(self) -> List[Dict]:
        """
        Run evaluation for all detectors.

        Returns:
            List of result dictionaries, one per detector
        """
        all_results = []

        print(f"\n{'='*60}")
        print(f"Starting evaluation of {len(self.detectors)} detectors")
        print(f"Dataset size: {len(self.emails)} emails")
        print(f"Ground truth: {sum(self.ground_truth)} with PII, "
              f"{len(self.ground_truth) - sum(self.ground_truth)} without PII")
        print(f"{'='*60}")

        for i, detector in enumerate(self.detectors, 1):
            print(f"\n[{i}/{len(self.detectors)}] Running {detector.name}...")
            result = self.run_evaluation(detector)
            all_results.append(result)

        print(f"\n{'='*60}")
        print(f"Evaluation complete!")
        print(f"{'='*60}\n")

        return all_results
