"""Test harness for PII type classification evaluation."""

import json
from typing import List, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.detectors.base import BaseDetector, PIIClassificationResult
from src.utils.data_loader import Email
from src.evaluation.classification_metrics import (
    calculate_exact_match_accuracy,
    calculate_per_type_metrics,
    get_all_pii_types_from_dataset
)
from src.evaluation.benchmark import BenchmarkStats
from src.config import RESULTS_DIR, PARALLEL_REQUESTS, TARGET_DATASET_SIZE
import statistics


class ClassificationTestHarness:
    """Orchestrates evaluation of PII type classification detectors."""

    def __init__(
        self,
        detectors: List[BaseDetector],
        emails: List[Email],
        parallel_requests: Optional[int] = None
    ):
        """
        Initialize classification test harness.

        Args:
            detectors: List of detector instances to evaluate
            emails: List of emails to process
            parallel_requests: Number of parallel API requests (defaults to PARALLEL_REQUESTS from config)
        """
        self.detectors = detectors
        self.emails = emails
        self.ground_truth = [email.get_pii_types() for email in emails]
        self.parallel_requests = parallel_requests if parallel_requests is not None else PARALLEL_REQUESTS

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
        print(f"  ⏳ Processing {len(self.emails)} emails with {self.parallel_requests} parallel requests...")

        # Run detection on all emails in parallel
        results: List[PIIClassificationResult] = [None] * len(self.emails)  # Pre-allocate to maintain order

        # Create progress bar
        with tqdm(total=len(self.emails), desc=f"  {detector.name}", unit="email", ncols=100) as pbar:
            with ThreadPoolExecutor(max_workers=self.parallel_requests) as executor:
                # Submit all tasks, storing email index with each future
                future_to_index = {
                    executor.submit(detector.detect, email): i
                    for i, email in enumerate(self.emails)
                }

                # Process completed tasks as they finish
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                    except Exception as e:
                        # If detection fails, create an error result
                        print(f"\n    Error processing email at index {index}: {e}")
                        results[index] = PIIClassificationResult(
                            email_id=self.emails[index].id,
                            pii_types=[],
                            time_ms=0.0,
                            cost_usd=0.0,
                            input_tokens=0,
                            output_tokens=0
                        )
                    finally:
                        # Update progress bar
                        pbar.update(1)

        print(f"  ✓ Completed {len(results)} detections")

        # Extract predictions
        predictions = [r.pii_types for r in results]

        # Calculate metrics
        print(f"  ⏳ Calculating metrics...")
        exact_match_acc = calculate_exact_match_accuracy(predictions, self.ground_truth)
        
        # Get all PII types from dataset
        all_pii_types = get_all_pii_types_from_dataset(self.ground_truth)
        per_type_metrics = calculate_per_type_metrics(predictions, self.ground_truth, all_pii_types)
        
        # Calculate benchmark stats (similar to calculate_benchmark_stats but for PIIClassificationResult)
        times_ms = [r.time_ms for r in results]
        costs = [r.cost_usd for r in results]
        input_tokens = [r.input_tokens or 0 for r in results]
        output_tokens = [r.output_tokens or 0 for r in results]
        
        total_time_seconds = sum(times_ms) / 1000
        avg_time_ms = statistics.mean(times_ms)
        median_time_ms = statistics.median(times_ms)
        min_time_ms = min(times_ms)
        max_time_ms = max(times_ms)
        total_cost = sum(costs)
        avg_cost = statistics.mean(costs)
        total_input_tokens = sum(input_tokens)
        total_output_tokens = sum(output_tokens)
        avg_input_tokens = statistics.mean(input_tokens)
        avg_output_tokens = statistics.mean(output_tokens)
        projected_time_50k_minutes = (avg_time_ms * TARGET_DATASET_SIZE) / 1000 / 60
        projected_cost_50k = avg_cost * TARGET_DATASET_SIZE
        
        benchmark_stats = BenchmarkStats(
            total_time_seconds=total_time_seconds,
            avg_time_per_email_ms=avg_time_ms,
            median_time_per_email_ms=median_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            total_cost_usd=total_cost,
            avg_cost_per_email_usd=avg_cost,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            projected_time_50k_minutes=projected_time_50k_minutes,
            projected_cost_50k_usd=projected_cost_50k
        )

        print(f"  ✓ Exact Match Accuracy: {exact_match_acc:.3f}")
        print(f"  ✓ Avg Time: {benchmark_stats.avg_time_per_email_ms:.1f}ms")
        print(f"  ✓ Avg Cost: ${benchmark_stats.avg_cost_per_email_usd:.6f}")

        # Build result dictionary
        result_dict = {
            'detector_name': detector.name,
            'total_emails': len(self.emails),
            'exact_match_accuracy': exact_match_acc,
            'predictions': [
                {
                    'email_id': r.email_id,
                    'predicted_pii_types': r.pii_types,
                    'ground_truth_pii_types': gt,
                    'time_ms': r.time_ms,
                    'cost_usd': r.cost_usd,
                    'input_tokens': r.input_tokens,
                    'output_tokens': r.output_tokens
                }
                for r, gt in zip(results, self.ground_truth)
            ],
            'per_type_metrics': {
                pii_type: metrics.to_dict()
                for pii_type, metrics in per_type_metrics.items()
            },
            'benchmarks': benchmark_stats.to_dict()
        }

        # Save results to JSON
        print(f"  ⏳ Saving results to JSON...")
        output_path = RESULTS_DIR / f"{detector.name}_classification_results.json"
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

        # Count emails with/without PII
        emails_with_pii = sum(1 for gt in self.ground_truth if gt)
        emails_without_pii = len(self.ground_truth) - emails_with_pii

        print(f"\n{'='*60}")
        print(f"Starting PII type classification evaluation of {len(self.detectors)} detectors")
        print(f"Dataset size: {len(self.emails)} emails")
        print(f"Ground truth: {emails_with_pii} with PII, {emails_without_pii} without PII")
        print(f"{'='*60}")

        for i, detector in enumerate(self.detectors, 1):
            print(f"\n[{i}/{len(self.detectors)}] Running {detector.name}...")
            result = self.run_evaluation(detector)
            all_results.append(result)

        print(f"\n{'='*60}")
        print(f"Classification evaluation complete!")
        print(f"{'='*60}\n")

        return all_results

