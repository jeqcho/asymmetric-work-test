"""Benchmarking utilities for cost and time tracking."""

from typing import List
from dataclasses import dataclass
import statistics

from src.detectors.base import DetectionResult
from src.config import TARGET_DATASET_SIZE


@dataclass
class BenchmarkStats:
    """Benchmark statistics for a detector."""
    total_time_seconds: float
    avg_time_per_email_ms: float
    median_time_per_email_ms: float
    min_time_ms: float
    max_time_ms: float
    total_cost_usd: float
    avg_cost_per_email_usd: float
    total_input_tokens: int
    total_output_tokens: int
    avg_input_tokens: float
    avg_output_tokens: float
    projected_time_50k_minutes: float  # Projected time for 50k emails
    projected_cost_50k_usd: float  # Projected cost for 50k emails

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total_time_seconds': self.total_time_seconds,
            'avg_time_per_email_ms': self.avg_time_per_email_ms,
            'median_time_per_email_ms': self.median_time_per_email_ms,
            'min_time_ms': self.min_time_ms,
            'max_time_ms': self.max_time_ms,
            'total_cost_usd': self.total_cost_usd,
            'avg_cost_per_email_usd': self.avg_cost_per_email_usd,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'avg_input_tokens': self.avg_input_tokens,
            'avg_output_tokens': self.avg_output_tokens,
            'projected_time_50k_minutes': self.projected_time_50k_minutes,
            'projected_cost_50k_usd': self.projected_cost_50k_usd
        }


def calculate_benchmark_stats(results: List[DetectionResult]) -> BenchmarkStats:
    """
    Calculate benchmark statistics from detection results.

    Args:
        results: List of DetectionResult objects

    Returns:
        BenchmarkStats object with aggregated statistics
    """
    if not results:
        raise ValueError("Cannot calculate stats from empty results")

    # Extract metrics
    times_ms = [r.time_ms for r in results]
    costs = [r.cost_usd for r in results]
    input_tokens = [r.input_tokens or 0 for r in results]
    output_tokens = [r.output_tokens or 0 for r in results]

    # Calculate time stats
    total_time_seconds = sum(times_ms) / 1000
    avg_time_ms = statistics.mean(times_ms)
    median_time_ms = statistics.median(times_ms)
    min_time_ms = min(times_ms)
    max_time_ms = max(times_ms)

    # Calculate cost stats
    total_cost = sum(costs)
    avg_cost = statistics.mean(costs)

    # Calculate token stats
    total_input_tokens = sum(input_tokens)
    total_output_tokens = sum(output_tokens)
    avg_input_tokens = statistics.mean(input_tokens)
    avg_output_tokens = statistics.mean(output_tokens)

    # Project for 50k emails
    n_samples = len(results)
    projected_time_50k_minutes = (avg_time_ms * TARGET_DATASET_SIZE) / 1000 / 60
    projected_cost_50k = avg_cost * TARGET_DATASET_SIZE

    return BenchmarkStats(
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
