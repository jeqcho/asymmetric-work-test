"""Cost calculation utilities for API usage."""

from pipeline.config import (
    HAIKU_INPUT_PRICE,
    HAIKU_OUTPUT_PRICE,
)


def calculate_haiku_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for Claude Haiku API call."""
    input_cost = input_tokens * HAIKU_INPUT_PRICE
    output_cost = output_tokens * HAIKU_OUTPUT_PRICE
    return input_cost + output_cost


def calculate_presidio_cost() -> float:
    """
    Calculate cost for Presidio (runs locally).

    Returns 0 for API cost. In production, could estimate EC2 costs,
    but for local execution the marginal cost is negligible.
    """
    return 0.0

