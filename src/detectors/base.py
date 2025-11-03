"""Base abstract class for PII detectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

from src.utils.data_loader import Email


@dataclass
class DetectionResult:
    """Result of PII detection for a single email."""
    email_id: int
    has_pii: bool  # Binary prediction: does email contain PII?
    confidence: Optional[float] = None  # Optional confidence score
    time_ms: float = 0.0  # Time taken for detection (milliseconds)
    cost_usd: float = 0.0  # Cost of detection (USD)
    input_tokens: Optional[int] = None  # For API-based detectors
    output_tokens: Optional[int] = None  # For API-based detectors


class BaseDetector(ABC):
    """Abstract base class for all PII detectors."""

    def __init__(self, name: str):
        """
        Initialize detector.

        Args:
            name: Unique identifier for this detector instance
        """
        self.name = name

    @abstractmethod
    def detect(self, email: Email) -> DetectionResult:
        """
        Detect PII in an email.

        Args:
            email: Email object to analyze

        Returns:
            DetectionResult with prediction and metadata
        """
        pass

    def get_name(self) -> str:
        """Get the detector name."""
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
