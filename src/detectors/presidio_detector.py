"""Presidio-based PII detector with configurable confidence thresholds."""

import time
from typing import List
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from src.detectors.base import BaseDetector, DetectionResult
from src.utils.data_loader import Email
from src.utils.cost_calculator import calculate_presidio_cost


class PresidioDetector(BaseDetector):
    """PII detector using Microsoft Presidio."""

    def __init__(self, name: str, confidence_threshold: float):
        """
        Initialize Presidio detector.

        Args:
            name: Detector instance name (e.g., 'presidio_lax')
            confidence_threshold: Minimum confidence score to flag PII (0-1)
                - Higher threshold (e.g., 0.8) = fewer flags = more lenient
                - Lower threshold (e.g., 0.3) = more flags = more strict
        """
        super().__init__(name)
        self.confidence_threshold = confidence_threshold

        # Initialize Presidio analyzer
        # Using spaCy NLP engine that was installed with presidio-analyzer
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
        }

        # Create NLP engine provider
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        # Create analyzer with NLP engine
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

        # Define PII entities to detect (Presidio entity names)
        # Mapping task PII types to Presidio recognizers
        self.entities_to_detect = [
            "PERSON",              # Full names
            "US_SSN",              # Social security numbers
            "US_DRIVER_LICENSE",   # Driver's license
            "US_PASSPORT",         # Passport
            "US_ITIN",             # Taxpayer ID (similar to SSN)
            "US_BANK_NUMBER",      # Bank account
            "CREDIT_CARD",         # Credit card
            "PHONE_NUMBER",        # Phone numbers
            "EMAIL_ADDRESS",       # Email addresses
            "DATE_TIME",           # Dates (for DOB)
            "LOCATION",            # Addresses
            "IBAN_CODE",           # International bank account
            "CRYPTO",              # Crypto wallet addresses
            "IP_ADDRESS",          # IP addresses
            "MEDICAL_LICENSE",     # Medical info
            "URL",                 # URLs that might contain credentials
        ]

    def detect(self, email: Email) -> DetectionResult:
        """
        Detect PII in email using Presidio.

        Args:
            email: Email object to analyze

        Returns:
            DetectionResult with binary prediction
        """
        start_time = time.time()

        # Analyze message body for PII
        results: List[RecognizerResult] = self.analyzer.analyze(
            text=email.message_body,
            entities=self.entities_to_detect,
            language="en"
        )

        # Filter results by confidence threshold
        high_confidence_results = [
            result for result in results
            if result.score >= self.confidence_threshold
        ]

        # Check if any PII detected above threshold
        has_pii = len(high_confidence_results) > 0

        # Calculate elapsed time
        elapsed_ms = (time.time() - start_time) * 1000

        # Presidio runs locally, so cost is 0
        cost = calculate_presidio_cost()

        # Get max confidence score if PII detected
        confidence = max(
            [result.score for result in high_confidence_results],
            default=0.0
        ) if high_confidence_results else 0.0

        return DetectionResult(
            email_id=email.id,
            has_pii=has_pii,
            confidence=confidence,
            time_ms=elapsed_ms,
            cost_usd=cost
        )


def create_presidio_detectors() -> List[PresidioDetector]:
    """
    Create all three Presidio detector instances.

    Returns:
        List of [presidio_lax, presidio_moderate, presidio_strict]
    """
    from src.config import PRESIDIO_THRESHOLDS

    return [
        PresidioDetector("presidio_lax", PRESIDIO_THRESHOLDS["lax"]),
        PresidioDetector("presidio_moderate", PRESIDIO_THRESHOLDS["moderate"]),
        PresidioDetector("presidio_strict", PRESIDIO_THRESHOLDS["strict"])
    ]
