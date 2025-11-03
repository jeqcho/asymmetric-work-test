"""Presidio-based PII detector with configurable confidence thresholds."""

import time
from typing import List, Dict, Any, Tuple
from presidio_analyzer import AnalyzerEngine, RecognizerResult, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider

from pipeline.utils.data_loader import Email
from pipeline.config import PRESIDIO_THRESHOLD


class PresidioDetector:
    """PII detector using Microsoft Presidio."""

    def __init__(self, confidence_threshold: float = PRESIDIO_THRESHOLD):
        """
        Initialize Presidio detector.

        Args:
            confidence_threshold: Minimum confidence score to flag PII (0-1)
                - Higher threshold (e.g., 0.8) = fewer flags = more lenient
                - Lower threshold (e.g., 0.3) = more flags = more strict
                - 0.0 = catches everything
        """
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

        # Create custom recognizers and add them to the analyzer
        custom_recognizers = self._create_custom_recognizers()

        # Create analyzer with NLP engine and custom recognizers
        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            registry=None  # Use default registry
        )

        # Add custom recognizers to the analyzer
        for recognizer in custom_recognizers:
            self.analyzer.registry.add_recognizer(recognizer)

        # Define PII entities to detect (Presidio entity names + custom ones)
        # Aligned with task requirements - detecting all specified PII types
        self.entities_to_detect = [
            # Built-in Presidio recognizers
            "PERSON",              # [full name] - but needs special handling (only with other PII)
            "US_SSN",              # [ssn] and [ssn, last 4]
            "US_DRIVER_LICENSE",   # [drivers license #]
            "US_PASSPORT",         # [passport #]
            "US_ITIN",             # [TIN] - Taxpayer ID
            "US_BANK_NUMBER",      # [bank account #]
            "CREDIT_CARD",         # [credit card #]
            "PHONE_NUMBER",        # [phone number]
            "EMAIL_ADDRESS",       # [email address]

            # Custom recognizers (added below)
            "CVV",                 # [cvv/cvc] - custom pattern
            "PASSWORD",            # [password] - keyword-based
            "USERNAME",            # [username] - keyword-based
            "IRS_PIN",             # [irs identity protection pin] - custom pattern
            "STUDENT_ID",          # [student identification #] - custom pattern
        ]

    def _create_custom_recognizers(self) -> List[PatternRecognizer]:
        """
        Create custom PII recognizers for entity types not supported by default Presidio.

        Returns:
            List of custom PatternRecognizer instances
        """
        recognizers = []

        # 1. CVV/CVC Recognizer
        cvv_recognizer = PatternRecognizer(
            supported_entity="CVV",
            name="CVV Recognizer",
            patterns=[
                Pattern(name="cvv_3_digits", regex=r"\b\d{3}\b", score=0.3),
                Pattern(name="cvv_4_digits", regex=r"\b\d{4}\b", score=0.3),
            ],
            context=["cvv", "cvc", "security code", "card code", "verification code"],
        )
        recognizers.append(cvv_recognizer)

        # 2. Password Recognizer
        password_recognizer = PatternRecognizer(
            supported_entity="PASSWORD",
            name="Password Recognizer",
            patterns=[
                Pattern(
                    name="password_with_value",
                    regex=r"(?:password|passwd|pwd|pw|pass|passphrase|passcode|secret|cred(?:ential)?s?)\s*[:=\s]+\s*\S+",
                    score=0.9
                ),
                Pattern(
                    name="password_keyword",
                    regex=r"\b(?:password|passwd|pwd|pw|pass|passphrase|passcode|secret|credential(?:s)?)\b",
                    score=0.5
                ),
                Pattern(
                    name="password_prompt",
                    regex=r"(?:enter|provide|your|my|the|new|old|current|temporary|temp)\s+(?:password|passwd|pwd|pw|pass)",
                    score=0.7
                ),
            ],
            context=[
                "password", "passwd", "pwd", "pw", "pass", "passphrase", "passcode",
                "credential", "credentials", "cred", "creds",
                "secret", "login", "signin", "sign-in", "authentication", "auth",
                "access", "account", "security", "key",
                "enter", "provide", "reset", "change", "update", "verify"
            ],
        )
        recognizers.append(password_recognizer)

        # 3. Username Recognizer
        username_recognizer = PatternRecognizer(
            supported_entity="USERNAME",
            name="Username Recognizer",
            patterns=[
                Pattern(
                    name="username_with_value",
                    regex=r"(?:user(?:name)?|login|userid|user[\s_-]?id|uid|account|id|handle)\s*[:=\s]+\s*\S+",
                    score=0.9
                ),
                Pattern(
                    name="username_keyword",
                    regex=r"\b(?:user(?:name)?|login|userid|user[\s_-]?id|uid|account(?:\s+name)?|handle)\b",
                    score=0.5
                ),
                Pattern(
                    name="username_prompt",
                    regex=r"(?:enter|provide|your|my|the|new)\s+(?:user(?:name)?|login|userid|user[\s_-]?id|account)",
                    score=0.7
                ),
                Pattern(
                    name="credential_id",
                    regex=r"\bid\b",
                    score=0.4
                ),
            ],
            context=[
                "username", "user", "login", "userid", "user id", "uid", "id",
                "account", "handle", "identifier", "name",
                "credential", "credentials", "cred", "creds",
                "signin", "sign-in", "authentication", "auth",
                "access", "security",
                "enter", "provide", "create", "register", "verify",
                "password", "passwd", "pwd", "pw", "pass"
            ],
        )
        recognizers.append(username_recognizer)

        # 4. IRS Identity Protection PIN Recognizer
        irs_pin_recognizer = PatternRecognizer(
            supported_entity="IRS_PIN",
            name="IRS PIN Recognizer",
            patterns=[
                Pattern(name="irs_6_digit_pin", regex=r"\b\d{6}\b", score=0.4),
            ],
            context=["irs", "identity protection", "ip pin", "tax pin", "irs pin"],
        )
        recognizers.append(irs_pin_recognizer)

        # 5. Student ID Recognizer
        student_id_recognizer = PatternRecognizer(
            supported_entity="STUDENT_ID",
            name="Student ID Recognizer",
            patterns=[
                Pattern(name="student_numeric", regex=r"\b\d{7,9}\b", score=0.3),
                Pattern(name="student_alpha_numeric", regex=r"\b[A-Z]{1,3}\d{6,8}\b", score=0.4),
            ],
            context=["student id", "student number", "student identification", "enrollment id", "university id"],
        )
        recognizers.append(student_id_recognizer)

        return recognizers

    def detect(self, email: Email) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Detect PII in email using Presidio.

        Args:
            email: Email object to analyze

        Returns:
            Tuple of (has_pii, entities) where entities is list of detected entities with text snippets
        """
        start_time = time.time()

        # Analyze message body for PII (using threshold 0.0 to get all results, then filter)
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

        # Apply special rule: [full name] only counts if paired with other PII (not EMAIL_ADDRESS)
        filtered_results = self._apply_person_rule(high_confidence_results)

        # Convert to list of dictionaries with extracted text snippets
        entities = []
        for result in filtered_results:
            text_snippet = email.message_body[result.start:result.end]
            entities.append({
                "entity_type": result.entity_type,
                "score": result.score,
                "start": result.start,
                "end": result.end,
                "text": text_snippet
            })

        # Check if any PII detected above threshold
        has_pii = len(filtered_results) > 0

        return has_pii, entities

    def _apply_person_rule(self, results: List[RecognizerResult]) -> List[RecognizerResult]:
        """
        Apply the special rule for PERSON entities.

        Rule: Only flag PERSON if it appears with another PII element (excluding EMAIL_ADDRESS).

        Args:
            results: List of detected entities above threshold

        Returns:
            Filtered list of entities with PERSON rule applied
        """
        # Get entity types (excluding PERSON)
        non_person_entities = [r for r in results if r.entity_type != "PERSON"]
        person_entities = [r for r in results if r.entity_type == "PERSON"]

        # Get non-PERSON, non-EMAIL entity types
        substantive_pii = [r for r in non_person_entities if r.entity_type != "EMAIL_ADDRESS"]

        # If there are substantive PII elements, include PERSON entities
        # Otherwise, exclude PERSON entities
        if substantive_pii:
            # Has substantive PII (SSN, credit card, phone, etc.) - keep PERSON
            return results
        else:
            # Only has EMAIL_ADDRESS or only PERSON - exclude PERSON
            return non_person_entities

