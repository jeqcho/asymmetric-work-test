"""Presidio-based PII detector with configurable confidence thresholds."""

import time
from typing import List
from presidio_analyzer import AnalyzerEngine, RecognizerResult, PatternRecognizer, Pattern
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
            "US_DRIVER_LICENSE",   # [drivers license #] - re-enabled per requirement
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
        # Detects 3-4 digit card security codes with context words
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
        # Detects mentions of passwords with comprehensive keyword variants
        # Covers: password, passwd, pwd, pw, pass, passphrase, passcode, secret, credentials
        password_recognizer = PatternRecognizer(
            supported_entity="PASSWORD",
            name="Password Recognizer",
            patterns=[
                # High confidence: Labeled password with value (colon/equals/spaces)
                # Matches: "password: secret", "pwd=abc123", "pw  westgasx"
                Pattern(
                    name="password_with_value",
                    regex=r"(?:password|passwd|pwd|pw|pass|passphrase|passcode|secret|cred(?:ential)?s?)\s*[:=\s]+\s*\S+",
                    score=0.9
                ),
                # Medium confidence: Standalone keywords (context-dependent)
                # Matches: "password", "pwd", "pw", etc.
                Pattern(
                    name="password_keyword",
                    regex=r"\b(?:password|passwd|pwd|pw|pass|passphrase|passcode|secret|credential(?:s)?)\b",
                    score=0.5
                ),
                # Medium-high confidence: Common password prompt patterns
                # Matches: "enter password", "provide pwd", "your pw is"
                Pattern(
                    name="password_prompt",
                    regex=r"(?:enter|provide|your|my|the|new|old|current|temporary|temp)\s+(?:password|passwd|pwd|pw|pass)",
                    score=0.7
                ),
            ],
            context=[
                # Common password keywords
                "password", "passwd", "pwd", "pw", "pass", "passphrase", "passcode",
                # Related terms
                "credential", "credentials", "cred", "creds",
                "secret", "login", "signin", "sign-in", "authentication", "auth",
                "access", "account", "security", "key",
                # Action words
                "enter", "provide", "reset", "change", "update", "verify"
            ],
        )
        recognizers.append(password_recognizer)

        # 3. Username Recognizer
        # Detects mentions of usernames with comprehensive keyword variants
        # Covers: username, user, userid, uid, login, id, account, name, handle
        username_recognizer = PatternRecognizer(
            supported_entity="USERNAME",
            name="Username Recognizer",
            patterns=[
                # High confidence: Labeled username with value (colon/equals/spaces)
                # Matches: "username: john", "user=alice", "id   pallen", "login bob"
                Pattern(
                    name="username_with_value",
                    regex=r"(?:user(?:name)?|login|userid|user[\s_-]?id|uid|account|id|handle)\s*[:=\s]+\s*\S+",
                    score=0.9
                ),
                # Medium confidence: Standalone keywords (context-dependent)
                # Matches: "username", "login", "userid", etc.
                Pattern(
                    name="username_keyword",
                    regex=r"\b(?:user(?:name)?|login|userid|user[\s_-]?id|uid|account(?:\s+name)?|handle)\b",
                    score=0.5
                ),
                # Medium-high confidence: Common username prompt patterns
                # Matches: "enter username", "provide login", "your id is"
                Pattern(
                    name="username_prompt",
                    regex=r"(?:enter|provide|your|my|the|new)\s+(?:user(?:name)?|login|userid|user[\s_-]?id|account)",
                    score=0.7
                ),
                # Special pattern for "id" in credential context (must be near password)
                # Matches: "id" when within 50 chars of password keywords
                Pattern(
                    name="credential_id",
                    regex=r"\bid\b",
                    score=0.4
                ),
            ],
            context=[
                # Common username keywords
                "username", "user", "login", "userid", "user id", "uid", "id",
                # Related terms
                "account", "handle", "identifier", "name",
                "credential", "credentials", "cred", "creds",
                "signin", "sign-in", "authentication", "auth",
                "access", "security",
                # Action words
                "enter", "provide", "create", "register", "verify",
                # Paired with password (boosts "id" detection)
                "password", "passwd", "pwd", "pw", "pass"
            ],
        )
        recognizers.append(username_recognizer)

        # 4. IRS Identity Protection PIN Recognizer
        # 6-digit PIN used by IRS for identity protection
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
        # Detects student identification numbers (various formats)
        student_id_recognizer = PatternRecognizer(
            supported_entity="STUDENT_ID",
            name="Student ID Recognizer",
            patterns=[
                # Common formats: 7-9 digits or alphanumeric
                Pattern(name="student_numeric", regex=r"\b\d{7,9}\b", score=0.3),
                Pattern(name="student_alpha_numeric", regex=r"\b[A-Z]{1,3}\d{6,8}\b", score=0.4),
            ],
            context=["student id", "student number", "student identification", "enrollment id", "university id"],
        )
        recognizers.append(student_id_recognizer)

        return recognizers

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

        # Apply special rule: [full name] only counts if paired with other PII (not EMAIL_ADDRESS)
        # Per task requirements: "Only flag [full name] if it appears in combination with
        # another PII element, NOT inclusive of [email address]."
        filtered_results = self._apply_person_rule(high_confidence_results)

        # Check if any PII detected above threshold
        has_pii = len(filtered_results) > 0

        # Calculate elapsed time
        elapsed_ms = (time.time() - start_time) * 1000

        # Presidio runs locally, so cost is 0
        cost = calculate_presidio_cost()

        # Get max confidence score if PII detected
        confidence = max(
            [result.score for result in filtered_results],
            default=0.0
        ) if filtered_results else 0.0

        return DetectionResult(
            email_id=email.id,
            has_pii=has_pii,
            confidence=confidence,
            time_ms=elapsed_ms,
            cost_usd=cost
        )

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


def create_presidio_detectors() -> List[PresidioDetector]:
    """
    Create all three Presidio V2 detector instances.

    V2 improvements:
    - Removed DATE_TIME, LOCATION, URL entities (reduces false positives)
    - Added 5 custom recognizers: CVV, PASSWORD, USERNAME, IRS_PIN, STUDENT_ID
    - Implemented special PERSON rule (only with other PII, not email)
    - Re-enabled US_DRIVER_LICENSE per task requirements

    Returns:
        List of [presidio_v2_lax, presidio_v2_moderate, presidio_v2_strict]
    """
    from src.config import PRESIDIO_THRESHOLDS

    return [
        PresidioDetector("presidio_v2_lax", PRESIDIO_THRESHOLDS["lax"]),
        PresidioDetector("presidio_v2_moderate", PRESIDIO_THRESHOLDS["moderate"]),
        PresidioDetector("presidio_v2_strict", PRESIDIO_THRESHOLDS["strict"])
    ]
