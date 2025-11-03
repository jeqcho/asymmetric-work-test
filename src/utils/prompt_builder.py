"""Prompt building utilities for Claude detectors."""

from typing import List
from src.utils.data_loader import Email
from src.config import PII_DESCRIPTIONS


def build_zero_shot_prompt(email: Email) -> str:
    """
    Build a zero-shot prompt for PII detection.

    Args:
        email: Email to analyze

    Returns:
        Formatted prompt string
    """
    # Build PII types description
    pii_types_text = "\n".join([
        f"{i+1}. {pii_type} - {description}"
        for i, (pii_type, description) in enumerate(PII_DESCRIPTIONS.items())
    ])

    prompt = f"""You are a PII detection expert for email breach analysis.

TASK: Determine if this email contains ANY Personal Identifiable Information (PII).

PII TYPES TO DETECT:
{pii_types_text}

IMPORTANT RULES:
1. Focus ONLY on the message body for PII detection
2. Ignore corporate/business information - we only care about PII
3. [full name] ONLY counts if it appears WITH another PII element (email addresses don't count)
4. Flag if you have any suspicion it could be PII

EMAIL TO ANALYZE:
Subject: {email.subject}
From: {email.from_addr}
To: {email.to_addr}
Message Body: {email.message_body}

Answer with ONLY "YES" if the email contains PII, or "NO" if it doesn't. Do not provide any explanation, just YES or NO."""

    return prompt


def build_5_shot_prompt(
    email: Email,
    no_pii_examples: List[Email],
    with_pii_examples: List[Email]
) -> str:
    """
    Build a 5-shot prompt for PII detection with examples.

    Args:
        email: Email to analyze
        no_pii_examples: 3 example emails without PII
        with_pii_examples: 2 example emails with PII

    Returns:
        Formatted prompt string
    """
    # Build PII types description
    pii_types_text = "\n".join([
        f"{i+1}. {pii_type} - {description}"
        for i, (pii_type, description) in enumerate(PII_DESCRIPTIONS.items())
    ])

    # Build examples
    examples_text = "Here are 5 examples to guide your analysis:\n\n"

    # Add no-PII examples
    for i, example in enumerate(no_pii_examples, 1):
        examples_text += f"""EXAMPLE {i} (NO PII):
Subject: {example.subject}
From: {example.from_addr}
To: {example.to_addr}
Message Body: {example.message_body}
Data Elements: {example.data_elements if example.data_elements else 'None'}
Answer: NO

"""

    # Add with-PII examples
    for i, example in enumerate(with_pii_examples, len(no_pii_examples) + 1):
        examples_text += f"""EXAMPLE {i} (WITH PII):
Subject: {example.subject}
From: {example.from_addr}
To: {example.to_addr}
Message Body: {example.message_body}
Data Elements: {example.data_elements}
Answer: YES

"""

    prompt = f"""You are a PII detection expert for email breach analysis.

TASK: Determine if this email contains ANY Personal Identifiable Information (PII).

PII TYPES TO DETECT:
{pii_types_text}

IMPORTANT RULES:
1. Focus ONLY on the message body for PII detection
2. Ignore corporate/business information - we only care about PII
3. [full name] ONLY counts if it appears WITH another PII element (email addresses don't count)
4. Flag if you have any suspicion it could be PII

{examples_text}
Now analyze this email:

EMAIL TO ANALYZE:
Subject: {email.subject}
From: {email.from_addr}
To: {email.to_addr}
Message Body: {email.message_body}

Answer with ONLY "YES" if the email contains PII, or "NO" if it doesn't. Do not provide any explanation, just YES or NO."""

    return prompt
