"""Prompt building utilities for Claude detectors."""

from typing import List, Optional, Dict, Any
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


def format_presidio_entities(entities: Optional[List[Dict[str, Any]]]) -> str:
    """
    Format Presidio entities as a readable string.
    
    Args:
        entities: List of entity dictionaries with entity_type, score, and text
        
    Returns:
        Formatted string of entities
    """
    if not entities or len(entities) == 0:
        return "None"
    
    formatted_entities = []
    for entity in entities:
        entity_type = entity.get("entity_type", "UNKNOWN")
        score = entity.get("score", 0.0)
        text = entity.get("text", "")
        formatted_entities.append(f"{entity_type} ({score:.2f}): '{text}'")
    
    return "\n".join(formatted_entities)


def build_zero_shot_prompt_with_presidio(email: Email) -> str:
    """
    Build a zero-shot prompt for PII detection with Presidio entities.
    
    Args:
        email: Email to analyze (should have presidio_entities populated)
        
    Returns:
        Formatted prompt string
    """
    # Build PII types description
    pii_types_text = "\n".join([
        f"{i+1}. {pii_type} - {description}"
        for i, (pii_type, description) in enumerate(PII_DESCRIPTIONS.items())
    ])
    
    # Format Presidio entities
    presidio_output = format_presidio_entities(email.presidio_entities)
    
    prompt = f"""You are a PII detection expert for email breach analysis.

TASK: Determine if this email contains ANY Personal Identifiable Information (PII).

PII TYPES TO DETECT:
{pii_types_text}

IMPORTANT RULES:
1. Focus ONLY on the message body for PII detection
2. Ignore corporate/business information - we only care about PII
3. [full name] ONLY counts if it appears WITH another PII element (email addresses don't count)
4. Flag if you have any suspicion it could be PII

PRESIDIO OUTPUT:
The Presidio tool has analyzed this email and found the following entities:
{presidio_output}

CRITICAL: Presidio may miss PII. Even if Presidio found nothing or only low-confidence matches, you must carefully review the entire email message body for PII of concern. Presidio's output is meant to help guide your analysis, but you should check for any PII that Presidio might have missed.

EMAIL TO ANALYZE:
Subject: {email.subject}
From: {email.from_addr}
To: {email.to_addr}
Message Body: {email.message_body}

Answer with ONLY "YES" if the email contains PII, or "NO" if it doesn't. Do not provide any explanation, just YES or NO."""

    return prompt


def build_5_shot_prompt_with_presidio(
    email: Email,
    no_pii_examples: List[Email],
    with_pii_examples: List[Email]
) -> str:
    """
    Build a 5-shot prompt for PII detection with examples and Presidio entities.
    
    Args:
        email: Email to analyze (should have presidio_entities populated)
        no_pii_examples: 3 example emails without PII (should have presidio_entities populated)
        with_pii_examples: 2 example emails with PII (should have presidio_entities populated)
        
    Returns:
        Formatted prompt string
    """
    # Build PII types description
    pii_types_text = "\n".join([
        f"{i+1}. {pii_type} - {description}"
        for i, (pii_type, description) in enumerate(PII_DESCRIPTIONS.items())
    ])
    
    # Build examples with Presidio output
    examples_text = "Here are 5 examples to guide your analysis:\n\n"
    
    # Add no-PII examples
    for i, example in enumerate(no_pii_examples, 1):
        presidio_output = format_presidio_entities(example.presidio_entities)
        examples_text += f"""EXAMPLE {i} (NO PII):
Subject: {example.subject}
From: {example.from_addr}
To: {example.to_addr}
Message Body: {example.message_body}
Presidio Output: {presidio_output}
Data Elements: {example.data_elements if example.data_elements else 'None'}
Answer: NO

"""
    
    # Add with-PII examples
    for i, example in enumerate(with_pii_examples, len(no_pii_examples) + 1):
        presidio_output = format_presidio_entities(example.presidio_entities)
        examples_text += f"""EXAMPLE {i} (WITH PII):
Subject: {example.subject}
From: {example.from_addr}
To: {example.to_addr}
Message Body: {example.message_body}
Presidio Output: {presidio_output}
Data Elements: {example.data_elements}
Answer: YES

"""
    
    # Format Presidio entities for the email being analyzed
    presidio_output = format_presidio_entities(email.presidio_entities)
    
    prompt = f"""You are a PII detection expert for email breach analysis.

TASK: Determine if this email contains ANY Personal Identifiable Information (PII).

PII TYPES TO DETECT:
{pii_types_text}

IMPORTANT RULES:
1. Focus ONLY on the message body for PII detection
2. Ignore corporate/business information - we only care about PII
3. [full name] ONLY counts if it appears WITH another PII element (email addresses don't count)
4. Flag if you have any suspicion it could be PII

CRITICAL: Presidio may miss PII. Even if Presidio found nothing or only low-confidence matches, you must carefully review the entire email message body for PII of concern. Presidio's output is meant to help guide your analysis, but you should check for any PII that Presidio might have missed.

{examples_text}
Now analyze this email:

EMAIL TO ANALYZE:
Subject: {email.subject}
From: {email.from_addr}
To: {email.to_addr}
Message Body: {email.message_body}
Presidio Output: {presidio_output}

Answer with ONLY "YES" if the email contains PII, or "NO" if it doesn't. Do not provide any explanation, just YES or NO."""

    return prompt
