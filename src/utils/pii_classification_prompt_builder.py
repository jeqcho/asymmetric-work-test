"""Prompt building utilities for Claude PII type classification detectors."""

from typing import List, Optional, Dict, Any
from src.utils.data_loader import Email
from src.utils.prompt_builder import format_presidio_entities
from src.config import PII_DESCRIPTIONS


# Helper to convert bracket format to dataset format
def bracket_to_dataset_format(pii_type: str) -> str:
    """Convert [pii type] format to lowercase dataset format."""
    # Remove brackets and convert to lowercase
    if pii_type.startswith('[') and pii_type.endswith(']'):
        return pii_type[1:-1].strip().lower()
    return pii_type.lower().strip()


def build_zero_shot_pii_classification_prompt(email: Email) -> str:
    """
    Build a zero-shot prompt for PII type classification.
    
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
    
    # Build valid PII types list for reference (convert from bracket format to dataset format)
    valid_types = [bracket_to_dataset_format(pii_type) for pii_type in PII_DESCRIPTIONS.keys()]
    valid_types_text = ", ".join(valid_types)
    
    prompt = f"""You are a PII detection expert for email breach analysis.

TASK: Identify all Personal Identifiable Information (PII) types present in this email.

PII TYPES TO DETECT:
{pii_types_text}

VALID OUTPUT FORMATS:
- If no PII is found: "None"
- If PII is found: comma-separated list of PII types, e.g., "email address, phone number"
- Valid PII types: {valid_types_text}

IMPORTANT RULES:
1. Focus ONLY on the message body for PII detection
2. Ignore corporate/business information - we only care about PII
3. [full name] ONLY counts if it appears WITH another PII element (email addresses don't count)
4. Use lowercase and match the exact format shown in the valid types list above
5. Output only the PII types, nothing else

EMAIL TO ANALYZE:
Subject: {email.subject}
From: {email.from_addr}
To: {email.to_addr}
Message Body: {email.message_body}

Output the PII types found as a comma-separated list, or "None" if no PII is found. Do not provide any explanation, just the list."""

    return prompt


def build_5_shot_pii_classification_prompt(
    email: Email,
    no_pii_examples: List[Email],
    with_pii_examples: List[Email]
) -> str:
    """
    Build a 5-shot prompt for PII type classification with examples.
    
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
    
    # Build valid PII types list for reference (convert from bracket format to dataset format)
    valid_types = [bracket_to_dataset_format(pii_type) for pii_type in PII_DESCRIPTIONS.keys()]
    valid_types_text = ", ".join(valid_types)
    
    # Build examples
    examples_text = "Here are 5 examples to guide your analysis:\n\n"
    
    # Add no-PII examples
    for i, example in enumerate(no_pii_examples, 1):
        gt_types = example.get_pii_types() if example.has_pii() else []
        gt_output = ", ".join(gt_types) if gt_types else "None"
        examples_text += f"""EXAMPLE {i} (NO PII):
Subject: {example.subject}
From: {example.from_addr}
To: {example.to_addr}
Message Body: {example.message_body}
Ground Truth Data Elements: {example.data_elements if example.data_elements else 'None'}
Output: {gt_output}

"""
    
    # Add with-PII examples
    for i, example in enumerate(with_pii_examples, len(no_pii_examples) + 1):
        gt_types = example.get_pii_types()
        gt_output = ", ".join(gt_types) if gt_types else "None"
        examples_text += f"""EXAMPLE {i} (WITH PII):
Subject: {example.subject}
From: {example.from_addr}
To: {example.to_addr}
Message Body: {example.message_body}
Ground Truth Data Elements: {example.data_elements}
Output: {gt_output}

"""
    
    prompt = f"""You are a PII detection expert for email breach analysis.

TASK: Identify all Personal Identifiable Information (PII) types present in this email.

PII TYPES TO DETECT:
{pii_types_text}

VALID OUTPUT FORMATS:
- If no PII is found: "None"
- If PII is found: comma-separated list of PII types, e.g., "email address, phone number"
- Valid PII types: {valid_types_text}

IMPORTANT RULES:
1. Focus ONLY on the message body for PII detection
2. Ignore corporate/business information - we only care about PII
3. [full name] ONLY counts if it appears WITH another PII element (email addresses don't count)
4. Use lowercase and match the exact format shown in the valid types list above
5. Output only the PII types, nothing else

{examples_text}
Now analyze this email:

EMAIL TO ANALYZE:
Subject: {email.subject}
From: {email.from_addr}
To: {email.to_addr}
Message Body: {email.message_body}

Output the PII types found as a comma-separated list, or "None" if no PII is found. Do not provide any explanation, just the list."""

    return prompt


def build_zero_shot_pii_classification_prompt_with_presidio(email: Email) -> str:
    """
    Build a zero-shot prompt for PII type classification with Presidio entities.
    
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
    
    # Build valid PII types list for reference (convert from bracket format to dataset format)
    valid_types = [bracket_to_dataset_format(pii_type) for pii_type in PII_DESCRIPTIONS.keys()]
    valid_types_text = ", ".join(valid_types)
    
    # Format Presidio entities
    presidio_output = format_presidio_entities(email.presidio_entities)
    
    prompt = f"""You are a PII detection expert for email breach analysis.

TASK: Identify all Personal Identifiable Information (PII) types present in this email.

PII TYPES TO DETECT:
{pii_types_text}

VALID OUTPUT FORMATS:
- If no PII is found: "None"
- If PII is found: comma-separated list of PII types, e.g., "email address, phone number"
- Valid PII types: {valid_types_text}

IMPORTANT RULES:
1. Focus ONLY on the message body for PII detection
2. Ignore corporate/business information - we only care about PII
3. [full name] ONLY counts if it appears WITH another PII element (email addresses don't count)
4. Use lowercase and match the exact format shown in the valid types list above
5. Output only the PII types, nothing else

PRESIDIO OUTPUT:
The Presidio tool has analyzed this email and found the following entities:
{presidio_output}

CRITICAL: Presidio may miss PII. Even if Presidio found nothing or only low-confidence matches, you must carefully review the entire email message body for PII of concern. Presidio's output is meant to help guide your analysis, but you should check for any PII that Presidio might have missed.

EMAIL TO ANALYZE:
Subject: {email.subject}
From: {email.from_addr}
To: {email.to_addr}
Message Body: {email.message_body}

Output the PII types found as a comma-separated list, or "None" if no PII is found. Do not provide any explanation, just the list."""

    return prompt


def build_5_shot_pii_classification_prompt_with_presidio(
    email: Email,
    no_pii_examples: List[Email],
    with_pii_examples: List[Email]
) -> str:
    """
    Build a 5-shot prompt for PII type classification with examples and Presidio entities.
    
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
    
    # Build valid PII types list for reference (convert from bracket format to dataset format)
    valid_types = [bracket_to_dataset_format(pii_type) for pii_type in PII_DESCRIPTIONS.keys()]
    valid_types_text = ", ".join(valid_types)
    
    # Build examples with Presidio output
    examples_text = "Here are 5 examples to guide your analysis:\n\n"
    
    # Add no-PII examples
    for i, example in enumerate(no_pii_examples, 1):
        presidio_output = format_presidio_entities(example.presidio_entities)
        gt_types = example.get_pii_types() if example.has_pii() else []
        gt_output = ", ".join(gt_types) if gt_types else "None"
        examples_text += f"""EXAMPLE {i} (NO PII):
Subject: {example.subject}
From: {example.from_addr}
To: {example.to_addr}
Message Body: {example.message_body}
Presidio Output: {presidio_output}
Ground Truth Data Elements: {example.data_elements if example.data_elements else 'None'}
Output: {gt_output}

"""
    
    # Add with-PII examples
    for i, example in enumerate(with_pii_examples, len(no_pii_examples) + 1):
        presidio_output = format_presidio_entities(example.presidio_entities)
        gt_types = example.get_pii_types()
        gt_output = ", ".join(gt_types) if gt_types else "None"
        examples_text += f"""EXAMPLE {i} (WITH PII):
Subject: {example.subject}
From: {example.from_addr}
To: {example.to_addr}
Message Body: {example.message_body}
Presidio Output: {presidio_output}
Ground Truth Data Elements: {example.data_elements}
Output: {gt_output}

"""
    
    # Format Presidio entities for the email being analyzed
    presidio_output = format_presidio_entities(email.presidio_entities)
    
    prompt = f"""You are a PII detection expert for email breach analysis.

TASK: Identify all Personal Identifiable Information (PII) types present in this email.

PII TYPES TO DETECT:
{pii_types_text}

VALID OUTPUT FORMATS:
- If no PII is found: "None"
- If PII is found: comma-separated list of PII types, e.g., "email address, phone number"
- Valid PII types: {valid_types_text}

IMPORTANT RULES:
1. Focus ONLY on the message body for PII detection
2. Ignore corporate/business information - we only care about PII
3. [full name] ONLY counts if it appears WITH another PII element (email addresses don't count)
4. Use lowercase and match the exact format shown in the valid types list above
5. Output only the PII types, nothing else

CRITICAL: Presidio may miss PII. Even if Presidio found nothing or only low-confidence matches, you must carefully review the entire email message body for PII of concern. Presidio's output is meant to help guide your analysis, but you should check for any PII that Presidio might have missed.

{examples_text}
Now analyze this email:

EMAIL TO ANALYZE:
Subject: {email.subject}
From: {email.from_addr}
To: {email.to_addr}
Message Body: {email.message_body}
Presidio Output: {presidio_output}

Output the PII types found as a comma-separated list, or "None" if no PII is found. Do not provide any explanation, just the list."""

    return prompt

