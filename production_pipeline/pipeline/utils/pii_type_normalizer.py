"""PII type normalization utilities."""

from typing import List


def normalize_pii_types(pii_str: str) -> List[str]:
    """
    Normalize PII type string to match ground truth format.
    
    Handles variations in format:
    - Case insensitivity
    - Bracket format [email address] -> email address
    - Whitespace normalization
    - "None" / empty string handling
    
    Args:
        pii_str: Raw PII types string from LLM response
        
    Returns:
        List of normalized PII type strings, empty list if no PII
    """
    if not pii_str:
        return []
    
    # Convert to string and strip
    pii_str = str(pii_str).strip()
    
    # Handle "None" case (case-insensitive)
    if pii_str.lower() in ['none', 'nan', '']:
        return []
    
    # Remove brackets if present (e.g., [email address] -> email address)
    # But preserve internal brackets in types like "[ssn, last 4]"
    # Split by comma first, then clean each part
    parts = pii_str.split(',')
    normalized_types = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Remove outer brackets but keep internal content
        if part.startswith('[') and part.endswith(']'):
            part = part[1:-1].strip()
        
        # Normalize to lowercase
        part = part.lower()
        
        # Skip if empty after normalization
        if part:
            normalized_types.append(part)
    
    return normalized_types

