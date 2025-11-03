"""
Email deduplication utilities.

This module provides functionality to detect and remove duplicate emails
using two strategies:
1. Exact duplicate detection via message body hashing
2. Forwarded email duplicate detection by removing forwarding headers

Author: Asymmetric Security
Date: 2025-11-02
"""

import hashlib
import re
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

from src.utils.data_loader import Email


@dataclass
class DeduplicationStats:
    """Statistics about the deduplication process."""
    total_emails: int
    exact_duplicates_removed: int
    forwarded_duplicates_removed: int
    unique_emails: int
    exact_duplicate_groups: int
    forwarded_duplicate_groups: int
    largest_duplicate_group: int
    forwarded_emails_detected: int

    @property
    def total_duplicates_removed(self) -> int:
        """Total number of duplicate emails removed."""
        return self.exact_duplicates_removed + self.forwarded_duplicates_removed

    @property
    def deduplication_rate(self) -> float:
        """Percentage of emails removed as duplicates."""
        return (self.total_duplicates_removed / self.total_emails) * 100 if self.total_emails > 0 else 0.0

    def __str__(self) -> str:
        """Human-readable statistics summary."""
        return f"""
Deduplication Statistics:
========================
Total Emails:                    {self.total_emails:,}
Unique Emails:                   {self.unique_emails:,}
Total Duplicates Removed:        {self.total_duplicates_removed:,} ({self.deduplication_rate:.2f}%)

Exact Duplicate Detection:
  - Duplicates Removed:          {self.exact_duplicates_removed:,}
  - Duplicate Groups:            {self.exact_duplicate_groups:,}

Forwarded Email Detection:
  - Forwarded Emails Found:      {self.forwarded_emails_detected:,}
  - Additional Duplicates:       {self.forwarded_duplicates_removed:,}
  - Duplicate Groups:            {self.forwarded_duplicate_groups:,}

Largest Duplicate Group:         {self.largest_duplicate_group} emails
"""


class EmailDeduplicator:
    """Handles email deduplication using hash-based and forwarded email detection."""

    # Patterns to detect forwarded emails
    FORWARDING_PATTERNS = [
        r'-{20,}\s*Forwarded by[^\n]*\n',
        r'-{5,}Original Message-{5,}',
        r'>\s*From:',
        r'>\s*Sent:',
        r'>\s*To:',
        r'>\s*Subject:',
        r'>\s*Date:',
    ]

    def __init__(self):
        """Initialize the deduplicator."""
        self.hash_to_emails: Dict[str, List[Email]] = defaultdict(list)
        self.cleaned_hash_to_emails: Dict[str, List[Email]] = defaultdict(list)

    def hash_message(self, message: str) -> str:
        """
        Generate SHA-256 hash of a message.

        Args:
            message: The message body to hash

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(message.encode('utf-8')).hexdigest()

    def is_forwarded(self, message_body: str) -> bool:
        """
        Detect if an email appears to be forwarded.

        Args:
            message_body: The email message body

        Returns:
            True if forwarding patterns are detected
        """
        for pattern in self.FORWARDING_PATTERNS:
            if re.search(pattern, message_body, re.IGNORECASE | re.MULTILINE):
                return True
        return False

    def clean_forwarded_email(self, message_body: str) -> str:
        """
        Remove forwarding headers and metadata from an email.

        Args:
            message_body: The original message body

        Returns:
            Cleaned message body with forwarding headers removed
        """
        cleaned = message_body

        # Remove "Forwarded by" header blocks
        cleaned = re.sub(
            r'-{20,}\s*Forwarded by[^\n]*\n[^\n]*\n',
            '',
            cleaned,
            flags=re.IGNORECASE
        )

        # Remove "Original Message" separators
        cleaned = re.sub(
            r'-{5,}Original Message-{5,}',
            '',
            cleaned,
            flags=re.IGNORECASE
        )

        # Remove quoted metadata lines (From:, To:, Sent:, etc.)
        metadata_patterns = [
            r'>\s*From:[^\n]*\n',
            r'>\s*Sent:[^\n]*\n',
            r'>\s*To:[^\n]*\n',
            r'>\s*Cc:[^\n]*\n',
            r'>\s*Subject:[^\n]*\n',
            r'>\s*Date:[^\n]*\n',
        ]

        for pattern in metadata_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Remove excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = cleaned.strip()

        return cleaned

    def find_exact_duplicates(self, emails: List[Email]) -> Tuple[List[Email], Dict[str, List[Email]]]:
        """
        Find exact duplicates by hashing message bodies.

        Args:
            emails: List of emails to process

        Returns:
            Tuple of (unique_emails, duplicate_groups)
        """
        self.hash_to_emails.clear()

        # Group emails by hash
        for email in emails:
            msg_hash = self.hash_message(email.message_body)
            self.hash_to_emails[msg_hash].append(email)

        # Extract unique emails (first email from each hash group)
        unique_emails = [emails[0] for emails in self.hash_to_emails.values()]

        # Extract duplicate groups (groups with more than 1 email)
        duplicate_groups = {
            hash_val: emails
            for hash_val, emails in self.hash_to_emails.items()
            if len(emails) > 1
        }

        return unique_emails, duplicate_groups

    def find_forwarded_duplicates(self, emails: List[Email]) -> Tuple[List[Email], Dict[str, List[Email]], int]:
        """
        Find duplicates among forwarded emails by cleaning and re-hashing.

        Args:
            emails: List of emails to process (typically already deduplicated for exact matches)

        Returns:
            Tuple of (unique_emails, duplicate_groups, forwarded_count)
        """
        self.cleaned_hash_to_emails.clear()
        forwarded_count = 0

        # Group emails by cleaned hash
        for email in emails:
            if self.is_forwarded(email.message_body):
                forwarded_count += 1
                cleaned_body = self.clean_forwarded_email(email.message_body)
                msg_hash = self.hash_message(cleaned_body)
            else:
                msg_hash = self.hash_message(email.message_body)

            self.cleaned_hash_to_emails[msg_hash].append(email)

        # Extract unique emails (first email from each hash group)
        unique_emails = [emails[0] for emails in self.cleaned_hash_to_emails.values()]

        # Extract duplicate groups (groups with more than 1 email)
        duplicate_groups = {
            hash_val: emails
            for hash_val, emails in self.cleaned_hash_to_emails.items()
            if len(emails) > 1
        }

        return unique_emails, duplicate_groups, forwarded_count

    def deduplicate(self, emails: List[Email]) -> Tuple[List[Email], DeduplicationStats]:
        """
        Perform full deduplication: exact + forwarded duplicate detection.

        Args:
            emails: List of emails to deduplicate

        Returns:
            Tuple of (unique_emails, statistics)
        """
        total_emails = len(emails)

        # Step 1: Find exact duplicates
        unique_after_exact, exact_dup_groups = self.find_exact_duplicates(emails)
        exact_duplicates_removed = total_emails - len(unique_after_exact)

        # Step 2: Find forwarded duplicates among the remaining unique emails
        unique_final, forwarded_dup_groups, forwarded_count = self.find_forwarded_duplicates(unique_after_exact)
        forwarded_duplicates_removed = len(unique_after_exact) - len(unique_final)

        # Calculate statistics
        largest_group = 0
        if exact_dup_groups:
            largest_group = max(len(emails) for emails in exact_dup_groups.values())
        if forwarded_dup_groups:
            largest_group = max(largest_group, max(len(emails) for emails in forwarded_dup_groups.values()))

        stats = DeduplicationStats(
            total_emails=total_emails,
            exact_duplicates_removed=exact_duplicates_removed,
            forwarded_duplicates_removed=forwarded_duplicates_removed,
            unique_emails=len(unique_final),
            exact_duplicate_groups=len(exact_dup_groups),
            forwarded_duplicate_groups=len(forwarded_dup_groups),
            largest_duplicate_group=largest_group,
            forwarded_emails_detected=forwarded_count
        )

        return unique_final, stats

    def get_duplicate_mapping(self, emails: List[Email]) -> Dict[str, List[str]]:
        """
        Create a mapping of unique email IDs to their duplicate IDs.

        Args:
            emails: List of emails to analyze

        Returns:
            Dictionary mapping representative email ID to list of duplicate IDs
        """
        # First pass: exact duplicates
        unique_after_exact, _ = self.find_exact_duplicates(emails)

        # Second pass: forwarded duplicates
        unique_final, _, _ = self.find_forwarded_duplicates(unique_after_exact)

        # Build mapping from the cleaned hash groups
        duplicate_mapping = {}
        for hash_val, email_group in self.cleaned_hash_to_emails.items():
            if len(email_group) > 1:
                representative_id = email_group[0].id
                duplicate_ids = [e.id for e in email_group]
                duplicate_mapping[representative_id] = duplicate_ids

        return duplicate_mapping
