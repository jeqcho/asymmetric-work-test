"""Data loading and parsing utilities."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from pipeline.config import LABELED_DATASET_PATH


@dataclass
class Email:
    """Represents a single email."""
    id: int
    subject: str
    from_addr: str
    to_addr: str
    cc: str
    bcc: str
    message_body: str
    data_elements: Optional[str] = None  # PII labels (only in labeled dataset)
    presidio_entities: Optional[List[Dict[str, Any]]] = None  # Presidio detected entities

    def has_pii(self) -> bool:
        """Check if email contains PII based on ground truth labels."""
        if self.data_elements is None:
            return False

        # Handle various representations of "no PII"
        if pd.isna(self.data_elements):
            return False

        data_str = str(self.data_elements).strip().lower()
        if data_str in ['', 'none', 'nan']:
            return False

        return True

    def get_pii_types(self) -> List[str]:
        """Extract list of PII types from data_elements column."""
        if not self.has_pii():
            return []

        # Parse comma-separated PII types
        pii_types = str(self.data_elements).split(',')
        return [pii.strip().lower() for pii in pii_types if pii.strip()]


def load_csv_dataset(csv_path: Path) -> List[Email]:
    """Load emails from an arbitrary CSV file."""
    df = pd.read_csv(csv_path)

    emails = []
    for _, row in df.iterrows():
        email = Email(
            id=int(row['ID']),
            subject=str(row.get('Subject', '')),
            from_addr=str(row.get('From', '')),
            to_addr=str(row.get('To', '')),
            cc=str(row.get('CC', '')),
            bcc=str(row.get('BCC', '')),
            message_body=str(row.get('Message Body', '')),
            data_elements=row.get('Data Elements') if 'Data Elements' in row else None
        )
        emails.append(email)

    return emails


def load_labeled_dataset() -> List[Email]:
    """Load the 250-sample gold-labeled dataset."""
    if not LABELED_DATASET_PATH.exists():
        return []
    
    df = pd.read_csv(LABELED_DATASET_PATH)

    emails = []
    for _, row in df.iterrows():
        email = Email(
            id=int(row['ID']),
            subject=str(row.get('Subject', '')),
            from_addr=str(row.get('From', '')),
            to_addr=str(row.get('To', '')),
            cc=str(row.get('CC', '')),
            bcc=str(row.get('BCC', '')),
            message_body=str(row.get('Message Body', '')),
            data_elements=row.get('Data Elements')
        )
        emails.append(email)

    return emails


def get_5_shot_examples(
    current_email_id: int,
    labeled_dataset: List[Email],
    random_seed: int = 42
) -> Tuple[List[Email], List[Email]]:
    """
    Sample 5-shot examples for few-shot learning.

    Returns:
        Tuple of (no_pii_examples, with_pii_examples)
        - no_pii_examples: 3 emails without PII
        - with_pii_examples: 2 emails with PII
    """
    # Create a pool excluding the current email
    pool = [email for email in labeled_dataset if email.id != current_email_id]

    # Split into no-PII and with-PII pools
    no_pii_pool = [email for email in pool if not email.has_pii()]
    with_pii_pool = [email for email in pool if email.has_pii()]

    # Set random seed for reproducibility
    rng = np.random.RandomState(random_seed)

    # Sample examples (handle cases where pools might be small)
    n_no_pii = min(3, len(no_pii_pool))
    n_with_pii = min(2, len(with_pii_pool))
    
    if n_no_pii > 0:
        no_pii_indices = rng.choice(len(no_pii_pool), size=n_no_pii, replace=False)
        no_pii_examples = [no_pii_pool[i] for i in no_pii_indices]
    else:
        no_pii_examples = []
    
    if n_with_pii > 0:
        with_pii_indices = rng.choice(len(with_pii_pool), size=n_with_pii, replace=False)
        with_pii_examples = [with_pii_pool[i] for i in with_pii_indices]
    else:
        with_pii_examples = []

    return no_pii_examples, with_pii_examples

