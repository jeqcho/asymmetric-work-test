"""Extract full email content for false negatives from all hybrid (Presidio-augmented) variants."""

import sys
import csv
from pathlib import Path
from collections import OrderedDict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_labeled_dataset
from src.config import ERROR_ANALYSIS_DIR


def get_unique_fn_email_ids():
    """Get unique email IDs from all hybrid variant false negative files."""
    fn_files = [
        'haiku_zeroshot_with_presidio_fn.csv',
        'haiku_5shot_with_presidio_fn.csv',
        'sonnet_zeroshot_with_presidio_fn.csv',
        'sonnet_5shot_with_presidio_fn.csv',
    ]
    
    email_ids = set()
    
    for fn_file in fn_files:
        fn_path = ERROR_ANALYSIS_DIR / fn_file
        if fn_path.exists():
            with open(fn_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    email_id = int(row['email_id'])
                    email_ids.add(email_id)
    
    return sorted(email_ids)


def main():
    """Extract and output full email content for false negatives."""
    print("="*80)
    print("Extracting False Negative Emails from Hybrid (Presidio-augmented) Variants")
    print("="*80)
    
    # Get unique email IDs
    fn_email_ids = get_unique_fn_email_ids()
    print(f"\nFound {len(fn_email_ids)} unique false negative email IDs")
    print(f"Email IDs: {fn_email_ids}")
    
    # Load full email dataset
    print("\nLoading labeled dataset...")
    all_emails = load_labeled_dataset()
    
    # Create email lookup
    email_dict = {email.id: email for email in all_emails}
    
    # Extract full email content
    print(f"\nExtracting full email content for {len(fn_email_ids)} emails...")
    print("="*80)
    print()
    
    for i, email_id in enumerate(fn_email_ids, 1):
        if email_id not in email_dict:
            print(f"WARNING: Email ID {email_id} not found in dataset")
            continue
        
        email = email_dict[email_id]
        
        print(f"{'='*80}")
        print(f"EMAIL #{i} (ID: {email_id})")
        print(f"{'='*80}")
        print(f"Subject: {email.subject}")
        print(f"From: {email.from_addr}")
        print(f"To: {email.to_addr}")
        if email.cc:
            print(f"CC: {email.cc}")
        if email.bcc:
            print(f"BCC: {email.bcc}")
        print(f"PII Types (Ground Truth): {email.data_elements}")
        print()
        print("Message Body:")
        print("-" * 80)
        print(email.message_body)
        print("-" * 80)
        print()
        print()
    
    print(f"\n{'='*80}")
    print(f"Total: {len(fn_email_ids)} unique false negative emails extracted")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

