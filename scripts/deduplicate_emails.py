"""
Email Deduplication Script

Processes the email dataset to remove duplicates using:
1. Exact duplicate detection (hash-based)
2. Forwarded email duplicate detection (hash after cleaning headers)

Outputs:
- Deduplicated email dataset (CSV)
- Duplicate mapping (JSON)
- Statistics report (TXT)
- Cost/time savings analysis (JSON)

Usage:
    python scripts/deduplicate_emails.py

Author: Asymmetric Security
Date: 2025-11-02
"""

import csv
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_unlabeled_dataset, Email
from src.utils.deduplicator import EmailDeduplicator, DeduplicationStats
from src import config


def save_deduplicated_csv(emails: list[Email], output_path: Path) -> None:
    """Save deduplicated emails to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(['ID', 'Subject', 'From', 'To', 'CC', 'BCC', 'Message Body'])

        # Write emails
        for email in emails:
            writer.writerow([
                email.id,
                email.subject,
                email.from_addr,
                email.to_addr,
                email.cc,
                email.bcc,
                email.message_body
            ])

    print(f"Saved {len(emails):,} unique emails to: {output_path}")


def save_duplicate_mapping(deduplicator: EmailDeduplicator, output_path: Path) -> None:
    """Save mapping of unique emails to their duplicates."""
    duplicate_mapping = {}

    # Get exact duplicate groups
    for hash_val, email_group in deduplicator.hash_to_emails.items():
        if len(email_group) > 1:
            representative_id = email_group[0].id
            duplicate_ids = [e.id for e in email_group]
            duplicate_mapping[f"exact_{representative_id}"] = {
                "representative_id": representative_id,
                "duplicate_ids": duplicate_ids,
                "count": len(duplicate_ids),
                "type": "exact"
            }

    # Get forwarded duplicate groups
    for hash_val, email_group in deduplicator.cleaned_hash_to_emails.items():
        if len(email_group) > 1:
            representative_id = email_group[0].id
            duplicate_ids = [e.id for e in email_group]
            duplicate_mapping[f"forwarded_{representative_id}"] = {
                "representative_id": representative_id,
                "duplicate_ids": duplicate_ids,
                "count": len(duplicate_ids),
                "type": "forwarded"
            }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(duplicate_mapping, f, indent=2)

    print(f"Saved duplicate mapping to: {output_path}")


def calculate_savings(stats: DeduplicationStats) -> dict:
    """Calculate cost and time savings from deduplication."""

    # Benchmark data from existing results (per email averages)
    # Based on the README and results analysis
    benchmarks = {
        "presidio_strict": {
            "time_ms": 24.0,  # milliseconds per email
            "cost_usd": 0.0   # free (runs locally)
        },
        "haiku_zero_shot": {
            "time_ms": 420.0,  # ~7 minutes for 1000 emails = 420ms per email
            "cost_usd": 0.00056  # $28 for 50k emails = $0.00056 per email
        },
        "sonnet_zero_shot": {
            "time_ms": 1038.0,  # ~17.3 minutes for 1000 emails = 1038ms per email
            "cost_usd": 0.00214  # $107 for 50k emails = $0.00214 per email
        }
    }

    # Calculate for 50k email production run
    scale_factor = 50000 / 9199  # Scale from current dataset to production
    original_emails = stats.total_emails * scale_factor
    unique_emails = stats.unique_emails * scale_factor
    duplicates_removed = stats.total_duplicates_removed * scale_factor

    savings = {
        "dataset_reduction": {
            "original_emails": int(original_emails),
            "unique_emails": int(unique_emails),
            "duplicates_removed": int(duplicates_removed),
            "reduction_percentage": stats.deduplication_rate
        },
        "presidio_strict": {},
        "haiku_zero_shot": {},
        "sonnet_zero_shot": {}
    }

    for detector, bench in benchmarks.items():
        # Original cost/time
        original_time_minutes = (original_emails * bench["time_ms"]) / 1000 / 60
        original_cost = original_emails * bench["cost_usd"]

        # After deduplication
        dedup_time_minutes = (unique_emails * bench["time_ms"]) / 1000 / 60
        dedup_cost = unique_emails * bench["cost_usd"]

        # Savings
        time_saved_minutes = original_time_minutes - dedup_time_minutes
        cost_saved = original_cost - dedup_cost

        savings[detector] = {
            "original": {
                "time_minutes": round(original_time_minutes, 2),
                "cost_usd": round(original_cost, 2)
            },
            "after_deduplication": {
                "time_minutes": round(dedup_time_minutes, 2),
                "cost_usd": round(dedup_cost, 2)
            },
            "savings": {
                "time_minutes": round(time_saved_minutes, 2),
                "time_percentage": round((time_saved_minutes / original_time_minutes) * 100, 2),
                "cost_usd": round(cost_saved, 2),
                "cost_percentage": round((cost_saved / original_cost) * 100, 2) if original_cost > 0 else 0
            }
        }

    return savings


def save_statistics_report(stats: DeduplicationStats, savings: dict, output_path: Path) -> None:
    """Save detailed statistics and savings report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EMAIL DEDUPLICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Basic statistics
        f.write(str(stats))
        f.write("\n")

        # Dataset reduction
        f.write("=" * 80 + "\n")
        f.write("PRODUCTION IMPACT (50,000 Email Dataset)\n")
        f.write("=" * 80 + "\n\n")

        reduction = savings["dataset_reduction"]
        f.write(f"Original Emails:             {reduction['original_emails']:,}\n")
        f.write(f"Unique Emails:               {reduction['unique_emails']:,}\n")
        f.write(f"Duplicates Removed:          {reduction['duplicates_removed']:,}\n")
        f.write(f"Reduction:                   {reduction['reduction_percentage']:.2f}%\n\n")

        # Cost and time savings per detector
        for detector in ["presidio_strict", "haiku_zero_shot", "sonnet_zero_shot"]:
            detector_name = detector.replace("_", " ").title()
            f.write("-" * 80 + "\n")
            f.write(f"{detector_name}\n")
            f.write("-" * 80 + "\n")

            data = savings[detector]
            orig = data["original"]
            dedup = data["after_deduplication"]
            save = data["savings"]

            f.write(f"\nOriginal Processing:\n")
            f.write(f"  Time:   {orig['time_minutes']:.2f} minutes\n")
            f.write(f"  Cost:   ${orig['cost_usd']:.2f}\n")

            f.write(f"\nAfter Deduplication:\n")
            f.write(f"  Time:   {dedup['time_minutes']:.2f} minutes\n")
            f.write(f"  Cost:   ${dedup['cost_usd']:.2f}\n")

            f.write(f"\nSavings:\n")
            f.write(f"  Time:   {save['time_minutes']:.2f} minutes ({save['time_percentage']:.2f}%)\n")
            if save['cost_usd'] > 0:
                f.write(f"  Cost:   ${save['cost_usd']:.2f} ({save['cost_percentage']:.2f}%)\n")
            else:
                f.write(f"  Cost:   $0.00 (already free)\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. Apply deduplication before PII detection to reduce processing time and cost\n")
        f.write("2. Use the deduplicated dataset for all detector evaluations\n")
        f.write("3. Forwarded email detection provides additional 4-5% improvement\n")
        f.write("4. Consider implementing deduplication in production pipeline\n")
        f.write("5. Monitor duplicate rates in new datasets for data quality issues\n\n")

    print(f"Saved statistics report to: {output_path}")


def main():
    """Main deduplication workflow."""
    print("=" * 80)
    print("Email Deduplication Process")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = config.PROJECT_ROOT / "task" / "dedup"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Load dataset
    print("Loading email dataset...")
    emails = load_unlabeled_dataset()
    print(f"Loaded {len(emails):,} emails\n")

    # Perform deduplication
    print("Performing deduplication...")
    print("  Step 1: Detecting exact duplicates (hash-based)...")
    deduplicator = EmailDeduplicator()
    unique_emails, stats = deduplicator.deduplicate(emails)
    print(f"  Step 2: Detecting forwarded email duplicates...")
    print("Done!\n")

    # Print statistics
    print(stats)

    # Save outputs
    print("Saving results...")

    # 1. Deduplicated CSV
    deduplicated_csv = output_dir / "email_content_dataset_deduped.csv"
    save_deduplicated_csv(unique_emails, deduplicated_csv)

    # 2. Duplicate mapping
    duplicate_mapping_json = output_dir / "duplicate_mapping.json"
    save_duplicate_mapping(deduplicator, duplicate_mapping_json)

    # 3. Calculate savings
    print("\nCalculating cost and time savings...")
    savings = calculate_savings(stats)

    # 4. Save savings JSON
    savings_json = output_dir / "savings_analysis.json"
    with open(savings_json, 'w', encoding='utf-8') as f:
        json.dump(savings, f, indent=2)
    print(f"Saved savings analysis to: {savings_json}")

    # 5. Statistics report
    report_txt = output_dir / "deduplication_report.txt"
    save_statistics_report(stats, savings, report_txt)

    # Summary
    print("\n" + "=" * 80)
    print("DEDUPLICATION COMPLETE")
    print("=" * 80)
    print(f"\nFiles created in: {output_dir}/")
    print(f"  - email_content_dataset_deduped.csv  ({len(unique_emails):,} unique emails)")
    print(f"  - duplicate_mapping.json             (duplicate group mappings)")
    print(f"  - savings_analysis.json              (cost/time projections)")
    print(f"  - deduplication_report.txt           (detailed report)")
    print()
    print(f"Dataset reduced by {stats.deduplication_rate:.2f}%")
    print(f"  {stats.total_emails:,} emails → {stats.unique_emails:,} unique emails")
    print()


if __name__ == "__main__":
    main()
