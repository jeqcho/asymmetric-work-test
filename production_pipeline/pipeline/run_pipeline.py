"""Main production pipeline for PII detection."""

import csv
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from pipeline.utils.data_loader import Email, load_csv_dataset, load_labeled_dataset
from pipeline.utils.deduplicator import EmailDeduplicator
from pipeline.detectors.presidio_detector import PresidioDetector
from pipeline.detectors.claude_classifier import ClaudePIIClassificationDetector
from pipeline.config import PII_TYPES, PARALLEL_REQUESTS


class PipelineResult:
    """Container for pipeline results per email."""
    def __init__(self, email_id: int):
        self.email_id = email_id
        self.is_duplicate = False
        self.representative_id = email_id
        self.pii_absent_high_confidence = True  # Presidio found nothing
        self.pii_absent_low_confidence = True  # Haiku found nothing
        self.pii_types = []  # List of detected PII types from Haiku


def run_deduplication(emails: List[Email]) -> Tuple[List[Email], Dict[int, Tuple[bool, int]], float]:
    """
    Step 1: Deduplicate emails.
    
    Args:
        emails: Original email list
        
    Returns:
        Tuple of (unique_emails, id_mapping, elapsed_time) where id_mapping maps original_id -> (is_duplicate, representative_id)
    """
    print("Step 1: Deduplicating emails...")
    print(f"  📊 Processing {len(emails):,} emails")
    
    start_time = time.time()
    deduplicator = EmailDeduplicator()
    unique_emails, stats = deduplicator.deduplicate(emails)
    
    # Get ID mapping
    id_mapping = deduplicator.get_duplicate_id_mapping(emails)
    elapsed_time = time.time() - start_time
    
    print(f"  ✓ Deduplicated: {len(emails):,} → {len(unique_emails):,} unique emails")
    print(f"  ✓ Removed {stats.total_duplicates_removed:,} duplicates ({stats.deduplication_rate:.2f}%)")
    print(f"  ⏱  Time: {elapsed_time:.2f}s")
    print(f"  📊 Next stage will process {len(unique_emails):,} unique emails")
    
    return unique_emails, id_mapping, elapsed_time


def run_presidio_detection(emails: List[Email], parallel_requests: int = PARALLEL_REQUESTS) -> Tuple[Dict[int, bool], float, Dict[str, Any]]:
    """
    Step 2: Run Presidio detection (threshold 0.0).
    
    Args:
        emails: List of deduplicated emails
        parallel_requests: Number of parallel requests
        
    Returns:
        Tuple of (results_dict, elapsed_time, summary_stats) where:
        - results_dict: email_id -> has_pii (True if Presidio found PII)
        - elapsed_time: time taken in seconds
        - summary_stats: dictionary with statistics
    """
    print(f"\nStep 2: Running Presidio detection (threshold 0.0)...")
    print(f"  📊 Processing {len(emails):,} unique emails (after deduplication)")
    
    start_time = time.time()
    detector = PresidioDetector(confidence_threshold=0.0)
    
    results: Dict[int, bool] = {}
    results_with_entities: Dict[int, List] = {}
    entity_type_counts = Counter()
    
    def process_email(email: Email) -> Tuple[int, bool, List]:
        """Process single email with Presidio."""
        has_pii, entities = detector.detect(email)
        email.presidio_entities = entities  # Store on email object
        return email.id, has_pii, entities
    
    # Process emails in parallel (Presidio runs locally, so we can parallelize)
    print(f"  ⏳ Processing with {parallel_requests} parallel workers...")
    with tqdm(total=len(emails), desc="  Presidio", unit="email", ncols=100) as pbar:
        with ThreadPoolExecutor(max_workers=parallel_requests) as executor:
            future_to_email = {
                executor.submit(process_email, email): email
                for email in emails
            }
            
            for future in as_completed(future_to_email):
                try:
                    email_id, has_pii, entities = future.result()
                    results[email_id] = has_pii
                    results_with_entities[email_id] = entities
                    # Count entity types
                    for entity in entities:
                        entity_type_counts[entity.get("entity_type", "UNKNOWN")] += 1
                except Exception as e:
                    email = future_to_email[future]
                    print(f"\n    Error processing email {email.id}: {e}")
                    results[email.id] = False
                    results_with_entities[email.id] = []
                finally:
                    pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    # Store entities back on email objects (in case they weren't stored)
    for email in emails:
        if email.id in results_with_entities:
            email.presidio_entities = results_with_entities[email.id]
    
    # Calculate statistics
    pii_count = sum(1 for has_pii in results.values() if has_pii)
    no_pii_count = len(emails) - pii_count
    
    summary_stats = {
        "total_processed": len(emails),
        "with_pii": pii_count,
        "without_pii": no_pii_count,
        "pii_percentage": (pii_count / len(emails) * 100) if emails else 0.0,
        "entity_type_counts": dict(entity_type_counts),
        "total_entities": sum(entity_type_counts.values()),
        "time_seconds": elapsed_time,
        "time_per_email_ms": (elapsed_time * 1000 / len(emails)) if emails else 0.0
    }
    
    print(f"  ✓ Completed: {pii_count:,} emails with PII detected ({pii_count/len(emails)*100:.1f}%)")
    print(f"  ✓ No PII: {no_pii_count:,} emails ({no_pii_count/len(emails)*100:.1f}%)")
    print(f"  ✓ Total entities detected: {sum(entity_type_counts.values()):,}")
    if entity_type_counts:
        print(f"  ✓ Top entity types: {', '.join([f'{k}({v:,})' for k, v in entity_type_counts.most_common(5)])}")
    print(f"  ⏱  Time: {elapsed_time:.2f}s ({elapsed_time/len(emails)*1000:.2f}ms per email)")
    print(f"  📊 Next stage will process {len(emails):,} emails (all emails, partitioned by Presidio results)")
    
    return results, elapsed_time, summary_stats


def run_haiku_classification(
    emails: List[Email],
    labeled_dataset: List[Email],
    parallel_requests: int = PARALLEL_REQUESTS
) -> Tuple[Dict[int, List[str]], float, Dict[str, Any]]:
    """
    Step 3: Run Haiku 5-shot classification with Presidio augmentation.
    
    Args:
        emails: List of deduplicated emails (with presidio_entities populated)
        labeled_dataset: Labeled dataset for 5-shot examples (should have presidio_entities)
        parallel_requests: Number of parallel API requests
        
    Returns:
        Tuple of (results_dict, elapsed_time, summary_stats) where:
        - results_dict: email_id -> list of detected PII types
        - elapsed_time: time taken in seconds
        - summary_stats: dictionary with statistics including costs
    """
    print(f"\nStep 3: Running Haiku 5-shot classification with Presidio...")
    print(f"  📊 Processing {len(emails):,} emails (all unique emails)")
    
    if not labeled_dataset:
        print("  ⚠ Warning: No labeled dataset available, cannot use 5-shot. Results may be limited.")
        return {email.id: [] for email in emails}, 0.0, {}
    
    detector = ClaudePIIClassificationDetector(labeled_dataset=labeled_dataset, use_few_shot=True)
    
    results: Dict[int, List[str]] = {}
    pii_type_counts = Counter()
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    
    def process_email(email: Email) -> Tuple[int, List[str], float, int, int]:
        """Process single email with Haiku."""
        pii_types, time_ms, cost, input_tokens, output_tokens = detector.classify(email)
        return email.id, pii_types, cost, input_tokens, output_tokens
    
    # Process emails in parallel
    print(f"  ⏳ Processing with {parallel_requests} parallel API requests...")
    start_time = time.time()
    
    with tqdm(total=len(emails), desc="  Haiku", unit="email", ncols=100) as pbar:
        with ThreadPoolExecutor(max_workers=parallel_requests) as executor:
            future_to_email = {
                executor.submit(process_email, email): email
                for email in emails
            }
            
            for future in as_completed(future_to_email):
                try:
                    email_id, pii_types, cost, input_tokens, output_tokens = future.result()
                    results[email_id] = pii_types
                    # Track costs and tokens
                    total_cost += cost
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    # Count PII types
                    for pii_type in pii_types:
                        pii_type_counts[pii_type] += 1
                except Exception as e:
                    email = future_to_email[future]
                    print(f"\n    Error processing email {email.id}: {e}")
                    results[email.id] = []
                finally:
                    pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    # Calculate statistics
    pii_count = sum(1 for pii_types in results.values() if pii_types)
    no_pii_count = len(emails) - pii_count
    
    summary_stats = {
        "total_processed": len(emails),
        "with_pii": pii_count,
        "without_pii": no_pii_count,
        "pii_percentage": (pii_count / len(emails) * 100) if emails else 0.0,
        "pii_type_counts": dict(pii_type_counts),
        "total_pii_types_detected": sum(pii_type_counts.values()),
        "unique_pii_types": len(pii_type_counts),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd": total_cost,
        "avg_cost_per_email_usd": total_cost / len(emails) if emails else 0.0,
        "time_seconds": elapsed_time,
        "time_per_email_ms": (elapsed_time * 1000 / len(emails)) if emails else 0.0
    }
    
    print(f"  ✓ Completed: {pii_count:,} emails with PII detected ({pii_count/len(emails)*100:.1f}%)")
    print(f"  ✓ No PII: {no_pii_count:,} emails ({no_pii_count/len(emails)*100:.1f}%)")
    print(f"  ✓ Total PII type detections: {sum(pii_type_counts.values()):,}")
    print(f"  ✓ Unique PII types found: {len(pii_type_counts)}")
    if pii_type_counts:
        print(f"  ✓ Top PII types: {', '.join([f'{k}({v:,})' for k, v in pii_type_counts.most_common(10)])}")
    print(f"  ⏱  Time: {elapsed_time:.2f}s ({elapsed_time/len(emails)*1000:.2f}ms per email)")
    print(f"  💰 Cost: ${total_cost:.4f} (${total_cost/len(emails):.6f} per email)")
    print(f"  📊 Tokens: {total_input_tokens:,} input, {total_output_tokens:,} output ({total_input_tokens + total_output_tokens:,} total)")
    
    return results, elapsed_time, summary_stats


def generate_output_csv(
    original_emails: List[Email],
    unique_emails: List[Email],
    id_mapping: Dict[int, Tuple[bool, int]],
    presidio_results: Dict[int, bool],
    haiku_results: Dict[int, List[str]],
    output_path: Path
) -> Tuple[float, Dict[str, int]]:
    """
    Step 4: Generate output CSV with all required columns.
    
    Args:
        original_emails: All original emails (including duplicates)
        unique_emails: Deduplicated unique emails
        id_mapping: Mapping from original_id -> (is_duplicate, representative_id)
        presidio_results: Presidio detection results (email_id -> has_pii)
        haiku_results: Haiku classification results (email_id -> list of PII types)
        output_path: Path to output CSV file
        
    Returns:
        Tuple of (elapsed_time, stats) where stats includes row counts
    """
    print(f"\nStep 4: Generating output CSV...")
    print(f"  📊 Writing {len(original_emails):,} rows (all original emails, including duplicates)")
    
    start_time = time.time()
    
    # Create mapping from representative_id -> results for quick lookup
    representative_to_unique = {email.id: email for email in unique_emails}
    
    # Create result objects for all original emails
    all_results: Dict[int, PipelineResult] = {}
    
    # Initialize results for all original emails
    for email in original_emails:
        result = PipelineResult(email.id)
        all_results[email.id] = result
    
    # Process unique emails and propagate to duplicates
    for unique_email in unique_emails:
        email_id = unique_email.id
        
        # Get results for this unique email
        presidio_has_pii = presidio_results.get(email_id, False)
        haiku_pii_types = haiku_results.get(email_id, [])
        
        # Update representative result
        result = all_results[email_id]
        result.is_duplicate = False
        result.representative_id = email_id
        result.pii_absent_high_confidence = not presidio_has_pii
        result.pii_absent_low_confidence = len(haiku_pii_types) == 0
        result.pii_types = haiku_pii_types
        
        # Propagate to all duplicates
        for orig_id, (is_dup, rep_id) in id_mapping.items():
            if rep_id == email_id and is_dup:
                dup_result = all_results[orig_id]
                dup_result.is_duplicate = True
                dup_result.representative_id = rep_id
                dup_result.pii_absent_high_confidence = result.pii_absent_high_confidence
                dup_result.pii_absent_low_confidence = result.pii_absent_low_confidence
                dup_result.pii_types = result.pii_types.copy()
    
    # Write CSV
    print(f"  ⏳ Writing CSV to {output_path}...")
    
    # Build column names
    columns = ['ID', 'duplicated', 'pii_absent_high_confidence', 'pii_absent_low_confidence']
    columns.extend(PII_TYPES)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        
        # Write rows for all original emails in order
        for email in original_emails:
            result = all_results[email.id]
            row = [
                result.email_id,
                result.is_duplicate,
                result.pii_absent_high_confidence,
                result.pii_absent_low_confidence
            ]
            
            # Add boolean columns for each PII type
            for pii_type in PII_TYPES:
                row.append(pii_type in result.pii_types)
            
            writer.writerow(row)
    
    elapsed_time = time.time() - start_time
    
    stats = {
        "total_rows": len(original_emails),
        "total_columns": len(columns),
        "pii_type_columns": len(PII_TYPES)
    }
    
    print(f"  ✓ Output CSV written: {len(original_emails):,} rows")
    print(f"  ✓ Columns: {len(columns)} total ({len(PII_TYPES)} PII type columns)")
    print(f"  ⏱  Time: {elapsed_time:.2f}s")
    
    return elapsed_time, stats


def run_pipeline(
    input_csv_path: Path,
    output_csv_path: Path,
    parallel_requests: int = PARALLEL_REQUESTS,
    labeled_dataset_path: Path = None
) -> None:
    """
    Run the complete production pipeline.
    
    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path to output CSV file
        parallel_requests: Number of parallel API requests
        labeled_dataset_path: Path to labeled dataset for 5-shot (optional, will use default if None)
    """
    print("=" * 80)
    print("PRODUCTION PII DETECTION PIPELINE")
    print("=" * 80)
    print()
    
    # Load labeled dataset for 5-shot examples
    print("Loading labeled dataset for 5-shot examples...")
    if labeled_dataset_path and labeled_dataset_path.exists():
        labeled_emails = load_csv_dataset(labeled_dataset_path)
        print(f"  ✓ Loaded {len(labeled_emails):,} labeled emails from {labeled_dataset_path}")
    else:
        from pipeline.config import LABELED_DATASET_PATH
        labeled_emails = load_labeled_dataset()
        if labeled_emails:
            print(f"  ✓ Loaded {len(labeled_emails):,} labeled emails from default path")
        else:
            print("  ⚠ Warning: No labeled dataset found. Will proceed without 5-shot examples.")
    
    # Pre-compute Presidio entities for labeled examples if needed
    if labeled_emails:
        print("Pre-computing Presidio entities for labeled examples...")
        presidio_detector = PresidioDetector(confidence_threshold=0.0)
        for email in labeled_emails:
            _, entities = presidio_detector.detect(email)
            email.presidio_entities = entities
        print(f"  ✓ Pre-computed Presidio entities for {len(labeled_emails):,} labeled emails")
    
    # Load input CSV
    print(f"\nLoading input CSV: {input_csv_path}")
    load_start = time.time()
    original_emails = load_csv_dataset(input_csv_path)
    load_time = time.time() - load_start
    print(f"  ✓ Loaded {len(original_emails):,} emails (took {load_time:.2f}s)")
    
    # Track overall timing and costs
    total_start_time = time.time()
    timing_breakdown = {}
    cost_breakdown = {}
    
    # Step 1: Deduplication
    unique_emails, id_mapping, dedup_time = run_deduplication(original_emails)
    timing_breakdown["deduplication"] = dedup_time
    
    # Step 2: Presidio detection
    presidio_results, presidio_time, presidio_stats = run_presidio_detection(unique_emails, parallel_requests)
    timing_breakdown["presidio"] = presidio_time
    cost_breakdown["presidio"] = 0.0  # Presidio runs locally, no cost
    
    # Step 3: Haiku classification
    haiku_results, haiku_time, haiku_stats = run_haiku_classification(unique_emails, labeled_emails, parallel_requests)
    timing_breakdown["haiku"] = haiku_time
    if haiku_stats:
        cost_breakdown["haiku"] = haiku_stats.get("total_cost_usd", 0.0)
    
    # Step 4: Generate output CSV
    csv_time, csv_stats = generate_output_csv(
        original_emails,
        unique_emails,
        id_mapping,
        presidio_results,
        haiku_results,
        output_csv_path
    )
    timing_breakdown["csv_generation"] = csv_time
    
    total_time = time.time() - total_start_time
    total_cost = sum(cost_breakdown.values())
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE - SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\n📊 EMAIL COUNTS:")
    print(f"  Total emails (input):          {len(original_emails):,}")
    print(f"  Unique emails (after dedup):   {len(unique_emails):,}")
    print(f"  Duplicates removed:           {len(original_emails) - len(unique_emails):,}")
    if presidio_stats:
        print(f"  Presidio: PII detected:      {presidio_stats['with_pii']:,} ({presidio_stats['pii_percentage']:.1f}%)")
        print(f"  Presidio: No PII:            {presidio_stats['without_pii']:,} ({100-presidio_stats['pii_percentage']:.1f}%)")
    if haiku_stats:
        print(f"  Haiku: PII detected:          {haiku_stats['with_pii']:,} ({haiku_stats['pii_percentage']:.1f}%)")
        print(f"  Haiku: No PII:                {haiku_stats['without_pii']:,} ({100-haiku_stats['pii_percentage']:.1f}%)")
    
    if haiku_stats and haiku_stats.get('pii_type_counts'):
        print(f"\n📋 PII TYPE BREAKDOWN (Haiku):")
        pii_counts = haiku_stats['pii_type_counts']
        sorted_types = sorted(pii_counts.items(), key=lambda x: x[1], reverse=True)
        for pii_type, count in sorted_types:
            percentage = (count / haiku_stats['total_processed'] * 100) if haiku_stats['total_processed'] > 0 else 0
            print(f"  {pii_type:30s} {count:6,} ({percentage:5.1f}% of emails)")
        print(f"  {'Total PII type detections':30s} {haiku_stats['total_pii_types_detected']:6,}")
        print(f"  {'Unique PII types found':30s} {haiku_stats['unique_pii_types']:6}")
    
    print(f"\n⏱  TIME BREAKDOWN:")
    print(f"  Data loading:                 {load_time:8.2f}s ({load_time/total_time*100:5.1f}%)")
    for stage, time_taken in timing_breakdown.items():
        print(f"  {stage.capitalize():25s} {time_taken:8.2f}s ({time_taken/total_time*100:5.1f}%)")
    print(f"  {'Total pipeline time':25s} {total_time:8.2f}s (100.0%)")
    
    if haiku_stats:
        print(f"\n💰 COST BREAKDOWN:")
        print(f"  Presidio (local):            ${cost_breakdown.get('presidio', 0.0):10.4f}")
        print(f"  Haiku API:                   ${cost_breakdown.get('haiku', 0.0):10.4f}")
        if haiku_stats.get('total_input_tokens'):
            print(f"    - Input tokens:            {haiku_stats['total_input_tokens']:10,}")
            print(f"    - Output tokens:           {haiku_stats['total_output_tokens']:10,}")
            print(f"    - Total tokens:            {haiku_stats['total_input_tokens'] + haiku_stats['total_output_tokens']:10,}")
            print(f"    - Avg cost per email:      ${haiku_stats.get('avg_cost_per_email_usd', 0.0):10.6f}")
        print(f"  {'Total cost':25s} ${total_cost:10.4f}")
    
    print(f"\n📁 OUTPUT:")
    print(f"  File: {output_csv_path}")
    print(f"  Rows: {len(original_emails):,}")
    print(f"  Columns: {csv_stats.get('total_columns', 0):,}")
    print()
    print("=" * 80)

