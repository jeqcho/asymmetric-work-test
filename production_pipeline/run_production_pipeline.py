#!/usr/bin/env python3
"""
Production PII Detection Pipeline Entry Point

Usage:
    python run_production_pipeline.py <input_csv> <output_csv> [--parallel N]
    
Example:
    python run_production_pipeline.py ../task/Datasets/email_content_dataset.csv output.csv --parallel 10
"""

import sys
import argparse
from pathlib import Path

from pipeline.run_pipeline import run_pipeline
from pipeline.config import PARALLEL_REQUESTS


def main():
    parser = argparse.ArgumentParser(
        description="Production PII Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.csv output.csv
  %(prog)s input.csv output.csv --parallel 20
  %(prog)s input.csv output.csv --labeled-dataset ../task/Datasets/250_labeled_dataset.csv
        """
    )
    
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to input CSV file (must have columns: ID, Subject, From, To, CC, BCC, Message Body)"
    )
    
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Path to output CSV file"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=PARALLEL_REQUESTS,
        help=f"Number of parallel API requests (default: {PARALLEL_REQUESTS})"
    )
    
    parser.add_argument(
        "--labeled-dataset",
        type=Path,
        default=None,
        help="Path to labeled dataset CSV for 5-shot examples (optional, will use default if not specified)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input_csv.exists():
        print(f"Error: Input file does not exist: {args.input_csv}")
        sys.exit(1)
    
    # Validate output directory exists
    output_dir = args.output_csv.parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate labeled dataset if provided
    if args.labeled_dataset and not args.labeled_dataset.exists():
        print(f"Warning: Labeled dataset file does not exist: {args.labeled_dataset}")
        print("  Pipeline will attempt to use default labeled dataset path.")
        args.labeled_dataset = None
    
    # Run pipeline
    try:
        run_pipeline(
            input_csv_path=args.input_csv,
            output_csv_path=args.output_csv,
            parallel_requests=args.parallel,
            labeled_dataset_path=args.labeled_dataset
        )
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

