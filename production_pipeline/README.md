# Production PII Detection Pipeline

A standalone production pipeline for detecting and classifying Personal Identifiable Information (PII) in email datasets.

## Overview

This pipeline processes email CSV files through three main steps:

1. **Deduplication**: Removes exact duplicates and forwarded email duplicates
2. **Presidio Detection**: High-confidence PII detection using Microsoft Presidio (threshold 0.0)
3. **Haiku Classification**: Low-confidence PII type classification using Claude Haiku 5-shot with Presidio augmentation

## Features

- **Deduplication**: Identifies exact and forwarded email duplicates, reducing processing time and costs
- **Two-stage Detection**:
  - Presidio (threshold 0.0): High-confidence detection with 0% false negative rate
  - Claude Haiku 5-shot: Detailed PII type classification with Presidio context
- **Parallel Processing**: Configurable parallel API requests for faster processing
- **Complete Output**: CSV with duplicate flags, confidence levels, and per-PII-type boolean columns

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Create .env file or set in environment
export ANTHROPIC_API_KEY="your-api-key-here"
```

3. Ensure Presidio's spaCy model is installed:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

```bash
python run_production_pipeline.py input.csv output.csv
```

### Advanced Usage

```bash
# Specify number of parallel requests
python run_production_pipeline.py input.csv output.csv --parallel 20

# Specify custom labeled dataset path
python run_production_pipeline.py input.csv output.csv --labeled-dataset /path/to/labeled_dataset.csv
```

### Command Line Arguments

- `input_csv`: Path to input CSV file (required)
  - Must have columns: `ID`, `Subject`, `From`, `To`, `CC`, `BCC`, `Message Body`
- `output_csv`: Path to output CSV file (required)
- `--parallel N`: Number of parallel API requests (default: 10)
- `--labeled-dataset PATH`: Path to labeled dataset for 5-shot examples (optional)

## Input Format

The input CSV must have the following columns:
- `ID`: Unique identifier for each email
- `Subject`: Email subject
- `From`: Sender address
- `To`: Recipient address(es)
- `CC`: CC recipients (optional)
- `BCC`: BCC recipients (optional)
- `Message Body`: Email message body text

Example:
```csv
ID,Subject,From,To,CC,BCC,Message Body
1,"Meeting","alice@example.com","bob@example.com",,,"Please meet at 2pm..."
2,"Re: Meeting","bob@example.com","alice@example.com",,,"Sounds good..."
```

## Output Format

The output CSV contains the following columns:

1. `ID`: Original email ID
2. `duplicated`: Boolean indicating if this email is a duplicate
3. `pii_absent_high_confidence`: Boolean - True if Presidio found no PII
4. `pii_absent_low_confidence`: Boolean - True if Haiku found no PII
5. One boolean column for each PII type (15 total):
   - `full name`
   - `ssn`
   - `ssn last 4`
   - `drivers license #`
   - `passport #`
   - `TIN`
   - `irs identity protection pin`
   - `student identification #`
   - `bank account #`
   - `credit card #`
   - `cvv/cvc`
   - `password`
   - `username`
   - `email address`
   - `phone number`

Example output:
```csv
ID,duplicated,pii_absent_high_confidence,pii_absent_low_confidence,full name,ssn,...
1,False,False,False,True,False,...
2,True,False,False,True,False,...
```

## Pipeline Steps

### Step 1: Deduplication
- Detects exact duplicate emails (identical message bodies)
- Detects forwarded email duplicates (by removing forwarding headers)
- Creates mapping of duplicate IDs to representative IDs
- All duplicates inherit results from their representative

### Step 2: Presidio Detection (Threshold 0.0)
- Runs Presidio on deduplicated emails
- Threshold 0.0 ensures maximum sensitivity (catches everything)
- Presidio has 0% false negative rate, so if it finds nothing, email truly has no PII
- Stores detected entities for use in Step 3

### Step 3: Haiku 5-shot Classification
- Loads labeled dataset for 5-shot examples
- Pre-computes Presidio entities for labeled examples
- Runs Claude Haiku API with:
  - 5-shot examples (3 without PII, 2 with PII)
  - Presidio entity context
  - Parallel API requests for speed
- Classifies specific PII types present in each email

### Step 4: Output Generation
- Maps results back to all original email IDs
- Duplicates inherit results from their representative
- Generates CSV with all required columns

## Configuration

Edit `pipeline/config.py` to modify:
- Presidio threshold (default: 0.0)
- Parallel requests (default: 10)
- Haiku model (default: claude-haiku-4-5)
- API retry settings
- PII type definitions

## Labeled Dataset

The pipeline requires a labeled dataset for 5-shot examples. By default, it looks for:
```
../task/Datasets/250_labeled_dataset.csv
```

This file should have the same structure as the input CSV, plus a `Data Elements` column with comma-separated PII types (e.g., "email address, phone number").

If no labeled dataset is found, the pipeline will run without 5-shot examples (results may be less accurate).

## Best Practices

1. **Deduplication First**: Always run deduplication to reduce costs and processing time
2. **Parallel Requests**: Adjust `--parallel` based on API rate limits (default 10 is safe)
3. **Labeled Dataset**: Ensure labeled dataset has Presidio entities pre-computed for best results
4. **Error Handling**: Pipeline includes retry logic with exponential backoff for API errors

## Troubleshooting

**Error: "No labeled dataset found"**
- Check that `task/Datasets/250_labeled_dataset.csv` exists
- Or use `--labeled-dataset` to specify a custom path

**Error: "ANTHROPIC_API_KEY not found"**
- Set environment variable: `export ANTHROPIC_API_KEY="your-key"`
- Or create `.env` file with: `ANTHROPIC_API_KEY=your-key`

**Slow Processing**
- Increase `--parallel` value (but watch API rate limits)
- Presidio runs locally and is fast
- Haiku API calls are the bottleneck

**High Costs**
- Reduce `--parallel` to avoid rate limiting
- Ensure deduplication is working (should reduce emails by ~40-50%)

## Architecture

```
production_pipeline/
├── pipeline/
│   ├── config.py              # Configuration settings
│   ├── run_pipeline.py        # Main pipeline orchestration
│   ├── detectors/
│   │   ├── presidio_detector.py
│   │   └── claude_classifier.py
│   └── utils/
│       ├── data_loader.py
│       ├── deduplicator.py
│       ├── prompt_builder.py
│       ├── pii_type_normalizer.py
│       └── cost_calculator.py
├── run_production_pipeline.py  # Entry point
├── requirements.txt
└── README.md
```

## License

This pipeline is part of the Asymmetric Security PII detection project.

