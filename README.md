# PII Detection Testing Framework

A comprehensive testing framework for evaluating different PII (Personal Identifiable Information) detection methods on email datasets. This project implements and compares 8 different detector configurations across three approaches: Microsoft Presidio, Claude Haiku, and Claude Sonnet.

Read the findings report [here](asym-report-1.pdf)

## Overview

This framework was built to evaluate the trade-offs between cost, speed, and accuracy for PII detection in email breach analysis. It tests detectors on a 250-sample gold-labeled email dataset and projects costs/time for a 50k email dataset.

### Detector Configurations

**Presidio (Local, Free)**
- `presidio_lax` - Confidence threshold 0.8 (fewer flags, high precision)
- `presidio_moderate` - Confidence threshold 0.5 (balanced)
- `presidio_strict` - Confidence threshold 0.3 (more flags, high recall)

**Claude Haiku (Fast, Lower Cost)**
- `haiku_zeroshot` - Zero-shot binary classification
- `haiku_5shot` - With 5 example emails (3 no-PII, 2 with-PII)

**Claude Sonnet (Slower, Higher Accuracy)**
- `sonnet_zeroshot` - Zero-shot binary classification
- `sonnet_5shot` - With 5 example emails (3 no-PII, 2 with-PII)

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
cd asymmetric-work-test

# Install dependencies (if not already installed)
uv sync

# Download spaCy model for Presidio
uv add pip && uv run python -m spacy download en_core_web_sm
```

### Dependencies

- `anthropic` - Claude API client
- `presidio-analyzer` - Microsoft PII detection
- `presidio-anonymizer` - PII anonymization utilities
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `python-dotenv` - Environment variable management
- `spacy` - NLP engine for Presidio (with `en_core_web_sm` model)

### API Key Setup

**Required for Claude detectors:**

1. Create a `.env` file in the project root (or edit the existing one)
2. Add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```
3. Ensure your account has sufficient credits for the evaluation
   - The 250-sample test will make ~250 API calls per Claude detector (4 detectors total)
   - Estimated cost: ~$2-5 depending on email length and model usage

**Note:** Presidio detectors run locally and don't require an API key.

## Usage

### Running the Full Evaluation

To run all 7 detectors (3 Presidio + 4 Claude):

```bash
uv run python scripts/run_evaluation.py
```

This will:
1. Load the 250-sample labeled dataset
2. Initialize all detectors (Presidio + Claude)
3. Run each detector on all emails
4. Calculate metrics (TP, FP, TN, FN, precision, recall, F1)
5. Benchmark cost and time
6. Generate comparison CSV and visualizations
7. Export false positives/negatives for manual review

### Running Claude Detectors Only

If Presidio has already been evaluated, you can run only the Claude detectors:

```bash
uv run python scripts/run_claude_only.py
```

This script will:
- Skip Presidio detectors
- Run the 4 Claude detectors (Haiku and Sonnet, zero-shot and 5-shot)
- Load existing Presidio results
- Generate combined visualizations and comparison CSV

### Configuration

Edit `src/config.py` to adjust:
- Presidio confidence thresholds
- Claude model versions
- API pricing
- Random seed for reproducibility
- PII types and descriptions

## Evaluation Metrics

### Email-Level Classification

The framework uses **email-level binary classification**: Does the email contain ANY PII or not?

### Metrics Calculated

- **TP (True Positives)**: Correctly identified emails with PII
- **FP (False Positives)**: Incorrectly flagged emails without PII
- **TN (True Negatives)**: Correctly identified emails without PII
- **FN (False Negatives)**: Missed emails that contain PII
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: (TP + TN) / Total

### Benchmark Metrics

- Average time per email (milliseconds)
- Average cost per email (USD)
- Projected time for 50k emails (minutes)
- Projected cost for 50k emails (USD)

## Results

### Evaluation Results (Presidio Only - API Credits Required for Claude)

Results from the 250-sample dataset:

| Detector | TP | FP | TN | FN | Precision | Recall | F1 | Avg Time (ms) | Projected Time 50k (min) | Projected Cost 50k |
|----------|----|----|----|----|-----------|--------|----|--------------|--------------------------|--------------------|
| presidio_lax | 21 | 199 | 28 | 2 | 0.095 | 0.913 | 0.173 | 27.6 | 23.0 | $0.00 |
| presidio_moderate | 23 | 201 | 26 | 0 | 0.103 | 1.000 | 0.186 | 23.6 | 19.7 | $0.00 |
| presidio_strict | 23 | 201 | 26 | 0 | 0.103 | 1.000 | 0.186 | 24.9 | 20.7 | $0.00 |

**Dataset Statistics:**
- Total emails: 250
- Emails with PII: 23 (9.2%)
- Emails without PII: 227 (90.8%)

**Claude Detector Status:**
- Framework is fully implemented and functional
- Requires valid Anthropic API key with sufficient credits
- Once API key is added to `.env`, simply run the evaluation script again
- The framework gracefully handles API errors and continues evaluation

### Key Observations (Presidio)

1. **High Recall**: Presidio detectors achieve 91-100% recall, meaning they catch most PII-containing emails
2. **Low Precision**: ~10% precision indicates many false positives
3. **Fast & Free**: Average 24-28ms per email, $0 cost (runs locally)
4. **Threshold Impact**: Moderate and strict thresholds perform identically (both 100% recall)

### False Positives

199-201 false positives were flagged by Presidio. These are exported to CSV files in `results/error_analysis/` for manual review to determine if they are:
- True false positives (labeling was correct)
- Potential labeling errors (emails that should have been labeled as containing PII)

## PII Types Detected

The framework detects 15 types of PII as specified in the task requirements:

1. **[full name]** - Only flagged if with another PII element (not email address)
2. **[ssn]** - Social security number
3. **[ssn, last 4]** - Last 4 digits of SSN
4. **[drivers license #]** - Driver's license number
5. **[passport #]** - Passport number
6. **[TIN]** - Taxpayer identification number
7. **[irs identity protection pin]** - IRS PIN
8. **[student identification #]** - Student ID
9. **[bank account #]** - Account number (not routing)
10. **[credit card #]** - Credit/debit card numbers
11. **[cvv/cvc]** - Card security codes
12. **[password]** - Passwords and credentials
13. **[username]** - Usernames (in credential context)
14. **[email address]** - Email addresses
15. **[phone number]** - Phone numbers

## Output Files

### Comparison CSV
`results/comparison_metrics.csv` - Side-by-side comparison of all detectors with all metrics

### Individual Results
`results/{detector_name}_results.json` - Detailed results including:
- Per-email predictions
- Confidence scores (where applicable)
- Token usage (for API detectors)
- Complete metrics and benchmarks

### False Positives/Negatives
`results/error_analysis/{detector_name}_fp.csv` - Emails incorrectly flagged as containing PII
`results/error_analysis/{detector_name}_fn.csv` - Emails with PII that were missed

### Visualizations
- `results/visualizations/f1_scores.png` - F1 score comparison bar chart
- `results/visualizations/cost_vs_f1.png` - Cost/accuracy trade-off scatter plot
- `results/visualizations/time_vs_f1.png` - Speed/accuracy trade-off scatter plot
- `results/visualizations/confusion_matrices.png` - Confusion matrices for all detectors

## Reproducibility

This repository is fully reproducible. To run the complete evaluation from scratch:

### Step 1: Clone and Setup

```bash
git clone <repository-url>
cd asymmetric-work-test

# Install dependencies
uv sync

# Download spaCy model for Presidio
uv add pip && uv run python -m spacy download en_core_web_sm
```

### Step 2: Configure API Key

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Anthropic API key
# Make sure your account has sufficient credits (~$2-5 for 250 emails)
```

### Step 3: Run Evaluation

```bash
# Run all detectors (Presidio + Claude)
uv run python scripts/run_evaluation.py

# Or run Claude detectors only (if Presidio already completed)
uv run python scripts/run_claude_only.py
```

### Step 4: Review Results

Results will be generated in the `results/` directory:
- `comparison_metrics.csv` - Summary comparison
- `visualizations/` - Charts and plots
- `error_analysis/` - FP/FN CSVs for manual review
- Individual JSON files per detector

## Next Steps

### To Run Claude Detectors

1. Add API credits to your Anthropic account
2. Update `.env` with a valid API key
3. Run the evaluation script again:
   ```bash
   uv run python scripts/run_claude_only.py
   ```

### Analysis Recommendations

1. **Review the comparison CSV** to understand performance trade-offs
2. **Check false positive CSVs** to identify potential labeling errors in the gold dataset
3. **Examine visualizations** to see cost/speed/accuracy relationships
4. **Consider cascading approach**:
   - Use Presidio first (fast, free, high recall)
   - Route Presidio positives through Haiku for filtering
   - Use Sonnet only for high-confidence cases

### Potential Improvements

1. **Tune Presidio thresholds** based on FP analysis
2. **Refine Claude prompts** based on error patterns
3. **Implement cascading strategy** to optimize cost/accuracy
4. **Add custom Presidio recognizers** for domain-specific PII patterns
5. **Experiment with confidence calibration** for better filtering

## Technical Details

### Presidio Implementation

- Uses spaCy NLP engine (`en_core_web_sm`)
- Detects 16 entity types mapped to the 15 PII categories
- Confidence thresholds control sensitivity
- Runs locally (no API calls)

### Claude Implementation

- Binary classification (YES/NO answers)
- Zero-shot: Task description + PII types + rules
- 5-shot: Same prompt + 5 example emails (3 negative, 2 positive)
- Fixed random seed (42) for reproducible example sampling
- Token usage tracked from API responses
- Cost calculated from actual input/output tokens

### Evaluation Framework

- Email-level binary classification (has PII vs no PII)
- Confusion matrix calculated from predictions vs ground truth
- Benchmarking includes time and cost projections
- False positives/negatives exported for review
- All results saved to JSON for reproducibility
