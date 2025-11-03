"""Configuration settings for PII detection evaluation."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (override=True ensures .env takes precedence over shell env vars)
load_dotenv(override=True)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
TASK_DIR = PROJECT_ROOT / "task"
DATASETS_DIR = TASK_DIR / "Datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"
ERROR_ANALYSIS_DIR = RESULTS_DIR / "error_analysis"

# Dataset paths
LABELED_DATASET_PATH = DATASETS_DIR / "250_labeled_dataset.csv"
UNLABELED_DATASET_PATH = DATASETS_DIR / "email_content_dataset.csv"

# API Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model Configuration
HAIKU_MODEL = "claude-haiku-4-5"
SONNET_MODEL = "claude-sonnet-4-5"

# Pricing (USD per million tokens) - as of January 2025
HAIKU_INPUT_PRICE = 0.80 / 1_000_000
HAIKU_OUTPUT_PRICE = 4.00 / 1_000_000
SONNET_INPUT_PRICE = 3.00 / 1_000_000
SONNET_OUTPUT_PRICE = 15.00 / 1_000_000

# Presidio Configuration
PRESIDIO_THRESHOLDS = {
    "lax": 0.8,      # High threshold = fewer flags = more lenient
    "moderate": 0.5,  # Balanced
    "strict": 0.3     # Low threshold = more flags = more strict
}

# Evaluation Configuration
RANDOM_SEED = 42  # For reproducible 5-shot sampling
N_SHOT_EXAMPLES = 5  # Number of examples for few-shot learning
N_NO_PII_EXAMPLES = 3  # Number of negative examples
N_WITH_PII_EXAMPLES = 2  # Number of positive examples

# Parallel Processing Configuration
PARALLEL_REQUESTS = 10  # Number of parallel API requests (default, can be overridden)

# Exponential Backoff Configuration
MAX_RETRIES = 3  # Maximum number of retry attempts for API calls
INITIAL_BACKOFF_SECONDS = 1  # Initial wait time before first retry
BACKOFF_MULTIPLIER = 2  # Multiplier for exponential backoff (wait_time = INITIAL_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER ^ retry_number))

# PII Types to detect (from task requirements)
PII_TYPES = [
    "[full name]",
    "[ssn]",
    "[ssn, last 4]",
    "[drivers license #]",
    "[passport #]",
    "[TIN]",
    "[irs identity protection pin]",
    "[student identification #]",
    "[bank account #]",
    "[credit card #]",
    "[cvv/cvc]",
    "[password]",
    "[username]",
    "[email address]",
    "[phone number]"
]

# PII Type descriptions for prompts
PII_DESCRIPTIONS = {
    "[full name]": "Full name - ONLY flag if paired with another PII element (email addresses don't count)",
    "[ssn]": "Social security number (XXX-XX-XXXX format)",
    "[ssn, last 4]": "Last 4 digits of SSN",
    "[drivers license #]": "Driver's license number",
    "[passport #]": "Passport number",
    "[TIN]": "Taxpayer identification number",
    "[irs identity protection pin]": "IRS identity protection PIN",
    "[student identification #]": "Student identification number",
    "[bank account #]": "Bank account number (NOT routing numbers)",
    "[credit card #]": "Credit or debit card numbers",
    "[cvv/cvc]": "Credit/debit card security code (CVV/CVC)",
    "[password]": "Passwords or security credentials",
    "[username]": "Login usernames (in context of credentials)",
    "[email address]": "Email addresses",
    "[phone number]": "Phone numbers"
}

# Projection configuration
TARGET_DATASET_SIZE = 50_000  # Project costs/time for 50k emails
