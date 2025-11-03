"""Configuration settings for production PII detection pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
LABELED_DATASET_PATH = PROJECT_ROOT / "task" / "Datasets" / "250_labeled_dataset.csv"

# API Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model Configuration
HAIKU_MODEL = "claude-haiku-4-5"

# Pricing (USD per million tokens) - as of January 2025
HAIKU_INPUT_PRICE = 0.80 / 1_000_000
HAIKU_OUTPUT_PRICE = 4.00 / 1_000_000

# Presidio Configuration
PRESIDIO_THRESHOLD = 0.0  # Most strict - catches everything

# Evaluation Configuration
RANDOM_SEED = 42  # For reproducible 5-shot sampling
N_SHOT_EXAMPLES = 5  # Number of examples for few-shot learning
N_NO_PII_EXAMPLES = 3  # Number of negative examples
N_WITH_PII_EXAMPLES = 2  # Number of positive examples

# Parallel Processing Configuration
PARALLEL_REQUESTS = 10  # Number of parallel API requests (default)

# Exponential Backoff Configuration
MAX_RETRIES = 3  # Maximum number of retry attempts for API calls
INITIAL_BACKOFF_SECONDS = 1  # Initial wait time before first retry
BACKOFF_MULTIPLIER = 2  # Multiplier for exponential backoff

# PII Types to detect (from MVP list) - normalized format for CSV columns
PII_TYPES = [
    "full name",
    "ssn",
    "ssn last 4",
    "drivers license #",
    "passport #",
    "TIN",
    "irs identity protection pin",
    "student identification #",
    "bank account #",
    "credit card #",
    "cvv/cvc",
    "password",
    "username",
    "email address",
    "phone number"
]

# PII Type descriptions for prompts (bracket format)
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

