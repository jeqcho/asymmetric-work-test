# Presidio Strict False Positive Analysis

## Summary

Analysis of 201 false positives from Presidio strict detector on the email dataset.

## False Positive Breakdown by Category

| Reason for False Positive | Count | Percentage | Example Detection |
|---------------------------|-------|------------|-------------------|
| **Business/Professional Names** | 201 | 100.0% | Names like "Phillip K Allen", "John J Lavorato" in work email headers and signatures |
| **Date/Time Information** | 144 | 71.6% | Scheduling info like "Tuesday at 11:45", "10/03/2000" in meeting coordination |
| **URLs/Web Links** | 98 | 48.8% | Domain names like "hotmail.com", "austin.rr.com" in email addresses |
| **Email Addresses** | 91 | 45.3% | Business emails like "stagecoachmama@hotmail.com", "pallen70@hotmail.com" |
| **Location/Address** | 51 | 25.4% | Business locations like "Austin", city names, office locations |
| **Phone Numbers/Extensions** | 5 | 2.5% | Business contact numbers like "512-748-7495", office extensions |

**Note:** Percentages sum to >100% because individual emails often trigger multiple categories.

## Entity Type Distribution

| Entity Type | Total Detections Across All False Positives |
|-------------|---------------------------------------------|
| PERSON | 1,732 |
| DATE_TIME | 738 |
| URL | 230 |
| EMAIL_ADDRESS | 165 |
| LOCATION | 152 |
| PHONE_NUMBER | 7 |
| NRP | 3 |
| US_DRIVER_LICENSE | 3 |

## Key Findings

### Why These Are False Positives:

1. **Business Names (100% of FPs)**: Every false positive contains person names, but these are business contacts, email senders/recipients, and professional colleagues mentioned in work context. These are not sensitive personal information requiring redaction.

2. **Scheduling Information (71.6% of FPs)**: Dates and times are detected because they follow PII patterns, but in business emails, scheduling information ("Tuesday at 11:45", "10/03/2000") is normal operational data, not sensitive PII.

3. **Business Contact Information (48.8% URLs, 45.3% emails)**: Email addresses and web domains are flagged, but in business context, these are organizational contact points (e.g., company domains like "enron.com", business email accounts), not personal sensitive data.

4. **Business Locations (25.4% of FPs)**: Location entities are detected (cities, office names), but these refer to business locations and office sites, not personal home addresses.

## Example False Positives

### Example 1: Business Communication
- **Email ID**: 2
- **Detected**: "Phillip K Allen", "John J Lavorato"
- **Context**: Work email with professional names in From/To fields
- **Why FP**: These are business colleagues in a professional email header

### Example 2: Meeting Scheduling
- **Email ID**: 5
- **Detected**: "Tuesday", "11:45"
- **Context**: "Let's shoot for Tuesday at 11:45"
- **Why FP**: Scheduling information for a business meeting

### Example 3: Business Email Address
- **Email ID**: 12
- **Detected**: "stagecoachmama@hotmail.com"
- **Context**: Email discussing business matters
- **Why FP**: Business/work-related email address, not personal sensitive info

## Conclusion

**Presidio's high false positive rate (201 FPs out of 227 NO-PII emails = 88.5% false positive rate on non-PII emails)** occurs because:

1. Generic PII detectors are trained on patterns (person names, emails, dates, locations) without understanding **business context**
2. Work emails naturally contain many entities that match PII patterns but aren't sensitive in a professional setting
3. Presidio cannot distinguish between:
   - Personal home address vs. office location
   - Personal email vs. business email
   - Sensitive personal name vs. business contact name
   - Private scheduling info vs. business meeting times

**Implication**: For business email PII detection, a context-aware approach (like LLM-based detection) that understands when names/contacts/locations are business-related rather than personal is crucial to reduce false positives.
