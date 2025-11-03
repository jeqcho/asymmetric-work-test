# Example Prompts for Hybrid LLM + Presidio Detectors

This document shows example prompts that will be fed to Haiku zero-shot and 5-shot models during evaluation.

## Haiku Zero-Shot Prompt

```
You are a PII detection expert for email breach analysis.

TASK: Determine if this email contains ANY Personal Identifiable Information (PII).

PII TYPES TO DETECT:
1. [full name] - Full name - ONLY flag if paired with another PII element (email addresses don't count)
2. [ssn] - Social security number (XXX-XX-XXXX format)
3. [ssn, last 4] - Last 4 digits of SSN
4. [drivers license #] - Driver's license number
5. [passport #] - Passport number
6. [TIN] - Taxpayer identification number
7. [irs identity protection pin] - IRS identity protection PIN
8. [student identification #] - Student identification number
9. [bank account #] - Bank account number (NOT routing numbers)
10. [credit card #] - Credit or debit card numbers
11. [cvv/cvc] - Credit/debit card security code (CVV/CVC)
12. [password] - Passwords or security credentials
13. [username] - Login usernames (in context of credentials)
14. [email address] - Email addresses
15. [phone number] - Phone numbers

IMPORTANT RULES:
1. Focus ONLY on the message body for PII detection
2. Ignore corporate/business information - we only care about PII
3. [full name] ONLY counts if it appears WITH another PII element (email addresses don't count)
4. Flag if you have any suspicion it could be PII

PRESIDIO OUTPUT:
The Presidio tool has analyzed this email and found the following entities:
EMAIL_ADDRESS (0.95): 'john.doe@example.com'
PHONE_NUMBER (0.82): '555-1234'
PERSON (0.65): 'John Doe'

CRITICAL: Presidio may miss PII. Even if Presidio found nothing or only low-confidence matches, you must carefully review the entire email message body for PII of concern. Presidio's output is meant to help guide your analysis, but you should check for any PII that Presidio might have missed.

EMAIL TO ANALYZE:
Subject: Re: Account Information
From: support@company.com
To: customer@example.com
Message Body: Dear John Doe,

Thank you for your inquiry. Your account number is 1234567890. Please call us at 555-1234 if you have any questions.

Best regards,
Support Team

Answer with ONLY "YES" if the email contains PII, or "NO" if it doesn't. Do not provide any explanation, just YES or NO.
```

## Haiku 5-Shot Prompt

```
You are a PII detection expert for email breach analysis.

TASK: Determine if this email contains ANY Personal Identifiable Information (PII).

PII TYPES TO DETECT:
1. [full name] - Full name - ONLY flag if paired with another PII element (email addresses don't count)
2. [ssn] - Social security number (XXX-XX-XXXX format)
3. [ssn, last 4] - Last 4 digits of SSN
4. [drivers license #] - Driver's license number
5. [passport #] - Passport number
6. [TIN] - Taxpayer identification number
7. [irs identity protection pin] - IRS identity protection PIN
8. [student identification #] - Student identification number
9. [bank account #] - Bank account number (NOT routing numbers)
10. [credit card #] - Credit or debit card numbers
11. [cvv/cvc] - Credit/debit card security code (CVV/CVC)
12. [password] - Passwords or security credentials
13. [username] - Login usernames (in context of credentials)
14. [email address] - Email addresses
15. [phone number] - Phone numbers

IMPORTANT RULES:
1. Focus ONLY on the message body for PII detection
2. Ignore corporate/business information - we only care about PII
3. [full name] ONLY counts if it appears WITH another PII element (email addresses don't count)
4. Flag if you have any suspicion it could be PII

CRITICAL: Presidio may miss PII. Even if Presidio found nothing or only low-confidence matches, you must carefully review the entire email message body for PII of concern. Presidio's output is meant to help guide your analysis, but you should check for any PII that Presidio might have missed.

Here are 5 examples to guide your analysis:

EXAMPLE 1 (NO PII):
Subject: Meeting Reminder
From: manager@company.com
To: team@company.com
Message Body: Please remember our team meeting tomorrow at 2pm in Conference Room A.
Presidio Output: None
Data Elements: None
Answer: NO

EXAMPLE 2 (NO PII):
Subject: Project Update
From: pm@company.com
To: stakeholders@company.com
Message Body: The Q4 project is on track. We expect completion by month end.
Presidio Output: EMAIL_ADDRESS (0.92): 'stakeholders@company.com'
Data Elements: None
Answer: NO

EXAMPLE 3 (NO PII):
Subject: Weekly Report
From: analyst@company.com
To: director@company.com
Message Body: This week's sales figures show a 15% increase. Details attached.
Presidio Output: None
Data Elements: None
Answer: NO

EXAMPLE 4 (WITH PII):
Subject: Account Verification
From: bank@example.com
To: customer@example.com
Message Body: Dear Sarah Johnson,

To verify your account, please confirm your SSN ending in 1234.

Best regards,
Bank Support
Presidio Output: PERSON (0.88): 'Sarah Johnson'
EMAIL_ADDRESS (0.95): 'customer@example.com'
Data Elements: [ssn, last 4], [full name]
Answer: YES

EXAMPLE 5 (WITH PII):
Subject: Password Reset Request
From: admin@example.com
To: user@example.com
Message Body: Your temporary password is: TempPass123!

Username: jsmith
Account: 9876543210

Please change your password after logging in.
Presidio Output: PASSWORD (0.90): 'TempPass123!'
USERNAME (0.85): 'jsmith'
US_BANK_NUMBER (0.75): '9876543210'
Data Elements: [password], [username], [bank account #]
Answer: YES

Now analyze this email:

EMAIL TO ANALYZE:
Subject: Re: Account Information
From: support@company.com
To: customer@example.com
Message Body: Dear John Doe,

Thank you for your inquiry. Your account number is 1234567890. Please call us at 555-1234 if you have any questions.

Best regards,
Support Team
Presidio Output: EMAIL_ADDRESS (0.95): 'customer@example.com'
PHONE_NUMBER (0.82): '555-1234'
PERSON (0.65): 'John Doe'

Answer with ONLY "YES" if the email contains PII, or "NO" if it doesn't. Do not provide any explanation, just YES or NO.
```

## Key Features

1. **Presidio Output Format**: Each entity shows as `ENTITY_TYPE (confidence_score): 'extracted_text'`
2. **Critical Instruction**: Clear instruction that Presidio may miss PII and the LLM should check beyond Presidio's output
3. **5-Shot Examples**: Include Presidio output in all examples, showing both cases where Presidio finds entities and where it finds nothing
4. **Consistent Format**: Both zero-shot and 5-shot prompts follow the same structure with Presidio output integrated

