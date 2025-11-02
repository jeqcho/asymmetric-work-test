import pandas as pd
import json
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from collections import Counter, defaultdict

# Load the results and false positives
with open('/Users/jeqcho/asymmetric-work-test/results/presidio_strict_results.json', 'r') as f:
    results = json.load(f)

fp_df = pd.read_csv('/Users/jeqcho/asymmetric-work-test/results/false_positives/presidio_strict_fp.csv')

# Load the dataset to get full email text
dataset_df = pd.read_csv('/Users/jeqcho/asymmetric-work-test/task/Datasets/250_labeled_dataset.csv')

# Initialize Presidio
registry = RecognizerRegistry()
registry.load_predefined_recognizers()
analyzer = AnalyzerEngine(registry=registry, supported_languages=["en"])

# Categories for false positive analysis
fp_categories = defaultdict(int)
entity_type_counts = Counter()
examples_per_category = defaultdict(list)

print("Analyzing false positives...\n")

for idx, row in fp_df.iterrows():
    email_id = row['email_id']

    # Get full email text from dataset (ID column in dataset maps to email_id in results)
    email_data = dataset_df[dataset_df['ID'] == email_id].iloc[0]

    # Combine all text fields
    full_text = ""
    if pd.notna(email_data.get('Subject')):
        full_text += str(email_data['Subject']) + "\n"
    if pd.notna(email_data.get('From')):
        full_text += "From: " + str(email_data['From']) + "\n"
    if pd.notna(email_data.get('To')):
        full_text += "To: " + str(email_data['To']) + "\n"
    if pd.notna(email_data.get('Message Body')):
        full_text += str(email_data['Message Body'])

    # Analyze with Presidio
    analysis_results = analyzer.analyze(text=full_text, language='en', score_threshold=0.5)

    # Categorize by entity type
    entity_types = set([result.entity_type for result in analysis_results])

    # Extract detected entities
    detected_entities = []
    for result in analysis_results:
        entity_text = full_text[result.start:result.end]
        detected_entities.append({
            'type': result.entity_type,
            'text': entity_text,
            'score': result.score
        })
        entity_type_counts[result.entity_type] += 1

    # Classify the false positive
    categorized = False

    # Check for business names (common in work emails)
    if 'PERSON' in entity_types:
        person_entities = [e for e in detected_entities if e['type'] == 'PERSON']
        # If we see multiple person names or names in professional context
        if len(person_entities) >= 1:
            fp_categories['Business/Professional Names (PERSON entities in work context)'] += 1
            if len(examples_per_category['Business/Professional Names (PERSON entities in work context)']) < 3:
                examples_per_category['Business/Professional Names (PERSON entities in work context)'].append({
                    'email_id': email_id,
                    'entities': [e['text'] for e in person_entities[:3]],
                    'preview': full_text[:200].replace('\n', ' ')
                })
            categorized = True

    # Check for organizational emails
    if 'EMAIL_ADDRESS' in entity_types:
        email_entities = [e for e in detected_entities if e['type'] == 'EMAIL_ADDRESS']
        fp_categories['Email Addresses (business/organizational emails)'] += 1
        if len(examples_per_category['Email Addresses (business/organizational emails)']) < 3:
            examples_per_category['Email Addresses (business/organizational emails)'].append({
                'email_id': email_id,
                'entities': [e['text'] for e in email_entities[:3]],
                'preview': full_text[:200].replace('\n', ' ')
            })
        categorized = True

    # URLs/Links
    if 'URL' in entity_types:
        url_entities = [e for e in detected_entities if e['type'] == 'URL']
        fp_categories['URLs/Web Links'] += 1
        if len(examples_per_category['URLs/Web Links']) < 3:
            examples_per_category['URLs/Web Links'].append({
                'email_id': email_id,
                'entities': [e['text'] for e in url_entities[:3]],
                'preview': full_text[:200].replace('\n', ' ')
            })
        categorized = True

    # Dates and times
    if 'DATE_TIME' in entity_types:
        date_entities = [e for e in detected_entities if e['type'] == 'DATE_TIME']
        fp_categories['Date/Time Information (scheduling/timestamps)'] += 1
        if len(examples_per_category['Date/Time Information (scheduling/timestamps)']) < 3:
            examples_per_category['Date/Time Information (scheduling/timestamps)'].append({
                'email_id': email_id,
                'entities': [e['text'] for e in date_entities[:3]],
                'preview': full_text[:200].replace('\n', ' ')
            })
        categorized = True

    # Phone numbers
    if 'PHONE_NUMBER' in entity_types:
        phone_entities = [e for e in detected_entities if e['type'] == 'PHONE_NUMBER']
        fp_categories['Phone Numbers/Extensions (business contact info)'] += 1
        if len(examples_per_category['Phone Numbers/Extensions (business contact info)']) < 3:
            examples_per_category['Phone Numbers/Extensions (business contact info)'].append({
                'email_id': email_id,
                'entities': [e['text'] for e in phone_entities[:3]],
                'preview': full_text[:200].replace('\n', ' ')
            })
        categorized = True

    # Location
    if 'LOCATION' in entity_types:
        loc_entities = [e for e in detected_entities if e['type'] == 'LOCATION']
        fp_categories['Location/Address (business addresses, cities)'] += 1
        if len(examples_per_category['Location/Address (business addresses, cities)']) < 3:
            examples_per_category['Location/Address (business addresses, cities)'].append({
                'email_id': email_id,
                'entities': [e['text'] for e in loc_entities[:3]],
                'preview': full_text[:200].replace('\n', ' ')
            })
        categorized = True

    # Check for number patterns that might trigger SSN/CC false positives
    if 'US_SSN' in entity_types or 'CREDIT_CARD' in entity_types:
        fp_categories['Number Patterns (dates, IDs, formatting)'] += 1
        if len(examples_per_category['Number Patterns (dates, IDs, formatting)']) < 3:
            examples_per_category['Number Patterns (dates, IDs, formatting)'].append({
                'email_id': email_id,
                'entities': [e['text'] for e in detected_entities if e['type'] in ['US_SSN', 'CREDIT_CARD']][:3],
                'preview': full_text[:200].replace('\n', ' ')
            })
        categorized = True

    # Other
    if not categorized:
        fp_categories['Other/Multiple Entity Types'] += 1
        if len(examples_per_category['Other/Multiple Entity Types']) < 3:
            examples_per_category['Other/Multiple Entity Types'].append({
                'email_id': email_id,
                'entities': [e['text'] for e in detected_entities[:3]],
                'preview': full_text[:200].replace('\n', ' ')
            })

# Calculate percentages
total_fp = len(fp_df)

# Create summary table
print("=" * 100)
print("FALSE POSITIVE ANALYSIS SUMMARY - PRESIDIO STRICT")
print("=" * 100)
print(f"\nTotal False Positives Analyzed: {total_fp}\n")

summary_data = []
for category, count in sorted(fp_categories.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / total_fp) * 100
    summary_data.append({
        'Reason for False Positive': category,
        'Count': count,
        'Percentage': f"{percentage:.1f}%"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
print("\n")

# Show examples
print("=" * 100)
print("EXAMPLES FROM EACH CATEGORY")
print("=" * 100)

for category in sorted(fp_categories.keys(), key=lambda x: fp_categories[x], reverse=True):
    if category in examples_per_category and examples_per_category[category]:
        print(f"\n{category} ({fp_categories[category]} occurrences):")
        print("-" * 100)
        for i, example in enumerate(examples_per_category[category], 1):
            print(f"\n  Example {i}:")
            print(f"  Email ID: {example['email_id']}")
            print(f"  Detected: {example['entities']}")
            print(f"  Context: \"{example['preview']}...\"")
        print()

# Entity type distribution
print("=" * 100)
print("ENTITY TYPE DISTRIBUTION ACROSS ALL FALSE POSITIVES")
print("=" * 100)

entity_dist_df = pd.DataFrame([
    {'Entity Type': entity_type, 'Total Detections': count}
    for entity_type, count in entity_type_counts.most_common()
])
print(entity_dist_df.to_string(index=False))

print("\n" + "=" * 100)
print("KEY INSIGHTS")
print("=" * 100)
print("""
The false positives occur primarily because:
1. Work emails contain many business names that Presidio flags as PERSON entities
2. Business email addresses and contact information are detected as PII
3. Scheduling information (dates, times) triggers DATE_TIME entity detection
4. Business locations and addresses are flagged as LOCATION entities

These are not actually sensitive PII in a business context, but generic PII detectors
flag them because they technically match the patterns for person names, contact info, etc.
""")

print("=" * 100)
