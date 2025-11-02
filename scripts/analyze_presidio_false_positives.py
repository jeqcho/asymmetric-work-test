import pandas as pd
import json
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from collections import Counter, defaultdict
import re

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
fp_categories = defaultdict(list)

print("Analyzing false positives...\n")

for idx, row in fp_df.iterrows():
    email_id = row['email_id']

    # Get full email text from dataset
    email_data = dataset_df[dataset_df['email_id'] == email_id].iloc[0]

    # Combine all text fields
    full_text = ""
    if pd.notna(email_data.get('subject')):
        full_text += str(email_data['subject']) + "\n"
    if pd.notna(email_data.get('from')):
        full_text += "From: " + str(email_data['from']) + "\n"
    if pd.notna(email_data.get('to')):
        full_text += "To: " + str(email_data['to']) + "\n"
    if pd.notna(email_data.get('message_body')):
        full_text += str(email_data['message_body'])

    # Analyze with Presidio
    analysis_results = analyzer.analyze(text=full_text, language='en', score_threshold=0.5)

    # Categorize by entity type
    entity_types = set([result.entity_type for result in analysis_results])

    # Extract actual detected entities
    detected_entities = []
    for result in analysis_results:
        entity_text = full_text[result.start:result.end]
        detected_entities.append({
            'type': result.entity_type,
            'text': entity_text,
            'score': result.score
        })

    # Determine primary category
    if 'PERSON' in entity_types:
        # Check if it's a common name pattern
        person_entities = [e['text'] for e in detected_entities if e['type'] == 'PERSON']
        # Common business name patterns (First + Last in professional context)
        if any(len(name.split()) >= 2 for name in person_entities):
            fp_categories['Business/Professional Names'].append({
                'email_id': email_id,
                'entities': detected_entities,
                'preview': row['message_body_preview'][:100] if pd.notna(row.get('message_body_preview')) else ''
            })
        else:
            fp_categories['Single Names'].append({
                'email_id': email_id,
                'entities': detected_entities,
                'preview': row['message_body_preview'][:100] if pd.notna(row.get('message_body_preview')) else ''
            })

    if 'EMAIL_ADDRESS' in entity_types:
        email_entities = [e['text'] for e in detected_entities if e['type'] == 'EMAIL_ADDRESS']
        # Check if it's an organizational email (e.g., enron.com)
        if any('@enron.com' in email.lower() or '@hotmail.com' in email.lower() for email in email_entities):
            fp_categories['Organizational Email Addresses'].append({
                'email_id': email_id,
                'entities': detected_entities,
                'preview': row['message_body_preview'][:100] if pd.notna(row.get('message_body_preview')) else ''
            })

    if 'PHONE_NUMBER' in entity_types:
        fp_categories['Phone Numbers'].append({
            'email_id': email_id,
            'entities': detected_entities,
            'preview': row['message_body_preview'][:100] if pd.notna(row.get('message_body_preview')) else ''
        })

    if 'US_SSN' in entity_types:
        fp_categories['SSN-like Patterns'].append({
            'email_id': email_id,
            'entities': detected_entities,
            'preview': row['message_body_preview'][:100] if pd.notna(row.get('message_body_preview')) else ''
        })

    if 'CREDIT_CARD' in entity_types:
        fp_categories['Credit Card-like Patterns'].append({
            'email_id': email_id,
            'entities': detected_entities,
            'preview': row['message_body_preview'][:100] if pd.notna(row.get('message_body_preview')) else ''
        })

    if 'LOCATION' in entity_types or 'US_DRIVER_LICENSE' in entity_types:
        fp_categories['Location/Address Information'].append({
            'email_id': email_id,
            'entities': detected_entities,
            'preview': row['message_body_preview'][:100] if pd.notna(row.get('message_body_preview')) else ''
        })

    if 'DATE_TIME' in entity_types:
        fp_categories['Date/Time Patterns'].append({
            'email_id': email_id,
            'entities': detected_entities,
            'preview': row['message_body_preview'][:100] if pd.notna(row.get('message_body_preview')) else ''
        })

    if 'URL' in entity_types:
        fp_categories['URLs'].append({
            'email_id': email_id,
            'entities': detected_entities,
            'preview': row['message_body_preview'][:100] if pd.notna(row.get('message_body_preview')) else ''
        })

    # If no specific category matched, add to general
    if email_id not in [item['email_id'] for cat_list in fp_categories.values() for item in cat_list]:
        fp_categories['Other'].append({
            'email_id': email_id,
            'entities': detected_entities,
            'preview': row['message_body_preview'][:100] if pd.notna(row.get('message_body_preview')) else ''
        })

# Calculate percentages
total_fp = len(fp_df)

# Create summary table
print("=" * 80)
print("FALSE POSITIVE ANALYSIS SUMMARY")
print("=" * 80)
print(f"\nTotal False Positives: {total_fp}\n")

summary_data = []
for category, items in sorted(fp_categories.items(), key=lambda x: len(x[1]), reverse=True):
    count = len(items)
    percentage = (count / total_fp) * 100
    summary_data.append({
        'Category': category,
        'Count': count,
        'Percentage': f"{percentage:.1f}%"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
print("\n")

# Show examples from top categories
print("=" * 80)
print("EXAMPLES FROM TOP CATEGORIES")
print("=" * 80)

for category, items in sorted(fp_categories.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
    print(f"\n{category} ({len(items)} cases):")
    print("-" * 80)

    # Show first 3 examples
    for item in items[:3]:
        print(f"\nEmail ID: {item['email_id']}")
        print(f"Detected entities: {[e['type'] + ': ' + e['text'] for e in item['entities'][:5]]}")
        print(f"Preview: {item['preview']}")
    print()

# Entity type distribution
print("=" * 80)
print("ENTITY TYPE DISTRIBUTION")
print("=" * 80)

entity_type_counts = Counter()
for category, items in fp_categories.items():
    for item in items:
        for entity in item['entities']:
            entity_type_counts[entity['type']] += 1

entity_dist_df = pd.DataFrame([
    {'Entity Type': entity_type, 'Count': count}
    for entity_type, count in entity_type_counts.most_common()
])
print(entity_dist_df.to_string(index=False))

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
