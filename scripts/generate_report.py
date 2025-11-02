"""Generate C-suite PDF performance report."""

import sys
from pathlib import Path
import csv
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from src.config import RESULTS_DIR, VISUALIZATIONS_DIR


def load_metrics():
    """Load metrics from comparison CSV."""
    metrics = []
    csv_path = RESULTS_DIR / "comparison_metrics.csv"

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tp = int(row['TP'])
            fp = int(row['FP'])
            tn = int(row['TN'])
            fn = int(row['FN'])

            # Calculate FNR and FPR
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            metrics.append({
                'detector': row['Detector'],
                'fnr': fnr,
                'fpr': fpr,
                'time_current': row['Projected Time 50k (min)'],
                'cost': row['Projected Cost 50k (USD)']
            })

    return metrics


def create_report():
    """Generate the PDF report."""
    output_path = RESULTS_DIR / "performance_report.pdf"
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                           topMargin=0.5*inch, bottomMargin=0.5*inch,
                           leftMargin=0.75*inch, rightMargin=0.75*inch)

    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=6,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=6,
        spaceBefore=8
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=4
    )

    small_style = ParagraphStyle(
        'Small',
        parent=styles['Normal'],
        fontSize=8,
        spaceAfter=3
    )

    # Title
    story.append(Paragraph("PII Detection: Performance Analysis", title_style))
    story.append(Paragraph(f"<i>{datetime.now().strftime('%B %Y')}</i>",
                          ParagraphStyle('subtitle', parent=styles['Normal'],
                                       fontSize=9, alignment=TA_CENTER, textColor=colors.grey)))
    story.append(Spacer(1, 0.15*inch))

    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    summary_text = """Email breach analysis evaluated on 250-sample gold-labeled dataset.
    <b>Primary goal: Zero false negatives to avoid missing PII.</b>
    Finding: Presidio achieves 0% FNR but 88% FPR (unusable in production).
    Sonnet zero-shot delivers 4.3% FNR with acceptable 22% FPR."""
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 0.1*inch))

    # Detector Overview Table
    story.append(Paragraph("Detector Overview", heading_style))
    detector_data = [
        ['Detector', 'Type', 'Description'],
        ['Presidio', 'Local/Free', 'Rule-based NLP, runs locally'],
        ['Haiku Zero-shot', 'Claude API', 'Fast LLM, no examples'],
        ['Sonnet Zero-shot', 'Claude API', 'Slower LLM, higher accuracy']
    ]

    detector_table = Table(detector_data, colWidths=[1.3*inch, 1.1*inch, 3.6*inch])
    detector_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0e0e0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(detector_table)
    story.append(Spacer(1, 0.15*inch))

    # Performance Metrics
    story.append(Paragraph("Performance Metrics", heading_style))

    # Load metrics
    metrics = load_metrics()

    # Filter to only show key detectors (remove 5-shot and duplicate Presidio)
    key_detectors = ['presidio_moderate', 'haiku_zeroshot', 'sonnet_zeroshot']
    filtered_metrics = [m for m in metrics if m['detector'] in key_detectors]

    # Performance table
    perf_data = [['Detector', 'FNR ↓', 'FPR ↓', 'Status']]

    for m in filtered_metrics:
        detector_name = m['detector'].replace('_', ' ').title().replace('Zeroshot', 'Zero-shot')
        fnr_str = f"{m['fnr']:.1%}"
        fpr_str = f"{m['fpr']:.1%}"

        if m['detector'] == 'presidio_moderate':
            status = '⚠️ Unusable'
        elif m['detector'] == 'sonnet_zeroshot':
            status = '✓ Best usable'
        else:
            status = '✓ Good'

        perf_data.append([detector_name, fnr_str, fpr_str, status])

    perf_table = Table(perf_data, colWidths=[2.2*inch, 1*inch, 1*inch, 1.8*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0e0e0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 1), (2, -1), 'CENTER'),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 0.1*inch))

    # Add FNR chart
    fnr_chart_path = VISUALIZATIONS_DIR / "classification_rates.png"
    if fnr_chart_path.exists():
        # Use the full classification rates chart
        img = Image(str(fnr_chart_path), width=6.5*inch, height=4.875*inch)
        story.append(img)

    story.append(PageBreak())

    # PAGE 2: Cost/Time & Next Steps
    story.append(Paragraph("50k Email Projections", heading_style))

    # Cost/Time table
    cost_data = [['Detector', 'Time (Current)', 'Time (Parallel)', 'Cost', 'FNR']]

    for m in filtered_metrics:
        detector_name = m['detector'].replace('_', ' ').title().replace('Zeroshot', 'Zero-shot')
        time_current = m['time_current']
        fnr_str = f"{m['fnr']:.1%}"
        cost_str = f"${m['cost']}"

        if m['detector'] == 'presidio_moderate':
            time_parallel = '2-3 min*'
        elif 'haiku' in m['detector']:
            time_parallel = '15-74 min**'
        else:  # sonnet
            time_parallel = '35-173 min**'

        cost_data.append([
            detector_name,
            f"{time_current} min",
            time_parallel,
            cost_str,
            fnr_str
        ])

    cost_table = Table(cost_data, colWidths=[1.8*inch, 1.2*inch, 1.3*inch, 0.9*inch, 0.8*inch])
    cost_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0e0e0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    story.append(cost_table)

    # Footnotes
    story.append(Paragraph("*With EC2 multi-core optimization", small_style))
    story.append(Paragraph("**With API parallelization (10-50x speedup; currently sequential)", small_style))
    story.append(Spacer(1, 0.15*inch))

    # Next Steps
    story.append(Paragraph("Next Steps", heading_style))

    next_steps = [
        "<b>1. Implement API parallelization</b> - 10-50x speedup potential (1-2 days development)",
        "<b>2. EC2 optimization for Presidio</b> - 5-10x speedup with multi-core processing",
        "<b>3. Tune Presidio to reduce FPR</b> - Goal: maintain 0% FNR while reducing 88% FPR"
    ]

    for step in next_steps:
        story.append(Paragraph(f"• {step}", body_style))

    story.append(Spacer(1, 0.15*inch))

    # Recommendation box
    rec_style = ParagraphStyle(
        'Recommendation',
        parent=styles['Normal'],
        fontSize=9,
        leftIndent=10,
        rightIndent=10,
        spaceAfter=6,
        borderWidth=1,
        borderColor=colors.HexColor('#4a90e2'),
        borderPadding=8,
        backColor=colors.HexColor('#f0f8ff')
    )

    recommendation = """<b>Recommendation:</b> Deploy Sonnet zero-shot (4.3% FNR) with API parallelization
    for production 50k emails. Cost: $107, Time: ~35-173 min (parallelized).
    Alternative: Haiku zero-shot if 17.4% FNR is acceptable ($28, ~15-74 min)."""
    story.append(Paragraph(recommendation, rec_style))

    # Build PDF
    doc.build(story)
    print(f"\n✓ Report generated: {output_path}")
    return output_path


if __name__ == "__main__":
    print("Generating C-suite performance report...")
    report_path = create_report()
    print(f"\nReport saved to: {report_path}")
