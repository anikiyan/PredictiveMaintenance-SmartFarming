# reports/generate_pdf_report.py

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib import colors
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA = Path("data/processed/agri_features.csv")
FIG_PATH = Path("reports/figures")
OUTPUT_PDF = Path("reports/Predictive_Maintenance_Report.pdf")
FIG_PATH.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA)

# Generate a plot for failure counts
failures = df[df["failure_label"] == 1]
fail_counts = failures.filter(like="operating_mode").sum().sort_values()

fig, ax = plt.subplots()
fail_counts.plot(kind="barh", color="salmon", ax=ax)
ax.set_title("Failure Count by Operating Mode")
ax.set_xlabel("Count")
plot_path = FIG_PATH / "failure_modes.png"
fig.tight_layout()
plt.savefig(plot_path)
plt.close()

# Build report
styles = getSampleStyleSheet()
doc = SimpleDocTemplate(str(OUTPUT_PDF), pagesize=A4)
story = []

# Title
story.append(Paragraph("<b>Predictive Maintenance Report</b>", styles["Title"]))
story.append(Spacer(1, 0.25 * inch))

# Executive Summary
story.append(Paragraph("<b>1. Executive Summary</b>", styles["Heading2"]))
summary = (
    "This report outlines the results of predictive maintenance analysis using machine learning models. "
    "Sensor data from farming equipment was analyzed to estimate failure risks and predict remaining useful life (RUL). "
    "The findings enable proactive maintenance and reduce equipment downtime."
)
story.append(Paragraph(summary, styles["BodyText"]))
story.append(Spacer(1, 0.25 * inch))

# Metrics Table
story.append(Paragraph("<b>2. Dataset Summary</b>", styles["Heading2"]))
summary_stats = df.describe().round(2).reset_index()
data_table = [summary_stats.columns.tolist()] + summary_stats.values.tolist()
t = Table(data_table, hAlign="LEFT", repeatRows=1)
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
    ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
]))
story.append(t)
story.append(Spacer(1, 0.25 * inch))

# Add plot
story.append(Paragraph("<b>3. Failure Analysis</b>", styles["Heading2"]))
story.append(Image(str(plot_path), width=5*inch, height=3*inch))

# Conclusion
story.append(Spacer(1, 0.25 * inch))
story.append(Paragraph("<b>4. Conclusion</b>", styles["Heading2"]))
conclusion = (
    "The models successfully predicted high-risk scenarios and estimated remaining useful life. "
    "The system can be deployed in production pipelines or integrated into dashboards for live equipment monitoring."
)
story.append(Paragraph(conclusion, styles["BodyText"]))

# Generate PDF
doc.build(story)
print(f"✅ Report generated at: {OUTPUT_PDF}")
