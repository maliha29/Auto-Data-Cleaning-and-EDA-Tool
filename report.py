"""report.py - builds the downloadable PDF report."""
from datetime import date
from io import BytesIO
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from cleaning import health
from insights import generate_insights, top_correlations
from ml_readiness import ml_checks
from quality import quality_scores
from recommendations import get_recommendations

NAVY = colors.HexColor("#1f3a5f")
styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor=NAVY, spaceBefore=14, spaceAfter=6)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14)
CELL = ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=8.5, leading=11)
CELL_HEAD = ParagraphStyle("CellHead", parent=CELL, textColor=colors.white, fontName="Helvetica-Bold")


def para(text, style=BODY):
    return Paragraph(escape(str(text)), style)


def bullets(items):
    return [Paragraph("&bull; " + escape(str(t)), BODY) for t in items]


def table(rows, widths=None):
    """rows[0] is the header row."""
    data = [[para(c, CELL_HEAD) for c in rows[0]]]
    data += [[para(c, CELL) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c5cdd8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def heatmap_image(num):
    """Draws a correlation heatmap with matplotlib and returns it as a PDF image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    corr = num.corr()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    if n <= 10:
        for i in range(n):
            for j in range(n):
                v = corr.iloc[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=11 * cm, height=11 * 6 / 7 * cm)


def page_number(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")


def build_report(original, clean, log, source_name):
    """Returns the PDF as bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Data Cleaning and EDA Report")
    s = []

    # ----- Title -----
    s.append(Paragraph("Data Cleaning &amp; EDA Report", ParagraphStyle(
        "Title", parent=styles["Title"], textColor=NAVY)))
    s.append(para(f"File: {source_name}   |   Generated: {date.today():%d %B %Y}"))

    # ----- 1. Dataset overview -----
    s.append(Paragraph("1. Dataset Overview", H1))
    s.append(para(f"The original dataset has {original.shape[0]:,} rows and {original.shape[1]} columns. "
                  f"After cleaning it has {clean.shape[0]:,} rows and {clean.shape[1]} columns."))
    s.append(Spacer(1, 6))
    rows = [["Column", "Type", "Missing %", "Unique values"]]
    for c in clean.columns[:30]:
        rows.append([c, clean[c].dtype, f"{clean[c].isna().mean() * 100:.1f}", clean[c].nunique()])
    s.append(table(rows, [6 * cm, 3.5 * cm, 3 * cm, 4 * cm]))
    if clean.shape[1] > 30:
        s.append(para(f"(Showing the first 30 of {clean.shape[1]} columns.)"))

    # ----- 2. Cleaning summary -----
    s.append(Paragraph("2. Data Cleaning Summary", H1))
    steps = [x for x in log if x != "Original data"]
    if steps:
        s += bullets([f"Step {i}: {x}" for i, x in enumerate(steps, 1)])
    else:
        s.append(para("No cleaning steps were applied."))
    s.append(Spacer(1, 8))
    m0, d0, h0 = health(original)
    m1, d1, h1 = health(clean)
    s.append(table([
        ["Measure", "Original", "Cleaned"],
        ["Rows", f"{original.shape[0]:,}", f"{clean.shape[0]:,}"],
        ["Missing cells", m0, m1],
        ["Duplicate rows", d0, d1],
        ["Simple quality score", f"{h0} / 100", f"{h1} / 100"],
    ], [6 * cm, 5 * cm, 5 * cm]))

    # ----- 3. Quality report -----
    s.append(Paragraph("3. Data Quality Report", H1))
    q0, q1 = quality_scores(original), quality_scores(clean)
    s.append(table([["Dimension", "Original", "Cleaned"]] +
                   [[k, f"{q0[k]}", f"{q1[k]}"] for k in q1], [6 * cm, 5 * cm, 5 * cm]))
    s.append(Spacer(1, 4))
    s.append(para("Completeness = how much data is filled in. Consistency = the same thing is written the same way. "
                  "Validity = values are stored in the right type. Accuracy = few extreme outliers. "
                  "Overall = the average of the four."))

    # ----- 4. EDA findings -----
    s.append(Paragraph("4. EDA Findings", H1))
    num = clean.select_dtypes("number")
    if num.shape[1]:
        rows = [["Column", "Mean", "Std", "Min", "Median", "Max"]]
        for c in num.columns[:15]:
            x = num[c]
            rows.append([c, f"{x.mean():,.2f}", f"{x.std():,.2f}", f"{x.min():,.2f}",
                         f"{x.median():,.2f}", f"{x.max():,.2f}"])
        s.append(table(rows, [4 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]))
    else:
        s.append(para("There are no numeric columns."))

    # ----- 5. Correlation -----
    s.append(Paragraph("5. Correlation Analysis", H1))
    pairs = top_correlations(clean, 5)
    if pairs:
        s += bullets([f"{a} and {b}: r = {r:.2f}" for a, b, r in pairs])
        img = heatmap_image(num)
        if img:
            s.append(Spacer(1, 8))
            s.append(img)
    else:
        s.append(para("At least two numeric columns are needed for correlation analysis."))

    # ----- 6. AI insights -----
    s.append(Paragraph("6. AI Insights", H1))
    s += bullets(generate_insights(clean))

    # ----- 7. Recommendations -----
    s.append(Paragraph("7. Recommendations", H1))
    recs = get_recommendations(clean)
    if recs:
        s.append(para("These issues are still present in the cleaned data:"))
        s += bullets([f"{r['title']} (confidence {r['confidence']}%): {r['why']}" for r in recs])
    else:
        s.append(para("No further cleaning is needed."))
    score, checks = ml_checks(clean)
    s.append(Spacer(1, 6))
    s.append(para(f"Machine learning readiness score: {score} / 100"))
    s += bullets([f"{t}. {a}" for ok, t, a in checks if not ok]) or []

    doc.build(s, onFirstPage=page_number, onLaterPages=page_number)
    return buf.getvalue()