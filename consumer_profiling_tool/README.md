# Theory-Enhanced Schema-Flexible Consumer Profiling Tool

## What this app does

This local Streamlit MVP turns arbitrary customer, account, or contact CSV files into theory-informed consumer profiling outputs. It detects field types, maps fields to business roles, assesses profile coverage, calculates dynamic scores, interprets existing segments or generates clusters, and exports scored data plus a Markdown report.

## What makes it schema-flexible

The app does not require fixed fields such as `Recency`, `Monetary`, or `Segment_Label`. It separates raw data type from business meaning, attaches confidence and reasons to each mapping, and lets the user override role and polarity before analysis.

## Theoretical frameworks included

- Four-pillar consumer profiling: demographic/socioeconomic, geographic/environmental, psychographic/motivational, behavioural/digital/transactional.
- B2C, B2B, and mixed B2B/B2C profile-mode detection.
- RFM and lifecycle analysis when value, frequency, and recency fields exist.
- Funnel analysis from exposure, engagement, intent, conversion, and retention signals.
- Negative persona and ROI/effectiveness diagnostics where data allows.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running locally

```bash
streamlit run app.py
```

If Python is installed at `E:\python`, run:

```powershell
E:\python\python.exe -m streamlit run app.py
```

## Uploading a dataset

Upload any CSV with customer, contact, account, transaction, engagement, campaign, or survey-like fields. The app performs a data quality and privacy scan before profiling.

## Field mapping review

The mapping table shows:

- inferred type
- suggested role
- role confidence
- polarity
- polarity confidence
- sensitive-field flag
- proxy-inference flag
- explanation reasons

Users can override role and polarity before running analysis.

## Profile coverage matrix

The app reports which profile dimensions are available, missing, or proxy-only:

- demographic and socioeconomic
- geographic and environmental
- psychographic and motivational
- behavioural, digital, and transactional
- B2B firmographic and decision-role

## Supported analyses

The analysis planner dynamically supports, skips, or marks as proxy:

- RFM and lifecycle analysis
- segment ranking or automatic clustering
- engagement and funnel analysis
- demographic/geographic profiling
- psychographic analysis or behavioural proxy warning
- B2B ICP analysis
- response modelling
- ROI and marketing-effectiveness diagnostics
- negative persona analysis

## Exporting reports

The Export tab provides:

- scored customer CSV
- field mapping JSON
- profile coverage JSON
- analysis summary JSON
- Markdown report

## Privacy and limitations

Likely PII or sensitive fields are flagged. Reports avoid exposing raw identifiers by default. This tool is for local analysis and does not provide legal advice. Recommendations are suggestive, not causal, and must be reviewed before operational use.

## Future development

- PDF export
- persistent project history
- multi-file identity resolution
- LLM-assisted mapping and insight narration
- SHAP and uplift modelling
- role-specific dashboards for retail, travel, SaaS, education, healthcare, and subscriptions

