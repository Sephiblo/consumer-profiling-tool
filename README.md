# Theory-Enhanced Schema-Flexible Consumer Profiling Tool

## What this app does

This Streamlit app turns arbitrary customer, account, or contact CSV files into theory-informed consumer profiling outputs. It detects field types, maps fields to business roles, assesses profile coverage, calculates dynamic scores, interprets existing segments or generates clusters, and exports scored data plus a Markdown report.

## What makes it schema-flexible

The app does not require fixed fields such as `Recency`, `Monetary`, or `Segment_Label`. It separates raw data type from business meaning, attaches confidence and reasons to each mapping, and lets the user override role and polarity before analysis.

## Theoretical frameworks included

- Four-pillar consumer profiling: demographic/socioeconomic, geographic/environmental, psychographic/motivational, behavioural/digital/transactional.
- B2C, B2B, and mixed B2B/B2C profile-mode detection.
- RFM and lifecycle analysis when value, frequency, and recency fields exist.
- Funnel analysis from exposure, engagement, intent, conversion, and retention signals.
- Negative persona and ROI/effectiveness diagnostics where data allows.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

If Python is installed at `E:\python`, run:

```powershell
E:\python\python.exe -m streamlit run app.py
```

## Streamlit Cloud deployment

Use these settings:

- Repository: `Sephiblo/consumer-profiling-tool`
- Branch: `main`
- Main file path: `app.py`

## Exporting reports

The Export tab provides:

- scored customer CSV
- field mapping JSON
- profile coverage JSON
- analysis summary JSON
- Markdown report

## Privacy and limitations

Likely PII or sensitive fields are flagged. Reports avoid exposing raw identifiers by default. This tool is for local analysis and does not provide legal advice. Recommendations are suggestive, not causal, and must be reviewed before operational use.
