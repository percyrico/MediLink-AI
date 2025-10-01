
# MediLink AI — Drug Shortage Tracker (Starter)

This starter lets you run a **Gradio app** locally with sample CSVs and flip to **Google Cloud (BigQuery + Vertex AI)** by setting env vars.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate  # (or .venv\Scripts\activate on Windows)
pip install -r requirements.txt
python app.py
```

## Use with Google Cloud
1. Create a GCP project and enable APIs:
   - BigQuery API
   - Vertex AI API
2. Create a service account and download JSON key. Export:
   ```bash
   export GCP_PROJECT=YOUR_PROJECT_ID
   export GCP_REGION=us-central1
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
   ```
3. Create the dataset/tables:
   ```bash
   bq query --use_legacy_sql=false < bigquery_schema.sql
   ```
4. (Later) Point Fivetran destination to BigQuery dataset `medilink` and start syncing data.

## Tabs
- **Find**: Search for a medicine, filter by city/price/distance.
- **Alerts**: Local demo of shortage alerts (for GCP, compute in scheduled job and store in `medilink.alerts`).
- **Admin**: Upload CSVs to replace local sample data quickly.

## Notes
- If GCP is configured, the app queries BigQuery and uses Vertex AI (Gemini-1.5-flash) for 100-word summaries.
- Otherwise, it stays in **Local Demo Mode** using the CSVs in this folder.
