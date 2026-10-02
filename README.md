# LeadLens AI — Sales Lead Scoring Assistant

End-term AI application prototype for Use Case #7: Sales lead-scoring assistant.

## Features
- Synthetic B2B CRM dataset
- Logistic Regression lead-conversion model
- 0–100 score with HOT/WARM/COLD tiers
- Single-lead scoring form
- Batch CSV scoring and CRM export
- AI-generated explanation and next-best-action using Gemini 3.5 Flash-Lite
- Guardrails and deterministic fallback
- Model transparency and limitation section

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-key"
```

Never commit the API key.

## Deployment
Deploy the repository on Streamlit Community Cloud and add the same key under App Settings → Secrets.
