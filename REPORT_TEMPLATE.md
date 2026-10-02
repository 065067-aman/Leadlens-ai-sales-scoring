# End Term Project Report — LeadLens AI

## 1. Executive Summary
LeadLens AI is an explainable sales lead-scoring assistant that helps sales teams prioritize inbound and outbound leads. It combines a deterministic machine-learning conversion model with Gemini-generated explanations and next-best-action suggestions.

## 2. Use Case Selection
**Use Case #7 — Sales lead-scoring assistant (App).**

Why selected:
- Clear business problem: sales teams cannot give equal attention to every lead.
- Strong data availability: firmographic and engagement variables are easy to collect.
- Natural AI application: predictive scoring + natural-language explanation.
- Demonstrable workflow: input → validation → model score → tier → action → CRM export.
- Strong fit for sales/marketing decision support.

## 3. Business & Strategic Framing (SWOT)
### Q1. Problem, user and customer
Problem: prioritize leads so sales representatives spend time on leads with stronger conversion signals.
End users: SDRs, sales executives, sales managers.
Potential paying customer: B2B companies using CRM/sales teams.

### Q2. Strengths
- Faster prioritization than manual spreadsheet review.
- Consistent scoring logic.
- Explainable signals.
- Batch scoring and CRM-ready export.
- LLM explanation is separated from the numerical decision.

### Q3. Weaknesses / unreliable output
The model is trained on synthetic data, so its score is a prototype rather than a production-grade probability. Gemini may occasionally produce an imperfect recommendation. Example: it may over-emphasize a high deal value even when engagement is weak. The app therefore shows the numerical score and input evidence separately.

### Q4. Opportunities
Scale to multiple industries, languages, CRM integrations, real-time scoring, model retraining, and sales forecasting.

### Q5. Threats
CRM vendors may add similar AI features; API pricing/limits can change; customer behavior can drift; privacy regulations can restrict use of personal data.

### Q6. Competitors
Salesforce Einstein / Agentforce and HubSpot's AI sales capabilities are examples of CRM platforms with AI-assisted sales workflows. LeadLens AI differs as a lightweight, transparent prototype where the scoring logic is visible and customizable.

### Q7. Monetization / adoption
A realistic path would be free pilot → paid team plan → CRM integration → enterprise deployment. Pricing could be based on number of users or leads scored.

## 4. AI / Technical Understanding
### Q1. Model/API
The numerical model is Logistic Regression because it is fast, interpretable and suitable for binary conversion prediction. Gemini 3.5 Flash-Lite is used for low-cost natural-language explanations and next-best-action generation.

### Q2. Prompt/system design
Gemini receives only lead facts, the model score and tier. It is instructed not to change the score, not to invent facts, and to produce a fixed output structure.

### Q3. Out-of-scope
The app does not ask Gemini to make unrelated decisions. If asked for something outside the sales-lead task, the intended behavior is to avoid using it as a general assistant.

### Q4. Privacy
The demonstration uses synthetic data. In a production system, PII should be minimized and governed. The Gemini key is stored as a secret rather than in source code.

### Q5. Failure mode
If Gemini is unavailable, the deterministic score still works and the app displays a fallback message. This prevents total application failure.

### Q6. Data validation
The app validates that uploaded CSVs contain the required feature columns before scoring.

## 5. Critical Thinking / Honest Capability Assessment
### Q1. Wrong or misleading answer
A plausible failure is a confident AI-generated sales message that assumes an unprovided business need. The guardrail explicitly prohibits inventing company facts, budget or intent.

### Q2. What not to trust unsupervised
Do not use the system to automatically reject leads, commit pricing, or make high-value customer decisions without human review.

### Q3. Accountability
A production deployment needs organizational governance: the company deploying the system remains responsible for how the output is used; model/API providers are responsible for their service under their terms.

### Q4. Biggest AI limitation
LLMs can generate fluent text without guaranteeing factual correctness. Therefore the LLM is used for explanation/action drafting, not for the core numerical score.

## 6. Execution
### Edge case tested
Uploaded files with missing columns are rejected with a clear error rather than being sent to the model. Another edge case is API failure: scoring continues without Gemini.

## 7. App-Specific Workflow
### Q1. Input-to-output
User enters lead attributes or uploads CRM CSV → input validation → preprocessing → logistic regression → 0–100 score → HOT/WARM/COLD → Gemini explanation → CRM-ready export.

### Q2. Input validation
The batch workflow checks for all required columns before scoring. Numeric fields use bounded controls in the single-lead form.

### Q3. Catching disagreements
Users can inspect the raw lead attributes and score drivers. The score is not hidden behind a generated narrative.

### Q4. Explainability
The app shows score, tier, evidence and AI rationale rather than only a recommendation.

### Q5. Session/state
The current prototype is stateless across browser refreshes. In production, authenticated CRM/database persistence would be added.

### Q6. Scaling to 100 users/day
Add caching, database-backed model artifacts, API rate-limit handling, request logging, authentication and asynchronous/queued Gemini calls.

### Q7. Similar inputs
Small score changes can occur when a categorical or engagement variable changes. Stability should be tested using controlled perturbation tests before production deployment.

## 8. Conclusion
LeadLens AI demonstrates a complete AI-enabled decision-support workflow while deliberately separating predictive scoring from generative explanation. This makes the prototype easier to audit, test and improve.
