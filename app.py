import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

st.set_page_config(page_title='LeadLens AI', page_icon='🎯', layout='wide')

st.markdown('''
<style>
.block-container {padding-top: 1.7rem; max-width: 1450px;}
.hero {padding: 1.35rem 1.5rem; border:1px solid rgba(148,163,184,.18); border-radius:18px; background:linear-gradient(135deg,rgba(30,41,59,.8),rgba(15,23,42,.95)); margin-bottom:1.2rem;}
.hero-title {font-size:2.25rem;font-weight:800;letter-spacing:-.03em;}
.hero-subtitle {color:#94a3b8;font-size:1rem;margin-top:.2rem;}
.kpi {padding:1rem 1.1rem;border:1px solid rgba(148,163,184,.16);border-radius:16px;background:rgba(15,23,42,.6);min-height:105px;}
.kpi-label {color:#94a3b8;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;}
.kpi-value {font-size:1.8rem;font-weight:800;margin-top:.3rem;}
.score-box {padding:1.2rem;border:1px solid rgba(96,165,250,.3);border-radius:18px;background:linear-gradient(135deg,rgba(30,64,175,.2),rgba(15,23,42,.72));}
.score {font-size:3rem;font-weight:850;line-height:1;}
.hot{color:#fb7185;font-weight:800}.warm{color:#fbbf24;font-weight:800}.cold{color:#60a5fa;font-weight:800}
</style>
''', unsafe_allow_html=True)

FEATURES = ['company_size','industry','lead_source','region','website_visits','email_opens','content_downloads','demo_requested','sales_calls','days_since_last_activity','estimated_deal_value']
CAT_COLS = ['company_size','industry','lead_source','region']
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

@st.cache_data
def load_data():
    return pd.read_csv('sample_leads.csv')

@st.cache_resource
def train_model(data):
    X, y = data[FEATURES], data['converted']
    prep = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS),
        ('num', StandardScaler(), NUM_COLS),
    ])
    pipe = Pipeline([('prep', prep), ('clf', LogisticRegression(max_iter=1500, class_weight='balanced'))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:,1]
    pred = (prob >= .5).astype(int)
    return pipe, roc_auc_score(y_test, prob), accuracy_score(y_test, pred)

data = load_data()
model, auc, accuracy = train_model(data)

def tier(score):
    return 'HOT' if score >= 70 else ('WARM' if score >= 40 else 'COLD')

def score_df(df):
    out = df.copy()
    out['score'] = np.round(model.predict_proba(out[FEATURES])[:,1] * 100, 1)
    out['tier'] = out['score'].apply(tier)
    return out.sort_values('score', ascending=False)

def score_one(row):
    return float(model.predict_proba(row[FEATURES])[:,1][0] * 100)

def contribution_table(row):
    prep = model.named_steps['prep']
    clf = model.named_steps['clf']
    transformed = prep.transform(row[FEATURES])
    if hasattr(transformed, 'toarray'):
        transformed = transformed.toarray()
    names = prep.get_feature_names_out()
    contributions = transformed[0] * clf.coef_[0]
    result = pd.DataFrame({'feature': names, 'contribution': contributions})
    result['feature'] = result['feature'].str.replace('cat__','',regex=False).str.replace('num__','',regex=False)
    return result.sort_values('contribution', ascending=False)

def pretty(name):
    mapping = {'website_visits':'Website visits','email_opens':'Email opens','content_downloads':'Content downloads','demo_requested':'Demo requested','sales_calls':'Sales calls','days_since_last_activity':'Days since last activity','estimated_deal_value':'Estimated deal value'}
    for prefix, label in [('company_size_','Company size: '),('industry_','Industry: '),('lead_source_','Lead source: '),('region_','Region: ')]:
        if name.startswith(prefix): return name.replace(prefix,label,1)
    return mapping.get(name,name)

def gemini_client():
    key = None
    try: key = st.secrets.get('GEMINI_API_KEY')
    except Exception: pass
    key = key or os.getenv('GEMINI_API_KEY')
    if not key: return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception: return None

def ai_recommendation(lead, score, tier_name, contrib):
    client = gemini_client()
    if client is None:
        return 'Gemini is not configured. The deterministic scoring engine is still working.'
    evidence = contrib.head(5).to_dict('records')
    prompt = f'''You are a B2B sales enablement copilot. The numerical score was calculated by an ML model. Never change it.
Lead facts: {lead.to_dict()}
Model score: {score:.1f}/100
Tier: {tier_name}
Top model contribution signals: {evidence}
Return exactly these sections:

### 🎯 Lead Assessment
One concise sentence summarizing the lead.

### Why this lead is in this tier
2-3 concise bullets based only on the supplied facts.

### Recommended next action
Give one practical sales action and a suggested time horizon.

### Suggested outreach
Write an actual ready-to-send B2B sales message.
- Start with "Hi" or "Hello".
- Write 2-3 complete sentences.
- Make it sound like a real salesperson speaking to the lead.
- Reference only facts provided in the lead data.
- Do not use placeholders such as [Name], [Company], or [Product].
- Do not invent a person's name, company name, budget, authority, competitors, requirements, or purchase timeline.
- Do not explain what the salesperson should write; write the actual message.
- End with a simple call-to-action or question.

### ⚠️ Risk / caution
Mention one thing the salesperson should verify.

Do not invent facts or claim certainty of conversion.'''
    try:
        response = client.models.generate_content(model='gemini-3.5-flash-lite', contents=prompt)
        return response.text
    except Exception as e:
        return f'AI recommendation unavailable. Numerical scoring remains available. ({type(e).__name__})'

st.sidebar.markdown('## 🎯 LeadLens AI')
st.sidebar.caption('Sales Intelligence & AI Copilot')
page = st.sidebar.radio('Navigate', ['🏠 Executive Dashboard','🎯 Score a Lead','📁 Batch Scoring','🤖 AI Sales Copilot','🧠 Model & Governance'])
st.sidebar.divider()
st.sidebar.metric('ROC-AUC', f'{auc:.2f}')
st.sidebar.metric('Test accuracy', f'{accuracy:.1%}')
st.sidebar.caption('1,000 synthetic B2B leads')

st.markdown('''<div class="hero"><div class="hero-title">🎯 LeadLens AI</div><div class="hero-subtitle">Explainable sales lead scoring + AI-powered next-best-action assistant</div></div>''', unsafe_allow_html=True)

if page == '🏠 Executive Dashboard':
    scored = score_df(data)
    hot = int((scored.tier=='HOT').sum()); warm = int((scored.tier=='WARM').sum()); cold = int((scored.tier=='COLD').sum())
    cols = st.columns(4)
    for col,label,value in zip(cols,['Total leads','Hot leads','Warm leads','Average score'],[f'{len(scored):,}',f'{hot:,}',f'{warm:,}',f'{scored.score.mean():.1f}/100']):
        col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',unsafe_allow_html=True)
    st.subheader('Lead priority distribution')
    counts = scored.tier.value_counts().reindex(['HOT','WARM','COLD']).fillna(0).astype(int)
    st.bar_chart(counts, height=320)
    st.subheader('Priority lead queue')
    cols = ['lead_id','company_size','industry','lead_source','demo_requested','days_since_last_activity','estimated_deal_value','score','tier']
    display = scored[cols].head(15).copy(); display['estimated_deal_value']=display.estimated_deal_value.map(lambda x:f'₹{x:,.0f}')
    st.dataframe(display,width='stretch',hide_index=True)
    st.info('Hybrid design: ML controls the numerical score; Gemini explains the result and recommends a sales action.')

elif page == '🎯 Score a Lead':
    st.subheader('Score a new lead')
    with st.form('lead_form'):
        c1,c2,c3=st.columns(3)
        company_size=c1.selectbox('Company size',['SMB','Mid-Market','Enterprise'])
        industry=c2.selectbox('Industry',sorted(data.industry.unique()))
        source=c3.selectbox('Lead source',sorted(data.lead_source.unique()))
        region=c1.selectbox('Region',sorted(data.region.unique()))
        website=c2.number_input('Website visits',0,50,6)
        opens=c3.number_input('Email opens',0,30,5)
        downloads=c1.number_input('Content downloads',0,15,2)
        demo=c2.selectbox('Demo requested?',['No','Yes'])
        calls=c3.number_input('Sales calls',0,20,2)
        days=c1.number_input('Days since last activity',0,120,5)
        deal=c2.number_input('Estimated deal value (₹)',25000,5000000,350000,step=25000)
        submitted=st.form_submit_button('Calculate Lead Score',type='primary',width='stretch')
    if submitted:
        lead=pd.DataFrame([{'company_size':company_size,'industry':industry,'lead_source':source,'region':region,'website_visits':website,'email_opens':opens,'content_downloads':downloads,'demo_requested':1 if demo=='Yes' else 0,'sales_calls':calls,'days_since_last_activity':days,'estimated_deal_value':deal}])
        score=score_one(lead); tier_name=tier(score); contrib=contribution_table(lead)
        a,b=st.columns([1,2])
        with a:
            st.markdown(f'<div class="score-box"><div style="color:#94a3b8">LEAD SCORE</div><div class="score">{score:.1f}</div><div>Priority: <span class="{tier_name.lower()}">{tier_name}</span></div></div>',unsafe_allow_html=True)
        with b:
            st.progress(int(score)); st.caption('The 0–100 score is a model-derived prioritization signal, not a guarantee of conversion.')
        st.subheader('🔍 Model explanation')
        st.caption('Contributions are in logistic-regression log-odds space: positive values push the prediction upward; negative values push it downward.')
        pos=contrib[contrib.contribution>0].head(5).copy(); neg=contrib[contrib.contribution<0].sort_values('contribution').head(5).copy()
        x,y=st.columns(2)
        with x:
            st.markdown('#### ⬆️ Positive contributors')
            if len(pos):
                pos['Signal']=pos.feature.map(pretty); pos['Contribution']=pos.contribution.map(lambda z:f'+{z:.2f}')
                st.dataframe(pos[['Signal','Contribution']],width='stretch',hide_index=True)
            else: st.info('No positive contributors.')
        with y:
            st.markdown('#### ⬇️ Negative contributors')
            if len(neg):
                neg['Signal']=neg.feature.map(pretty); neg['Contribution']=neg.contribution.map(lambda z:f'{z:.2f}')
                st.dataframe(neg[['Signal','Contribution']],width='stretch',hide_index=True)
            else: st.info('No negative contributors.')
        st.subheader('🤖 AI Sales Recommendation')
        with st.spinner('Generating sales recommendation...'):
            st.markdown(ai_recommendation(lead.iloc[0],score,tier_name,contrib))

elif page == '📁 Batch Scoring':
    st.subheader('Batch score a CRM export')
    st.write('Upload a CSV containing the required lead fields. The converted column is optional.')
    uploaded=st.file_uploader('Upload CSV',type=['csv'])
    if uploaded:
        try:
            raw=pd.read_csv(uploaded); missing=[c for c in FEATURES if c not in raw.columns]
            if missing:
                st.error(f'Missing required columns: {", ".join(missing)}')
            elif raw[FEATURES].isnull().any().any():
                nulls=raw[FEATURES].columns[raw[FEATURES].isnull().any()].tolist(); st.error(f'Missing values detected in: {", ".join(nulls)}')
            else:
                bad=[c for c in NUM_COLS if not pd.api.types.is_numeric_dtype(raw[c])]
                if bad: st.error(f'Numeric validation failed for: {", ".join(bad)}')
                else:
                    scored=score_df(raw); st.success(f'Successfully scored {len(scored):,} leads.')
                    a,b,c=st.columns(3); a.metric('Hot',int((scored.tier=='HOT').sum())); b.metric('Warm',int((scored.tier=='WARM').sum())); c.metric('Cold',int((scored.tier=='COLD').sum()))
                    st.dataframe(scored,width='stretch',hide_index=True)
                    st.download_button('⬇️ Download CRM-ready scored CSV',scored.to_csv(index=False).encode('utf-8'),'scored_leads.csv','text/csv',width='stretch')
        except Exception as e:
            st.error(f'The file could not be processed safely: {type(e).__name__}')
    else:
        st.download_button('Download sample CSV',data.to_csv(index=False).encode('utf-8'),'sample_leads.csv','text/csv',width='stretch')

elif page == '🤖 AI Sales Copilot':
    st.subheader('AI Sales Copilot')
    scored=score_df(data); lead_id=st.selectbox('Choose a demo lead',scored.lead_id.head(50).tolist()); lead=scored[scored.lead_id==lead_id].iloc[0]
    a,b=st.columns(2); a.metric('Lead score',f"{lead.score:.1f}/100"); b.metric('Priority',lead.tier)
    profile={k:v for k,v in [('Company size',lead.company_size),('Industry',lead.industry),('Lead source',lead.lead_source),('Region',lead.region),('Website visits',int(lead.website_visits)),('Email opens',int(lead.email_opens)),('Content downloads',int(lead.content_downloads)),('Demo requested','Yes' if lead.demo_requested else 'No'),('Sales calls',int(lead.sales_calls)),('Days since activity',int(lead.days_since_last_activity)),('Estimated deal value',f'₹{lead.estimated_deal_value:,.0f}')]}
    st.dataframe(pd.DataFrame({'Attribute':list(profile.keys()),'Value':list(profile.values())}),width='stretch',hide_index=True)
    if st.button('Generate AI sales play',type='primary',width='stretch'):
        frame=pd.DataFrame([lead[FEATURES].to_dict()]); contrib=contribution_table(frame)
        with st.spinner('Gemini is preparing the sales play...'): st.markdown(ai_recommendation(lead[FEATURES],float(lead.score),lead.tier,contrib))

else:
    st.subheader('🧠 Model transparency & governance')
    st.markdown('''**Architecture**\n\nCRM / Lead CSV → validation → preprocessing → Logistic Regression → 0–100 score → HOT/WARM/COLD → model contribution explanation → Gemini → next-best-action''')
    st.markdown('''**Why hybrid AI?**\n\n- ML produces the measurable numerical score.\n- Contribution analysis makes the score auditable.\n- Gemini turns evidence into usable sales language.\n- Gemini is instructed not to change the score or invent customer facts.\n- If Gemini is unavailable, the core score still works.''')
    st.info('The demonstration uses synthetic data. Production use would require privacy controls, access control, monitoring, model validation and human oversight.')
    st.warning('Limitations: synthetic data, possible model drift, no guarantee of conversion, imperfect generative output, API availability/rate limits, and need for human review.')
    st.subheader('Model metrics'); a,b=st.columns(2); a.metric('ROC-AUC',f'{auc:.2f}'); b.metric('Test accuracy',f'{accuracy:.1%}')
    st.caption('Metrics use a held-out test split of the synthetic demonstration dataset.')
