import streamlit as st
import joblib
import pandas as pd

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Schurn analytics | Prediction Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS INJECTION
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg-base:      #05050A;
    --bg-card:      #0D0D14;
    --bg-card-2:    #11111C;
    --border:       #1E1E2E;
    --border-hi:    #2A2A40;
    --accent:       #00D4FF;
    --accent-dim:   rgba(0, 212, 255, 0.12);
    --accent-glow:  rgba(0, 212, 255, 0.25);
    --danger:       #FF4D6D;
    --danger-dim:   rgba(255, 77, 109, 0.12);
    --success:      #00E5A0;
    --success-dim:  rgba(0, 229, 160, 0.12);
    --warning:      #FFB800;
    --text-primary: #E8E8F0;
    --text-sub:     #7A7A9A;
    --text-muted:   #44445A;
    --font-display: 'Syne', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
    --font-body:    'DM Sans', sans-serif;
}

/* ── Base reset ── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background-color: var(--bg-base) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
.stDeployButton { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 2px; }

/* ── Top nav bar ── */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 0 24px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 36px;
}
.nav-logo {
    font-family: var(--font-display);
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--text-primary);
}
.nav-logo span { color: var(--accent); }
.nav-pill {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 500;
    color: var(--accent);
    background: var(--accent-dim);
    border: 1px solid rgba(0,212,255,0.2);
    padding: 5px 12px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* ── Section label ── */
.section-label {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 500;
    color: var(--text-muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 14px;
}

/* ── Card ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 28px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.card:hover { border-color: var(--border-hi); }

.card-title {
    font-family: var(--font-display);
    font-size: 13px;
    font-weight: 600;
    color: var(--text-sub);
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 20px;
}

/* ── Input labels ── */
.stSlider label, .stNumberInput label,
.stSelectbox label, .stRadio label {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    color: var(--text-sub) !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

/* ── Slider track ── */
.stSlider [data-baseweb="slider"] {
    padding: 0 !important;
}
.stSlider [data-testid="stSlider"] > div > div > div {
    background: var(--accent) !important;
}
[data-testid="stSliderThumb"] {
    background: var(--accent) !important;
    border: 2px solid var(--bg-base) !important;
    width: 18px !important;
    height: 18px !important;
    box-shadow: 0 0 12px var(--accent-glow) !important;
}
.stSlider [data-baseweb="slider"] div[role="progressbar"] {
    background: var(--border) !important;
}

/* ── Number input ── */
.stNumberInput input {
    background: var(--bg-card-2) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 0 !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 16px !important;
    padding: 10px 14px !important;
}
.stNumberInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
    outline: none !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--bg-card-2) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 0 !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 13px !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: none !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--font-mono) !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    padding: 16px 24px !important;
    border-radius: 0 !important;
    transition: all 0.2s !important;
    margin-top: 8px !important;
}
.stButton > button:hover {
    background: var(--accent-dim) !important;
    box-shadow: 0 0 20px var(--accent-glow) !important;
    transform: none !important;
}
.stButton > button:active {
    background: rgba(0,212,255,0.2) !important;
}

/* ── KPI cards ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    margin-bottom: 28px;
}
.kpi-cell {
    background: var(--bg-card);
    padding: 22px 24px;
}
.kpi-label {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 400;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: var(--font-display);
    font-size: 28px;
    font-weight: 800;
    line-height: 1;
}
.kpi-value.accent  { color: var(--accent); }
.kpi-value.danger  { color: var(--danger); }
.kpi-value.success { color: var(--success); }

/* ── Probability gauge bar ── */
.gauge-wrap { margin: 8px 0 20px 0; }
.gauge-track {
    height: 6px;
    background: var(--border-hi);
    position: relative;
    overflow: hidden;
}
.gauge-fill {
    height: 100%;
    transition: width 0.6s ease;
}
.gauge-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 6px;
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-muted);
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 6px 14px;
    border: 1px solid;
}
.risk-high   { color: var(--danger);  background: var(--danger-dim);  border-color: var(--danger); }
.risk-medium { color: var(--warning); background: rgba(255,184,0,0.1); border-color: var(--warning); }
.risk-low    { color: var(--success); background: var(--success-dim);  border-color: var(--success); }

/* ── Result headline ── */
.result-headline {
    font-family: var(--font-display);
    font-size: 26px;
    font-weight: 800;
    line-height: 1.2;
    margin: 16px 0 6px 0;
}

/* ── Recommendation box ── */
.rec-box {
    background: var(--bg-card-2);
    border-left: 3px solid var(--accent);
    padding: 14px 18px;
    margin-top: 18px;
}
.rec-title {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.rec-text {
    font-family: var(--font-body);
    font-size: 13px;
    color: var(--text-sub);
    line-height: 1.6;
}

/* ── Debug table ── */
.stDataFrame {
    border: 1px solid var(--border) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 28px 0 !important;
}

/* ── Reduce Streamlit default padding ── */
.block-container {
    padding-top: 28px !important;
    padding-bottom: 48px !important;
    max-width: 1100px !important;
}

/* ── Column gap ── */
[data-testid="column"] { padding: 0 10px !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD ASSETS
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load(r"D:\Churn-Analytics\models\churn_model.pkl")
    cols  = joblib.load(r"D:\Churn-Analytics\models\model_columns.pkl")
    return model, cols

try:
    model, model_columns = load_model()
    model_status = True
except Exception as e:
    model_status = False
    model_error  = str(e)


# ─────────────────────────────────────────────
#  NAV BAR
# ─────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">Churn<span>Sight</span></div>
    <div class="nav-pill">⚡ Prediction Engine v1.0</div>
</div>
""", unsafe_allow_html=True)

if not model_status:
    st.markdown(f"""
    <div class="card" style="border-color:var(--danger)">
        <div class="rec-title" style="color:var(--danger)">⚠ Model Load Error</div>
        <div class="rec-text">{model_error}</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
#  KPI SUMMARY ROW
# ─────────────────────────────────────────────
st.markdown("""
<div class="kpi-row">
    <div class="kpi-cell">
        <div class="kpi-label">Model Status</div>
        <div class="kpi-value accent">LIVE</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-label">Features Active</div>
        <div class="kpi-value accent">2</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-label">Algorithm</div>
        <div class="kpi-value" style="font-size:16px;color:var(--text-primary);margin-top:6px">XGBoost Classifier</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN LAYOUT — 2 columns
# ─────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")


# ────────── LEFT — Inputs ──────────
with col_input:
    st.markdown('<div class="section-label">01 — Customer Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Input Parameters</div>', unsafe_allow_html=True)

    tenure = st.slider(
        "Tenure (Months)",
        min_value=0, max_value=72, value=12, step=1,
        help="How long the customer has been with the company"
    )

    monthly_charge = st.number_input(
        "Monthly Charges (₹)",
        min_value=0.0, max_value=20000.0,
        value=50.0, step=0.5,
        help="Average monthly billing amount"
    )

    contract_type = st.selectbox(
        "Contract Type",
        options=["Month-to-month", "One year", "Two year"],
        index=0
    )

    internet_service = st.selectbox(
        "Internet Service",
        options=["DSL", "Fiber optic", "No"],
        index=0
    )

    st.markdown('</div>', unsafe_allow_html=True)  # close card

    st.markdown("")  # spacer
    predict_clicked = st.button("⚡  RUN PREDICTION")


# ────────── RIGHT — Results ──────────
with col_result:
    st.markdown('<div class="section-label">02 — Prediction Output</div>', unsafe_allow_html=True)

    if not predict_clicked:
        st.markdown("""
        <div class="card" style="min-height:340px;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:12px;">
            <div style="font-family:var(--font-mono);font-size:40px;color:var(--border-hi)">◈</div>
            <div style="font-family:var(--font-mono);font-size:11px;color:var(--text-muted);letter-spacing:2px;text-transform:uppercase;">
                Awaiting input parameters
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        try:
            # ── Build input dataframe ──
            full_input = pd.DataFrame(columns=model_columns)
            full_input.loc[0] = 0
            # XGBoost requires numeric dtypes — convert all object columns to float
            full_input = full_input.astype(float)

            if "Tenure Months" in full_input.columns:
                full_input["Tenure Months"] = tenure
            if "Monthly Charges" in full_input.columns:
                full_input["Monthly Charges"] = monthly_charge

            # One-hot encoded columns (common patterns)
            contract_map = {
                "Month-to-month": "Contract_Month-to-month",
                "One year":        "Contract_One year",
                "Two year":        "Contract_Two year",
            }
            internet_map = {
                "DSL":         "InternetService_DSL",
                "Fiber optic": "InternetService_Fiber optic",
                "No":          "InternetService_No",
            }
            col_c = contract_map.get(contract_type, "")
            col_i = internet_map.get(internet_service, "")
            if col_c and col_c in full_input.columns:
                full_input[col_c] = 1
            if col_i and col_i in full_input.columns:
                full_input[col_i] = 1

            # ── Predict ──
            prediction   = model.predict(full_input)[0]
            probability  = model.predict_proba(full_input)[0][1]
            prob_pct     = probability * 100

            # ── Risk tier ──
            if prob_pct >= 65:
                risk_class = "risk-high"
                risk_label = "HIGH RISK"
                bar_color  = "var(--danger)"
                headline   = "Customer likely to churn"
                rec_text   = (
                    "Immediate intervention recommended. Offer a personalised "
                    "retention package — a contract upgrade discount or service "
                    "bundle within the next 48 hours can reduce churn likelihood "
                    "by up to 40%."
                )
            elif prob_pct >= 35:
                risk_class = "risk-medium"
                risk_label = "MEDIUM RISK"
                bar_color  = "var(--warning)"
                headline   = "Customer shows warning signs"
                rec_text   = (
                    "Monitor this segment closely. A proactive check-in email "
                    "or a loyalty reward touchpoint in the next 7 days is advised "
                    "before risk escalates."
                )
            else:
                risk_class = "risk-low"
                risk_label = "LOW RISK"
                bar_color  = "var(--success)"
                headline   = "Customer likely to stay"
                rec_text   = (
                    "Customer shows stable engagement patterns. Focus on "
                    "upselling or cross-selling premium features to maximise "
                    "lifetime value from this segment."
                )

            # ── Render result card via components.html ──
            # st.markdown unsafe_allow_html is unreliable for dynamic content
            # inside conditional blocks. st.components.v1.html() always works.

            # Map risk tier to actual hex values (CSS vars unavailable in iframe)
            COLOR_MAP = {
                "var(--success)": "#00E5A0",
                "var(--danger)":  "#FF4D6D",
                "var(--warning)": "#FFB800",
            }
            badge_bg_map = {
                "risk-high":   "rgba(255,77,109,0.12)",
                "risk-medium": "rgba(255,184,0,0.10)",
                "risk-low":    "rgba(0,229,160,0.12)",
            }
            color     = COLOR_MAP[bar_color]
            badge_bg  = badge_bg_map[risk_class]

            result_html = f"""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;800&family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:transparent; font-family:'DM Sans',sans-serif; }}
.card {{
  background:#0D0D14;
  border:1px solid #1E1E2E;
  padding:28px;
  position:relative;
  overflow:hidden;
}}
.card::before {{
  content:'';
  position:absolute;
  top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#00D4FF,transparent);
}}
.card-title {{
  font-family:'JetBrains Mono',monospace;
  font-size:12px;font-weight:600;
  color:#7A7A9A;letter-spacing:1px;
  text-transform:uppercase;margin-bottom:20px;
}}
.prob-row {{
  display:flex;align-items:center;
  justify-content:space-between;margin-bottom:4px;
}}
.prob-label {{
  font-family:'JetBrains Mono',monospace;
  font-size:11px;color:#44445A;
}}
.badge {{
  font-family:'JetBrains Mono',monospace;
  font-size:10px;font-weight:700;
  letter-spacing:2px;text-transform:uppercase;
  padding:6px 14px;border:1px solid {color};
  color:{color};background:{badge_bg};
}}
.prob-value {{
  font-family:'Syne',sans-serif;
  font-size:52px;font-weight:800;
  color:{color};line-height:1;margin:10px 0;
}}
.prob-value span {{ font-size:24px;opacity:0.6; }}
.gauge-track {{
  height:6px;background:#2A2A40;
  position:relative;overflow:hidden;margin-top:4px;
}}
.gauge-fill {{
  height:100%;width:{prob_pct:.2f}%;
  background:{color};
  box-shadow:0 0 8px {color};
}}
.gauge-labels {{
  display:flex;justify-content:space-between;
  margin-top:6px;
  font-family:'JetBrains Mono',monospace;
  font-size:10px;color:#44445A;
}}
.headline {{
  font-family:'Syne',sans-serif;
  font-size:22px;font-weight:800;
  color:{color};margin:16px 0 6px 0;
  line-height:1.2;
}}
.rec-box {{
  background:#11111C;
  border-left:3px solid #00D4FF;
  padding:14px 18px;margin-top:18px;
}}
.rec-title {{
  font-family:'JetBrains Mono',monospace;
  font-size:10px;color:#00D4FF;
  letter-spacing:2px;text-transform:uppercase;
  margin-bottom:6px;
}}
.rec-text {{
  font-family:'DM Sans',sans-serif;
  font-size:13px;color:#7A7A9A;line-height:1.6;
}}
</style>
</head>
<body>
<div class="card">
  <div class="card-title">Analysis Result</div>
  <div class="prob-row">
    <div class="prob-label">CHURN PROBABILITY</div>
    <div class="badge">{risk_label}</div>
  </div>
  <div class="prob-value">{prob_pct:.1f}<span>%</span></div>
  <div class="gauge-track">
    <div class="gauge-fill"></div>
  </div>
  <div class="gauge-labels"><span>0%</span><span>50%</span><span>100%</span></div>
  <div class="headline">{headline}</div>
  <div class="rec-box">
    <div class="rec-title">Recommended Action</div>
    <div class="rec-text">{rec_text}</div>
  </div>
</div>
</body>
</html>
"""
            import streamlit.components.v1 as components
            components.html(result_html, height=380, scrolling=False)

        except Exception as e:
            import streamlit.components.v1 as components
            components.html(f"""
<html><body style="background:#0D0D14;font-family:'JetBrains Mono',monospace;padding:20px;">
<div style="border:1px solid #FF4D6D;padding:20px;">
<div style="color:#FF4D6D;font-size:10px;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Prediction Error</div>
<div style="color:#7A7A9A;font-size:12px;">{e}</div>
</div>
</body></html>
""", height=120)


# ─────────────────────────────────────────────
#  DEBUG TABLE (collapsible)
# ─────────────────────────────────────────────
if predict_clicked:
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("◈  DEBUG — Processed Input Vector"):
        try:
            st.dataframe(full_input, use_container_width=True)
        except Exception:
            st.info("Run a prediction first.")


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-top:60px;padding-top:20px;border-top:1px solid var(--border);
            display:flex;justify-content:space-between;align-items:center;">
    <div style="font-family:var(--font-mono);font-size:10px;color:var(--text-muted);letter-spacing:1.5px;">
        CHURNSIGHT — CUSTOMER RETENTION INTELLIGENCE
    </div>
    <div style="font-family:var(--font-mono);font-size:10px;color:var(--text-muted);letter-spacing:1.5px;">
        MODEL: XGBOOST ◈ STACK: PYTHON / STREAMLIT
    </div>
</div>
""", unsafe_allow_html=True)