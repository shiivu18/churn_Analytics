import streamlit as st
import joblib
import pandas as pd
import streamlit.components.v1 as components

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight | Prediction Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg-base:      #05050A;
    --bg-card:      #0D0D14;
    --bg-card-2:    #11111C;
    --border:       #1E1E2E;
    --border-hi:    #2A2A40;
    --accent:       #00D4FF;
    --accent-dim:   rgba(0,212,255,0.12);
    --accent-glow:  rgba(0,212,255,0.25);
    --danger:       #FF4D6D;
    --danger-dim:   rgba(255,77,109,0.12);
    --success:      #00E5A0;
    --success-dim:  rgba(0,229,160,0.12);
    --warning:      #FFB800;
    --text-primary: #E8E8F0;
    --text-sub:     #7A7A9A;
    --text-muted:   #44445A;
    --font-display: 'Syne', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
    --font-body:    'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}
.stApp { background-color: var(--bg-base) !important; }
#MainMenu, footer, header, .stDeployButton { display: none !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 2px; }

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

/* ── Slider ── */
[data-testid="stSliderThumb"] {
    background: var(--accent) !important;
    border: 2px solid var(--bg-base) !important;
    box-shadow: 0 0 12px var(--accent-glow) !important;
}

/* ── Number input ── */
.stNumberInput input {
    background: #11111C !important;
    border: 1px solid #2A2A40 !important;
    border-radius: 0 !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 15px !important;
    padding: 10px 14px !important;
}
.stNumberInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #11111C !important;
    border: 1px solid #2A2A40 !important;
    border-radius: 0 !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 13px !important;
}

/* ── Button ── */
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
}

/* ── Layout ── */
.block-container {
    padding-top: 0px !important;
    padding-bottom: 48px !important;
    max-width: 1200px !important;
}
[data-testid="column"] { padding: 0 10px !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }

hr {
    border: none !important;
    border-top: 1px solid #1E1E2E !important;
    margin: 28px 0 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    color: var(--text-muted) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
}
.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid var(--border) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SHARED INLINE STYLE BLOCK FOR components.html
# ─────────────────────────────────────────────
SHARED_STYLES = """
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;800&family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:transparent; font-family:'DM Sans',sans-serif; color:#E8E8F0; }
</style>
"""


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    mdl  = joblib.load(r"D:\Churn-Analytics\models\churn_model.pkl")
    cols = joblib.load(r"D:\Churn-Analytics\models\model_columns.pkl")
    return mdl, cols

try:
    model, model_columns = load_model()
    model_status = True
except Exception as e:
    model_status = False
    model_error  = str(e)


# ─────────────────────────────────────────────
#  NAV BAR
# ─────────────────────────────────────────────
components.html(f"""
{SHARED_STYLES}
<div style="
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:18px 0 20px 0;
    border-bottom:1px solid #1E1E2E;
    margin-bottom:4px;
">
    <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
                letter-spacing:-0.5px;color:#E8E8F0;">
        Churn<span style="color:#00D4FF;">Sight</span>
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:500;
                color:#00D4FF;background:rgba(0,212,255,0.12);
                border:1px solid rgba(0,212,255,0.2);
                padding:5px 12px;letter-spacing:1.5px;text-transform:uppercase;">
        ⚡ Prediction Engine v1.0
    </div>
</div>
""", height=70)


# ─────────────────────────────────────────────
#  MODEL ERROR STATE
# ─────────────────────────────────────────────
if not model_status:
    components.html(f"""
{SHARED_STYLES}
<div style="background:#0D0D14;border:1px solid #FF4D6D;padding:28px;position:relative;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#FF4D6D;
                letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">
        ⚠ Model Load Error
    </div>
    <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:#7A7A9A;">
        {model_error}
    </div>
</div>
""", height=120)
    st.stop()


# ─────────────────────────────────────────────
#  KPI ROW
# ─────────────────────────────────────────────
components.html(f"""
{SHARED_STYLES}
<div style="display:grid;grid-template-columns:repeat(3,1fr);
            gap:1px;background:#1E1E2E;
            border:1px solid #1E1E2E;margin-bottom:4px;">
    <div style="background:#0D0D14;padding:22px 24px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                    font-weight:400;color:#44445A;letter-spacing:2px;
                    text-transform:uppercase;margin-bottom:8px;">Model Status</div>
        <div style="font-family:'Syne',sans-serif;font-size:28px;
                    font-weight:800;color:#00D4FF;line-height:1;">LIVE</div>
    </div>
    <div style="background:#0D0D14;padding:22px 24px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                    font-weight:400;color:#44445A;letter-spacing:2px;
                    text-transform:uppercase;margin-bottom:8px;">Features Active</div>
        <div style="font-family:'Syne',sans-serif;font-size:28px;
                    font-weight:800;color:#00D4FF;line-height:1;">
            {len(model_columns)}
        </div>
    </div>
    <div style="background:#0D0D14;padding:22px 24px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                    font-weight:400;color:#44445A;letter-spacing:2px;
                    text-transform:uppercase;margin-bottom:8px;">Algorithm</div>
        <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:800;
                    color:#E8E8F0;line-height:1;margin-top:6px;">
            XGBoost Classifier
        </div>
    </div>
</div>
""", height=100)


# ─────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")


# ────────── LEFT — Input Panel ──────────
with col_input:

    # Section label
    components.html(f"""
{SHARED_STYLES}
<div style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:500;
            color:#44445A;letter-spacing:3px;text-transform:uppercase;
            margin-bottom:6px;margin-top:8px;">
    01 — Customer Profile
</div>
""", height=32)

    # Card top border accent
    components.html(f"""
{SHARED_STYLES}
<div style="background:#0D0D14;border:1px solid #1E1E2E;
            border-bottom:none;padding:20px 28px 0 28px;position:relative;">
    <div style="position:absolute;top:0;left:0;right:0;height:2px;
                background:linear-gradient(90deg,#00D4FF,transparent);"></div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;
                color:#7A7A9A;letter-spacing:0.5px;text-transform:uppercase;">
        Input Parameters
    </div>
</div>
""", height=58)

    # ── Streamlit Inputs inside a styled wrapper ──
    with st.container():
        st.markdown("""
        <div style="background:#0D0D14;border:1px solid #1E1E2E;
                    border-top:none;padding:20px 28px 28px 28px;">
        </div>
        """, unsafe_allow_html=True)

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

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡  RUN PREDICTION")


# ────────── RIGHT — Results Panel ──────────
with col_result:

    # Section label
    components.html(f"""
{SHARED_STYLES}
<div style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:500;
            color:#44445A;letter-spacing:3px;text-transform:uppercase;
            margin-bottom:6px;margin-top:8px;">
    02 — Prediction Output
</div>
""", height=32)

    # ── Idle state ──
    if not predict_clicked:
        components.html(f"""
{SHARED_STYLES}
<div style="background:#0D0D14;border:1px solid #1E1E2E;
            min-height:380px;display:flex;align-items:center;
            justify-content:center;flex-direction:column;gap:14px;
            position:relative;">
    <div style="position:absolute;top:0;left:0;right:0;height:2px;
                background:linear-gradient(90deg,#00D4FF,transparent);"></div>
    <div style="font-size:42px;color:#2A2A40;font-family:'JetBrains Mono',monospace;">◈</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                color:#44445A;letter-spacing:2px;text-transform:uppercase;">
        Awaiting input parameters
    </div>
</div>
""", height=400)

    # ── Prediction state ──
    else:
        try:
            # Build input dataframe
            full_input = pd.DataFrame(columns=model_columns)
            full_input.loc[0] = 0
            full_input = full_input.astype(float)

            if "Tenure Months" in full_input.columns:
                full_input["Tenure Months"] = float(tenure)
            if "Monthly Charges" in full_input.columns:
                full_input["Monthly Charges"] = float(monthly_charge)

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
                full_input[col_c] = 1.0
            if col_i and col_i in full_input.columns:
                full_input[col_i] = 1.0

            # Predict
            prediction  = model.predict(full_input)[0]
            probability = model.predict_proba(full_input)[0][1]
            prob_pct    = probability * 100

            # Risk tier
            if prob_pct >= 65:
                color      = "#FF4D6D"
                badge_bg   = "rgba(255,77,109,0.12)"
                risk_label = "HIGH RISK"
                headline   = "Customer likely to churn"
                rec_text   = (
                    "Immediate intervention recommended. Offer a personalised "
                    "retention package — a contract upgrade discount or service "
                    "bundle within the next 48 hours can reduce churn likelihood "
                    "by up to 40%."
                )
            elif prob_pct >= 35:
                color      = "#FFB800"
                badge_bg   = "rgba(255,184,0,0.10)"
                risk_label = "MEDIUM RISK"
                headline   = "Customer shows warning signs"
                rec_text   = (
                    "Monitor this segment closely. A proactive check-in email "
                    "or a loyalty reward touchpoint in the next 7 days is advised "
                    "before risk escalates."
                )
            else:
                color      = "#00E5A0"
                badge_bg   = "rgba(0,229,160,0.12)"
                risk_label = "LOW RISK"
                headline   = "Customer likely to stay"
                rec_text   = (
                    "Customer shows stable engagement patterns. Focus on "
                    "upselling or cross-selling premium features to maximise "
                    "lifetime value from this segment."
                )

            components.html(f"""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;800&family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:transparent; font-family:'DM Sans',sans-serif; color:#E8E8F0; }}

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
    font-size:12px;font-weight:600;color:#7A7A9A;
    letter-spacing:1px;text-transform:uppercase;
    margin-bottom:20px;
}}
.top-row {{
    display:flex;align-items:center;
    justify-content:space-between;
    margin-bottom:4px;
}}
.prob-label {{
    font-family:'JetBrains Mono',monospace;
    font-size:11px;color:#44445A;letter-spacing:1px;
    text-transform:uppercase;
}}
.badge {{
    font-family:'JetBrains Mono',monospace;
    font-size:10px;font-weight:700;
    letter-spacing:2px;text-transform:uppercase;
    padding:6px 14px;
    border:1px solid {color};
    color:{color};
    background:{badge_bg};
}}
.prob-value {{
    font-family:'Syne',sans-serif;
    font-size:56px;font-weight:800;
    color:{color};line-height:1;
    margin:10px 0 8px 0;
}}
.prob-value span {{
    font-size:26px;opacity:0.6;
}}
.gauge-track {{
    height:6px;background:#2A2A40;
    position:relative;overflow:hidden;
}}
.gauge-fill {{
    height:100%;
    width:{prob_pct:.2f}%;
    background:{color};
    box-shadow:0 0 10px {color};
    transition:width 0.6s ease;
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
    color:{color};
    margin:18px 0 6px 0;
    line-height:1.2;
}}
.divider {{
    height:1px;background:#1E1E2E;
    margin:18px 0;
}}
.rec-box {{
    background:#11111C;
    border-left:3px solid #00D4FF;
    padding:14px 18px;
}}
.rec-title {{
    font-family:'JetBrains Mono',monospace;
    font-size:10px;color:#00D4FF;
    letter-spacing:2px;text-transform:uppercase;
    margin-bottom:8px;
}}
.rec-text {{
    font-family:'DM Sans',sans-serif;
    font-size:13px;color:#7A7A9A;line-height:1.6;
}}
.meta-row {{
    display:flex;gap:24px;margin-top:18px;
}}
.meta-cell {{
    flex:1;
    background:#0A0A12;
    border:1px solid #1E1E2E;
    padding:12px 16px;
}}
.meta-label {{
    font-family:'JetBrains Mono',monospace;
    font-size:9px;color:#44445A;
    letter-spacing:2px;text-transform:uppercase;
    margin-bottom:4px;
}}
.meta-value {{
    font-family:'JetBrains Mono',monospace;
    font-size:13px;font-weight:700;color:#E8E8F0;
}}
</style>
</head>
<body>
<div class="card">
    <div class="card-title">Analysis Result</div>

    <div class="top-row">
        <div class="prob-label">Churn Probability</div>
        <div class="badge">{risk_label}</div>
    </div>

    <div class="prob-value">
        {prob_pct:.1f}<span>%</span>
    </div>

    <div class="gauge-track">
        <div class="gauge-fill"></div>
    </div>
    <div class="gauge-labels">
        <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
    </div>

    <div class="headline">{headline}</div>

    <div class="divider"></div>

    <div class="rec-box">
        <div class="rec-title">⟶ Recommended Action</div>
        <div class="rec-text">{rec_text}</div>
    </div>

    <div class="meta-row">
        <div class="meta-cell">
            <div class="meta-label">Tenure</div>
            <div class="meta-value">{tenure} mo</div>
        </div>
        <div class="meta-cell">
            <div class="meta-label">Monthly Charges</div>
            <div class="meta-value">₹ {monthly_charge:.2f}</div>
        </div>
        <div class="meta-cell">
            <div class="meta-label">Contract</div>
            <div class="meta-value" style="font-size:11px;">{contract_type}</div>
        </div>
        <div class="meta-cell">
            <div class="meta-label">Internet</div>
            <div class="meta-value" style="font-size:11px;">{internet_service}</div>
        </div>
    </div>
</div>
</body>
</html>
""", height=480, scrolling=False)

        except Exception as e:
            components.html(f"""
{SHARED_STYLES}
<div style="background:#0D0D14;border:1px solid #FF4D6D;padding:24px;position:relative;">
    <div style="position:absolute;top:0;left:0;right:0;height:2px;background:#FF4D6D;"></div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#FF4D6D;
                letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">
        ⚠ Prediction Error
    </div>
    <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:#7A7A9A;">
        {str(e)}
    </div>
</div>
""", height=120)


# ─────────────────────────────────────────────
#  DEBUG TABLE
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
components.html(f"""
{SHARED_STYLES}
<div style="margin-top:40px;padding-top:20px;
            border-top:1px solid #1E1E2E;
            display:flex;justify-content:space-between;align-items:center;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                color:#44445A;letter-spacing:1.5px;">
        CHURNSIGHT — CUSTOMER RETENTION INTELLIGENCE
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                color:#44445A;letter-spacing:1.5px;">
        MODEL: XGBOOST ◈ STACK: PYTHON / STREAMLIT
    </div>
</div>
""", height=60)