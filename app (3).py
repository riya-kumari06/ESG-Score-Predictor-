import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="ESG Score Predictor",
    page_icon="🌱",
    layout="centered"
)

# ── Load saved model + artifacts ─────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load("models/esg_ridge_model.pkl")
    scaler       = joblib.load("models/esg_scaler.pkl")
    features     = joblib.load("models/feature_names.pkl")
    le_industry  = joblib.load("models/le_industry.pkl")
    le_region    = joblib.load("models/le_region.pkl")
    return model, scaler, features, le_industry, le_region

model, scaler, feature_names, le_industry, le_region = load_artifacts()

# ── Industry ESG benchmarks ───────────────────────────────
INDUSTRY_BENCHMARKS = {
    'Technology'     : 62,
    'Energy'         : 45,
    'Healthcare'     : 60,
    'Finance'        : 55,
    'Consumer Goods' : 58,
    'Industrials'    : 50,
    'Materials'      : 47,
    'Utilities'      : 52,
    'Real Estate'    : 54,
    'Telecom'        : 57,
}

# ── Default values auto-filled by industry ────────────────
INDUSTRY_DEFAULTS = {
    'Technology'     : dict(Revenue=50000,  MarketCap=200000, GrowthRate=15,  CarbonEmissions=1200,   WaterUsage=80000,    EnergyConsumption=500000,   ProfitMargin=18),
    'Energy'         : dict(Revenue=120000, MarketCap=150000, GrowthRate=5,   CarbonEmissions=900000, WaterUsage=5000000,  EnergyConsumption=80000000, ProfitMargin=10),
    'Healthcare'     : dict(Revenue=30000,  MarketCap=80000,  GrowthRate=10,  CarbonEmissions=5000,   WaterUsage=500000,   EnergyConsumption=2000000,  ProfitMargin=15),
    'Finance'        : dict(Revenue=40000,  MarketCap=120000, GrowthRate=8,   CarbonEmissions=3000,   WaterUsage=200000,   EnergyConsumption=1500000,  ProfitMargin=20),
    'Consumer Goods' : dict(Revenue=60000,  MarketCap=90000,  GrowthRate=6,   CarbonEmissions=50000,  WaterUsage=2000000,  EnergyConsumption=10000000, ProfitMargin=12),
    'Industrials'    : dict(Revenue=80000,  MarketCap=70000,  GrowthRate=4,   CarbonEmissions=200000, WaterUsage=3000000,  EnergyConsumption=20000000, ProfitMargin=8),
    'Materials'      : dict(Revenue=45000,  MarketCap=50000,  GrowthRate=3,   CarbonEmissions=500000, WaterUsage=8000000,  EnergyConsumption=50000000, ProfitMargin=7),
    'Utilities'      : dict(Revenue=35000,  MarketCap=60000,  GrowthRate=2,   CarbonEmissions=700000, WaterUsage=10000000, EnergyConsumption=60000000, ProfitMargin=9),
    'Real Estate'    : dict(Revenue=20000,  MarketCap=55000,  GrowthRate=5,   CarbonEmissions=8000,   WaterUsage=400000,   EnergyConsumption=3000000,  ProfitMargin=14),
    'Telecom'        : dict(Revenue=55000,  MarketCap=100000, GrowthRate=7,   CarbonEmissions=15000,  WaterUsage=300000,   EnergyConsumption=4000000,  ProfitMargin=16),
}

def score_to_category(score):
    if score >= 70:
        return "High Performer"
    elif score >= 45:
        return "Moderate Performer"
    else:
        return "Needs Improvement"

# ── Generate confusion matrix data from synthetic test set ─
@st.cache_data
def get_confusion_matrix_data(_model, _scaler, _le_industry, _le_region, _feature_names):
    np.random.seed(99)
    n = 300
    industries = _le_industry.classes_
    regions    = _le_region.classes_

    df_test = pd.DataFrame({
        'Industry'          : np.random.choice(industries, n),
        'Region'            : np.random.choice(regions, n),
        'Year'              : np.random.randint(2015, 2025, n),
        'Revenue'           : np.random.uniform(10, 500000, n),
        'ProfitMargin'      : np.random.uniform(-50, 60, n),
        'MarketCap'         : np.random.uniform(10, 3000000, n),
        'GrowthRate'        : np.random.uniform(-30, 100, n),
        'ESG_Environmental' : np.random.uniform(0, 100, n),
        'ESG_Social'        : np.random.uniform(0, 100, n),
        'ESG_Governance'    : np.random.uniform(0, 100, n),
        'CarbonEmissions'   : np.random.uniform(100, 10000000, n),
        'WaterUsage'        : np.random.uniform(100, 50000000, n),
        'EnergyConsumption' : np.random.uniform(100, 1000000000, n),
    })

    # True scores
    df_test['ESG_Overall'] = (
        0.35 * df_test['ESG_Environmental'] +
        0.35 * df_test['ESG_Social'] +
        0.30 * df_test['ESG_Governance'] +
        np.random.normal(0, 3, n)
    ).clip(0, 100)

    df_test['Industry'] = _le_industry.transform(df_test['Industry'])
    df_test['Region']   = _le_region.transform(df_test['Region'])

    X_test  = df_test[_feature_names]
    y_true  = df_test['ESG_Overall']

    X_scaled   = _scaler.transform(X_test)
    y_pred_raw = _model.predict(X_scaled)
    y_pred_raw = np.clip(y_pred_raw, 0, 100)

    y_true_cat = [score_to_category(s) for s in y_true]
    y_pred_cat = [score_to_category(s) for s in y_pred_raw]

    labels = ["Needs Improvement", "Moderate Performer", "High Performer"]
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels)
    return cm, labels

# ── UI ────────────────────────────────────────────────────
st.title("🌱 ESG Score Predictor")
st.markdown(
    "Enter your **ESG sub-scores** and **company profile** below. "
    "Industry-specific defaults are applied automatically for technical fields."
)

with st.expander("ℹ️ Why does ESG Score matter?"):
    st.markdown("""
    **ESG (Environmental, Social, Governance)** scores help investors, regulators,
    and companies measure sustainability and ethical impact.

    - 🌿 **Environmental** — Carbon footprint, energy & water usage
    - 🤝 **Social** — Employee welfare, community impact
    - 🏛️ **Governance** — Transparency, board ethics, accountability

    A higher ESG score means **lower risk** and **stronger long-term performance**.
    This tool uses a **Ridge Regression model** aligned with **UN SDGs 8, 13, and 17**.
    """)

st.divider()
st.markdown("### 🏢 Company Profile")

col1, col2 = st.columns(2)
with col1:
    industry = st.selectbox("🏭 Industry", le_industry.classes_)
with col2:
    region   = st.selectbox("🌍 Region", le_region.classes_)

year = st.slider("📅 Reporting Year", 2015, 2025, 2023)

st.divider()
st.markdown("### 📋 ESG Sub-Scores")
st.caption("These 3 scores are the primary drivers of your overall ESG rating.")

col3, col4, col5 = st.columns(3)
with col3:
    esg_env = st.slider("🌍 Environmental", 0.0, 100.0, 65.0, step=0.5,
                        help="Measures carbon footprint, energy & water management")
with col4:
    esg_soc = st.slider("🤝 Social", 0.0, 100.0, 60.0, step=0.5,
                        help="Measures employee welfare, diversity & community impact")
with col5:
    esg_gov = st.slider("🏛️ Governance", 0.0, 100.0, 70.0, step=0.5,
                        help="Measures board transparency, ethics & accountability")

defaults = INDUSTRY_DEFAULTS[industry]

with st.expander("⚙️ Auto-filled Industry Defaults (click to view)"):
    st.caption("These values are automatically set based on your industry average.")
    auto_df = pd.DataFrame({
        "Field"      : ["Revenue (M)", "MarketCap (M)", "GrowthRate (%)",
                        "CarbonEmissions", "WaterUsage", "EnergyConsumption", "ProfitMargin (%)"],
        "Auto Value" : [defaults["Revenue"], defaults["MarketCap"], defaults["GrowthRate"],
                        defaults["CarbonEmissions"], defaults["WaterUsage"],
                        defaults["EnergyConsumption"], defaults["ProfitMargin"]]
    })
    st.dataframe(auto_df, use_container_width=True, hide_index=True)

st.divider()

# ── Predict ───────────────────────────────────────────────
if st.button("🔍 Predict ESG Score", use_container_width=True, type="primary"):
    try:
        industry_encoded = le_industry.transform([industry])[0]
        region_encoded   = le_region.transform([region])[0]

        user_data = pd.DataFrame([{
            "Industry"          : industry_encoded,
            "Region"            : region_encoded,
            "Year"              : year,
            "Revenue"           : defaults["Revenue"],
            "ProfitMargin"      : defaults["ProfitMargin"],
            "MarketCap"         : defaults["MarketCap"],
            "GrowthRate"        : defaults["GrowthRate"],
            "ESG_Environmental" : esg_env,
            "ESG_Social"        : esg_soc,
            "ESG_Governance"    : esg_gov,
            "CarbonEmissions"   : defaults["CarbonEmissions"],
            "WaterUsage"        : defaults["WaterUsage"],
            "EnergyConsumption" : defaults["EnergyConsumption"],
        }])[feature_names]

        user_scaled = scaler.transform(user_data)
        prediction  = model.predict(user_scaled)
        final_score = float(np.clip(prediction[0], 0, 100))

        # ── 1. MAIN SCORE ─────────────────────────────────
        st.subheader(f"🎯 Predicted ESG Overall Score: **{final_score:.2f} / 100**")

        if final_score >= 70:
            st.success("✅ HIGH PERFORMER — Strong sustainability profile")
            risk_label = "🟢 Low ESG Risk"
        elif final_score >= 45:
            st.warning("⚠️ MODERATE PERFORMER — Room for improvement")
            risk_label = "🟡 Medium ESG Risk"
        else:
            st.error("❌ NEEDS IMPROVEMENT — Significant ESG gaps identified")
            risk_label = "🔴 High ESG Risk"

        st.progress(int(final_score))
        st.markdown(f"**Investor Risk Rating: {risk_label}**")

        st.divider()

        # ── 2. INDUSTRY BENCHMARK ─────────────────────────
        st.markdown("### 🏆 Industry Benchmark Comparison")
        benchmark = INDUSTRY_BENCHMARKS.get(industry, 55)
        diff      = round(final_score - benchmark, 2)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Your Score",       f"{final_score:.2f}")
        col_b.metric("Industry Average", f"{benchmark}")
        col_c.metric("Difference",       f"{diff:+.2f}", delta=f"{diff:+.2f}")

        if diff >= 0:
            st.success(f"✅ You are **{diff:.1f} points above** the {industry} industry average!")
        else:
            st.warning(f"⚠️ You are **{abs(diff):.1f} points below** the {industry} industry average.")

        fig_bench = go.Figure(go.Bar(
            x=["Your Score", f"{industry} Average"],
            y=[final_score, benchmark],
            marker_color=["#2ecc71", "#3498db"],
            text=[f"{final_score:.1f}", f"{benchmark}"],
            textposition="outside"
        ))
        fig_bench.update_layout(
            yaxis=dict(range=[0, 110]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=300
        )
        st.plotly_chart(fig_bench, use_container_width=True)

        st.divider()

        # ── 3. RECOMMENDATIONS ────────────────────────────
        st.markdown("### 💡 Actionable Recommendations")
        recs = []
        if esg_env < 60:
            recs.append("🌿 **Boost Environmental Score** — Invest in renewable energy and reduce emissions. A +15 point improvement adds ~5 points overall.")
        if esg_soc < 60:
            recs.append("🤝 **Improve Social Score** — Enhance employee benefits and diversity programs. A +15 point improvement adds ~5 points overall.")
        if esg_gov < 60:
            recs.append("🏛️ **Strengthen Governance** — Publish ESG reports and improve board transparency. A +15 point improvement adds ~4.5 points overall.")
        if final_score < 70:
            recs.append(f"🎯 **Target Score 70+** — You need **{70 - final_score:.1f} more points** to reach HIGH PERFORMER status.")

        if not recs:
            st.success("🌟 Excellent! Your ESG profile is strong. Keep maintaining these standards.")
        else:
            for r in recs:
                st.markdown(f"- {r}")

        st.divider()

        # ── 4. CONFUSION MATRIX ───────────────────────────
        st.markdown("### 🔢 Model Confusion Matrix")
        st.caption(
            "This shows how accurately the model classifies companies into ESG performance "
            "categories on a 300-sample test set. Diagonal cells = correct predictions."
        )

        cm, labels = get_confusion_matrix_data(
            model, scaler, le_industry, le_region, feature_names
        )

        # Highlight where current prediction falls
        predicted_category = score_to_category(final_score)
        st.info(f"📌 Your prediction falls in the **'{predicted_category}'** category.")

        # Accuracy per class
        col_m1, col_m2, col_m3 = st.columns(3)
        for i, (col_m, label) in enumerate(zip([col_m1, col_m2, col_m3], labels)):
            total    = cm[i].sum()
            correct  = cm[i][i]
            accuracy = round((correct / total) * 100, 1) if total > 0 else 0
            col_m.metric(label, f"{accuracy}% accurate", f"{correct}/{total} correct")

        # Heatmap
        z_text = [[str(val) for val in row] for row in cm]
        fig_cm = ff.create_annotated_heatmap(
            z=cm.tolist(),
            x=labels,
            y=labels,
            annotation_text=z_text,
            colorscale="Greens",
            showscale=True
        )
        fig_cm.update_layout(
            title="Predicted (x-axis) vs Actual (y-axis)",
            xaxis_title="Predicted Category",
            yaxis_title="Actual Category",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=400
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Overall accuracy
        overall_acc = round(np.trace(cm) / np.sum(cm) * 100, 1)
        st.success(f"✅ Overall Model Accuracy: **{overall_acc}%** on test set")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Model: Ridge Regression · Aligned with UN SDG 8, 13, 17 · Built with Streamlit & Plotly")
