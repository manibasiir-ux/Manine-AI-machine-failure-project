import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
import os


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Manine AI",
    page_icon="🤖",
    layout="wide")

# =========================
# LOAD MODELS
# =========================


dt_model = joblib.load('decisiontree_machine_failure_model.pkl')

rf_model = joblib.load('random_forest_explainer.pkl')

# =========================
# TITLE
# =========================

st.title("🤖 Manine AI")
st.subheader("Industrial Predictive Maintenance Platform")

st.markdown("""
AI-powered machine failure prediction system with explainable AI analysis.
""")

# =========================
# SIDEBAR
# =========================

st.sidebar.title("Creator")

st.sidebar.markdown("""
### Mani Basir

AI/ML Developer  
Industrial AI Enthusiast  
Robotics & Automation
""")

st.sidebar.markdown(
    "[GitHub](https://github.com/maniasiir-ux)")

st.sidebar.markdown(
    "[LinkedIn](https://linkedin.com/in/mani-basir)")

# =========================
# INPUTS
# =========================

st.header("Machine Sensor Inputs")

col1, col2 = st.columns(2)

with col1:

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=200.0,
        value=50.0)

    vibration = st.number_input(
        "Vibration (Hz)",
        min_value=0.0,
        max_value=200.0,
        value=40.0)

with col2:

    power = st.number_input(
        "Power Usage (kW)",
        min_value=0.0,
        max_value=500.0,
        value=60.0)

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0)

machine_type = st.selectbox(
    "Machine Type",
    ["Drill", "Lathe", "Mill"])

# =========================
# PREDICT BUTTON
# =========================

if st.button("Run AI Prediction"):

    data = pd.DataFrame({
        'Temperature': [temperature],
        'Vibration': [vibration],
        'Power_Usage': [power],
        'Humidity': [humidity],
        'Machine_Type': [machine_type]})

    # =========================
    # PREDICTION
    # =========================

    prediction = dt_model.predict(data)[0]
    prob = dt_model.predict_proba(data)[0]

    safe = prob[0] * 100
    fail = prob[1] * 100

    # =========================
    # STATUS
    # =========================

    st.header("Prediction Result")

    if prediction == 1:
        st.error("⚠️ FAILURE RISK DETECTED")
    else:
        st.success("✅ MACHINE NORMAL")

    # =========================
    # METRICS
    # =========================

    col1, col2, col3 = st.columns(3)

    col1.metric("Safe Probability", f"{safe:.2f}%")
    col2.metric("Failure Probability", f"{fail:.2f}%")

    if fail < 25:
        risk = "LOW"
    elif fail < 50:
        risk = "MEDIUM"
    elif fail < 75:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    col3.metric("Risk Level", risk)

    # =========================
    # PIE CHART
    # =========================

    fig = px.pie(
        values=[safe, fail],
        names=["Safe", "Failure"],
        title="Risk Distribution")

    st.plotly_chart(fig, use_container_width=True)
    
    
    
    # =========================
    # EXPLAINABLE AI ANALYSIS
    # =========================

    classifier = rf_model.named_steps['classifier']
    preprocessor = rf_model.named_steps['preprocessor']

    processed = preprocessor.transform(data)

    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(processed)

    # -------------------------
    # HANDLE SHAP OUTPUT
    # -------------------------

    if isinstance(shap_values, list):
        impacts = shap_values[1][0]
    else:
        if len(shap_values.shape) == 3:
            impacts = shap_values[0, :, 1]
        else:
            impacts = shap_values[0]

    # Convert to 1D
    impacts = impacts.flatten()

    # Feature names
    feature_names = preprocessor.get_feature_names_out()

    # -------------------------
    # FIX LENGTH MISMATCH
    # -------------------------

    min_len = min(len(feature_names), len(impacts))

    feature_names = feature_names[:min_len]
    impacts = impacts[:min_len]

    # -------------------------
    # CREATE DATAFRAME
    # -------------------------

    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": impacts})

    explanation_df["ABS"] = explanation_df["Impact"].abs()

    explanation_df = explanation_df.sort_values(
        by="ABS",
        ascending=False)

    # Show top 5
    top_explanations = explanation_df.head(5)
    
    # =========================
    # SHAP VISUALIZATION
    # =========================

    st.header("Explainable AI Analysis")

    st.markdown("""
### Understanding SHAP Impacts

- Positive impact (+)  
  Means this feature pushed the prediction TOWARD machine failure.

- Negative impact (-)  
  Means this feature pushed the prediction AWAY from failure.
""")

    # Create colors
    colors = [
        "red" if x > 0 else "green"
        for x in top_explanations["Impact"]
    ]

    # SHAP BAR CHART
    shap_fig = go.Figure()

    shap_fig.add_trace(go.Bar(
        x=top_explanations["Impact"],
        y=top_explanations["Feature"],
        orientation='h',
        marker_color=colors))

    shap_fig.update_layout(
        title="Top Feature Impacts",
        xaxis_title="SHAP Impact",
        yaxis_title="Features",
        height=500)

    st.plotly_chart(shap_fig, use_container_width=True)

    # Human-readable explanations
    st.subheader("AI Explanation Summary")

    for _, row in top_explanations.iterrows():

        feature = row["Feature"]
        impact = row["Impact"]

        if impact > 0:
            st.warning(
                f"{feature} increased machine failure risk "
                f"(Impact: {impact:.4f})")
        else:
            st.success(
                f"{feature} reduced machine failure risk "
                f"(Impact: {impact:.4f})")

   
    # =========================
    # RULE INSIGHTS
    # =========================

    st.header("Rule Insights")

    if vibration > 70:
        st.warning("High vibration detected")

    if temperature > 80:
        st.warning("High temperature detected")

    if power > 80:
        st.warning("High power usage detected")

    if humidity < 20:
        st.warning("Low humidity detected")

    st.success("System analysis complete.")
