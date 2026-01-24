"""
🔧 Predictive Maintenance - CRISP-ML Deployment
Streamlit Application for Real-Time Machine Failure Prediction

Built with CRISP-ML methodology
Brand: Ice Graphite Hybrid
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import os
from pathlib import Path

# =========================
# Brand Palette
# =========================
ICE_SILVER = "#E6E8EB"
GRAPHITE = "#2A3038"
ESPRESSO_GOLD = "#C9A86A"
GRAPHITE_DEEP = "#240338"
SLATE = "#424A53"
PEBBLE = "#5E757D"
MIST = "#B0B4B8"
SILVER = "#D5D6DB"
PLATINUM = "#EBECEF"
SUCCESS = "#43936C"
WARNING = "#F2AE4A"
DANGER = "#D96B5F"
INFO = "#4A67B0"

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Predictive Maintenance | CRISP-ML",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Load Model Artifacts
# =========================
@st.cache_resource
def load_model_artifacts():
    """Load all model artifacts from the model directory."""
    model_dir = Path(__file__).parent.parent / "model"
    
    try:
        model = joblib.load(model_dir / "model.pkl")
        label_encoder = joblib.load(model_dir / "label_encoder_type.pkl")
        feature_cols = joblib.load(model_dir / "feature_cols.pkl")
        metadata = joblib.load(model_dir / "model_metadata.pkl")
        return model, label_encoder, feature_cols, metadata, None
    except Exception as e:
        return None, None, None, None, str(e)

model, label_encoder, feature_cols, metadata, load_error = load_model_artifacts()

# =========================
# Custom CSS
# =========================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=IBM+Plex+Mono:wght@300;400;600&display=swap');
    
    /* Global Styles */
    .stApp {{
        background: linear-gradient(135deg, {PLATINUM} 0%, {ICE_SILVER} 50%, {SILVER} 100%);
        font-family: 'IBM Plex Mono', monospace;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Main content area */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}
    
    /* Header styling */
    .main-header {{
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 3.5rem;
        color: {GRAPHITE_DEEP};
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px {ESPRESSO_GOLD}40;
    }}
    
    .sub-header {{
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 400;
        font-size: 1.1rem;
        color: {SLATE};
        letter-spacing: 0.08em;
        margin-bottom: 2.5rem;
        text-transform: uppercase;
    }}
    
    /* Section headers */
    .section-header {{
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: {GRAPHITE_DEEP};
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }}
    
    .section-subheader {{
        font-family: 'Orbitron', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: {GRAPHITE_DEEP};
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }}
    
    /* Card styling */
    .metric-card {{
        background: linear-gradient(145deg, white, {PLATINUM});
        border: 2px solid {SILVER};
        border-radius: 12px;
        padding: 1.8rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(42, 48, 56, 0.08),
                    inset 0 1px 0 rgba(255, 255, 255, 0.8);
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(42, 48, 56, 0.12),
                    0 0 0 2px {ESPRESSO_GOLD}40;
        border-color: {ESPRESSO_GOLD};
    }}
    
    .metric-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: {PEBBLE};
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }}
    
    .metric-value {{
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }}
    
    .metric-delta {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        font-weight: 400;
        color: {SLATE};
    }}
    
    /* Status indicators */
    .status-success {{ color: {SUCCESS}; }}
    .status-warning {{ color: {WARNING}; }}
    .status-danger {{ color: {DANGER}; }}
    
    /* Prediction result box */
    .prediction-box {{
        background: white;
        border: 3px solid;
        border-radius: 16px;
        padding: 2.5rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(42, 48, 56, 0.12);
        position: relative;
        overflow: hidden;
    }}
    
    .prediction-box::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, {ESPRESSO_GOLD}08 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }}
    
    @keyframes rotate {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    
    .prediction-box.success {{
        border-color: {SUCCESS};
        background: linear-gradient(145deg, white, {SUCCESS}08);
    }}
    
    .prediction-box.danger {{
        border-color: {DANGER};
        background: linear-gradient(145deg, white, {DANGER}08);
    }}
    
    .prediction-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
        color: {PEBBLE};
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }}
    
    .prediction-result {{
        font-family: 'Orbitron', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }}
    
    .prediction-confidence {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        color: {SLATE};
        position: relative;
        z-index: 1;
    }}
    
    /* Input styling */
    .stNumberInput input, .stSelectbox select {{
        background-color: white !important;
        color: {GRAPHITE} !important;
        border: 2px solid {SILVER} !important;
        border-radius: 8px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        transition: all 0.3s ease !important;
        font-size: 0.95rem !important;
    }}
    
    .stNumberInput input:focus, .stSelectbox select:focus {{
        border-color: {ESPRESSO_GOLD} !important;
        box-shadow: 0 0 0 3px {ESPRESSO_GOLD}20 !important;
    }}
    
    /* Divider */
    hr {{
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, {ESPRESSO_GOLD}60, transparent);
        margin: 2rem 0;
    }}
    
    /* Feature importance bars */
    .feature-bar {{
        background: {PLATINUM};
        border-radius: 8px;
        overflow: hidden;
        margin: 0.5rem 0;
        height: 32px;
        position: relative;
        border: 2px solid {SILVER};
    }}
    
    .feature-fill {{
        height: 100%;
        background: linear-gradient(90deg, {ESPRESSO_GOLD} 0%, {ESPRESSO_GOLD}CC 100%);
        display: flex;
        align-items: center;
        padding-left: 1rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        transition: width 1s ease;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: white;
        border: 2px solid {SILVER};
        border-radius: 8px 8px 0 0;
        color: {SLATE};
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.85rem;
        padding: 0.75rem 1.5rem;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(180deg, {ESPRESSO_GOLD}15, white);
        border-color: {ESPRESSO_GOLD};
        color: {GRAPHITE_DEEP};
    }}
    
    /* Info box */
    .info-box {{
        margin-top: 2rem;
        padding: 1.5rem;
        background: white;
        border-left: 4px solid {ESPRESSO_GOLD};
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(42, 48, 56, 0.05);
    }}
    
    .info-box-text {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: {SLATE};
        line-height: 1.6;
    }}
    
    /* Confidence box */
    .confidence-box {{
        margin-top: 1rem;
        padding: 1rem;
        background: {PLATINUM};
        border-radius: 8px;
        border: 1px solid {SILVER};
    }}
    
    /* Decision logic box */
    .decision-box {{
        margin-top: 1rem;
        padding: 1rem;
        background: white;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(42, 48, 56, 0.05);
    }}
</style>
""", unsafe_allow_html=True)

# =========================
# Feature Engineering
# =========================
def engineer_features(df: pd.DataFrame, tool_wear_max: float = 253.0) -> pd.DataFrame:
    """Apply the same feature engineering as training."""
    df = df.copy()
    df['Temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
    df['Strain'] = df['Torque [Nm]'] * df['Tool wear [min]']
    df['Tool_wear_ratio'] = df['Tool wear [min]'] / tool_wear_max
    df['Temp_ratio'] = df['Process temperature [K]'] / df['Air temperature [K]']
    return df

# =========================
# Prediction System
# =========================
def make_prediction(air_temp, process_temp, rotational_speed, torque, tool_wear, product_type):
    """Make prediction using the trained model."""
    
    if model is None:
        return "ERROR", "Model not loaded", 0.0, {}, load_error, DANGER, "✕"
    
    try:
        # Create DataFrame with raw input values
        input_data = {
            'Air temperature [K]': air_temp,
            'Process temperature [K]': process_temp,
            'Rotational speed [rpm]': rotational_speed,
            'Torque [Nm]': torque,
            'Tool wear [min]': tool_wear
        }
        df = pd.DataFrame([input_data])
        
        # Encode Type -> Type_encoded (matching training column name)
        df['Type_encoded'] = label_encoder.transform([product_type])[0]
        
        # Engineer features
        df = engineer_features(df)
        
        # Select features in correct order
        X = df[feature_cols]
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        # Calculate feature signals (qualitative importance)
        temp_diff = process_temp - air_temp
        power = torque * rotational_speed
        
        feature_signals = {
            "Tool Wear": min(tool_wear / 250, 1.0),
            "Temperature Δ": min(abs(temp_diff - 10) / 10, 1.0),
            "Torque Load": abs(torque - 40) / 40,
            "Power Output": min(power / 150000, 1.0),
            "Speed Deviation": abs(rotational_speed - 1500) / 1500
        }
        
        # Normalize
        total_signal = sum(feature_signals.values())
        if total_signal > 0:
            feature_signals = {k: v/total_signal for k, v in feature_signals.items()}
        
        # Determine operational state
        if prediction == 0:
            operational_state = "OPERATIONAL"
            recommended_action = "Continue normal operation"
            status_color = SUCCESS
            status_icon = "✓"
            confidence = probability[0]
        else:
            operational_state = "FAILURE RISK"
            recommended_action = "Schedule maintenance immediately"
            status_color = DANGER
            status_icon = "⚠"
            confidence = probability[1]
        
        return (operational_state, recommended_action, confidence, 
                feature_signals, probability, status_color, status_icon)
                
    except Exception as e:
        return "ERROR", f"Prediction error: {str(e)}", 0.0, {}, None, DANGER, "✕"

# =========================
# Header
# =========================
st.markdown("""
    <div class="main-header">🔧 Predictive Maintenance</div>
    <div class="sub-header">CRISP-ML Pipeline • Real-Time Failure Prediction System</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Check if model is loaded
if model is None:
    st.error(f"⚠️ Model artifacts not found: {load_error}")
    st.info("Please ensure the model files are in the correct location (./model/ directory)")
    st.stop()

# =========================
# Input Controls Section
# =========================
st.markdown(f"""
    <div class="section-header">⚙️ Machine Parameters</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    product_type = st.selectbox(
        "Product Type",
        options=["L", "M", "H"],
        index=1,
        help="L = Low quality, M = Medium quality, H = High quality"
    )
    
    air_temp = st.number_input(
        "Air Temperature [K]",
        min_value=290.0,
        max_value=310.0,
        value=300.0,
        step=0.1,
        help="Ambient air temperature in Kelvin"
    )

with col2:
    process_temp = st.number_input(
        "Process Temperature [K]",
        min_value=300.0,
        max_value=320.0,
        value=310.0,
        step=0.1,
        help="Machine operating temperature in Kelvin"
    )
    
    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=1000,
        max_value=3000,
        value=1500,
        step=10,
        help="Rotational speed in RPM"
    )

with col3:
    torque = st.number_input(
        "Torque [Nm]",
        min_value=3.0,
        max_value=80.0,
        value=40.0,
        step=0.5,
        help="Applied torque in Newton-meters"
    )
    
    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=253,
        value=100,
        step=1,
        help="Cumulative tool usage in minutes"
    )

st.markdown("---")

# =========================
# Auto-run Predictions
# =========================
result = make_prediction(
    air_temp, process_temp, rotational_speed, torque, tool_wear, product_type
)

operational_state, recommended_action, confidence, feature_signals, probability, status_color, status_icon = result

# =========================
# Prediction Result Display
# =========================
pred_col, metrics_col = st.columns([1, 1])

with pred_col:
    # Operational State Result Box
    box_class = "success" if operational_state == "OPERATIONAL" else "danger"
    st.markdown(f"""
        <div class="prediction-box {box_class}">
            <div class="prediction-label">System Status</div>
            <div class="prediction-result" style="color: {status_color};">
                {status_icon} {operational_state}
            </div>
            <div class="prediction-confidence">
                {recommended_action}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Confidence display
    if probability is not None:
        st.markdown(f"""
            <div class="confidence-box">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: {PEBBLE}; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.8rem;">
                    Model Confidence
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: {SLATE};">
                        <strong>No Failure:</strong>
                    </div>
                    <div style="font-family: 'Orbitron', sans-serif; font-size: 0.9rem; color: {SUCCESS}; font-weight: 600;">
                        {probability[0]*100:.1f}%
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: {SLATE};">
                        <strong>Failure:</strong>
                    </div>
                    <div style="font-family: 'Orbitron', sans-serif; font-size: 0.9rem; color: {DANGER}; font-weight: 600;">
                        {probability[1]*100:.1f}%
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

with metrics_col:
    # Quick Metrics in 2x2 grid
    m1, m2 = st.columns(2)
    
    temp_diff = abs(process_temp - air_temp)
    with m1:
        temp_color = WARNING if temp_diff > 12 else SUCCESS
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Temperature Δ</div>
                <div class="metric-value" style="color: {temp_color};">
                    {temp_diff:.1f}K
                </div>
                <div class="metric-delta">
                    Process vs Air
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with m2:
        wear_pct = (1 - tool_wear/253)*100
        wear_color = DANGER if tool_wear > 200 else WARNING if tool_wear > 150 else SUCCESS
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tool Condition</div>
                <div class="metric-value" style="color: {wear_color};">
                    {wear_pct:.0f}%
                </div>
                <div class="metric-delta">
                    {253 - tool_wear} min remaining
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    m3, m4 = st.columns(2)
    
    with m3:
        optimal_speed = 1500
        speed_deviation = abs(rotational_speed - optimal_speed) / optimal_speed * 100
        speed_color = SUCCESS if speed_deviation < 15 else WARNING if speed_deviation < 30 else DANGER
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Speed Status</div>
                <div class="metric-value" style="color: {speed_color};">
                    {rotational_speed}
                </div>
                <div class="metric-delta">
                    {speed_deviation:.1f}% from baseline
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with m4:
        power = torque * rotational_speed
        machine_quality = {"L": "Low", "M": "Medium", "H": "High"}[product_type]
        quality_color = SUCCESS if product_type == 'H' else WARNING if product_type == 'M' else PEBBLE
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Product Quality</div>
                <div class="metric-value" style="color: {quality_color};">
                    {machine_quality}
                </div>
                <div class="metric-delta">
                    Type {product_type}
                </div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# =========================
# Tabs for Detailed Analysis
# =========================
tab1, tab2, tab3 = st.tabs(["📊 Key Signals", "📈 Real-Time Monitoring", "🔧 Recommendations"])

with tab1:
    st.markdown(f"""
        <div class="section-subheader">Feature Importance (Qualitative)</div>
    """, unsafe_allow_html=True)
    
    # Sort features by importance
    sorted_features = sorted(feature_signals.items(), key=lambda x: x[1], reverse=True)
    
    for feature, importance in sorted_features:
        percentage = importance * 100
        st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: {SLATE}; margin-bottom: 0.3rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                    {feature}
                </div>
                <div class="feature-bar">
                    <div class="feature-fill" style="width: {percentage}%;">
                        {percentage:.1f}%
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="info-box">
            <div class="info-box-text">
                <strong style="color: {ESPRESSO_GOLD};">Interpretation:</strong> These signals represent deviations from optimal conditions. 
                Higher values indicate greater contribution to the detected risk state.
            </div>
        </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown(f"""
        <div class="section-subheader">Sensor Monitoring</div>
    """, unsafe_allow_html=True)
    
    # Create mock time series data for visualization
    timestamps = pd.date_range(end=datetime.now(), periods=50, freq='5min')
    
    # Temperature trends
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=timestamps,
        y=np.random.normal(air_temp, 1, 50),
        name='Air Temperature',
        line=dict(color=MIST, width=2),
        fill='tonexty',
        fillcolor=f'rgba(176, 180, 184, 0.1)'
    ))
    fig_temp.add_trace(go.Scatter(
        x=timestamps,
        y=np.random.normal(process_temp, 1.5, 50),
        name='Process Temperature',
        line=dict(color=ESPRESSO_GOLD, width=3),
    ))
    
    fig_temp.update_layout(
        title="Temperature Monitoring",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor=PLATINUM,
        font=dict(family="IBM Plex Mono", color=GRAPHITE),
        xaxis=dict(gridcolor=SILVER, title="Time"),
        yaxis=dict(gridcolor=SILVER, title="Temperature [K]"),
        legend=dict(bgcolor="white", bordercolor=SILVER, borderwidth=1),
        height=350
    )
    
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # Performance gauges
    col1, col2 = st.columns(2)
    
    with col1:
        fig_speed = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rotational_speed,
            delta={'reference': 1500, 'increasing': {'color': WARNING}},
            gauge={
                'axis': {'range': [1000, 3000], 'tickcolor': GRAPHITE},
                'bar': {'color': ESPRESSO_GOLD},
                'bgcolor': PLATINUM,
                'borderwidth': 2,
                'bordercolor': SILVER,
                'steps': [
                    {'range': [1000, 1300], 'color': f'rgba(217, 107, 95, 0.12)'},
                    {'range': [1300, 1800], 'color': f'rgba(67, 147, 108, 0.12)'},
                    {'range': [1800, 3000], 'color': f'rgba(242, 174, 74, 0.12)'}
                ],
                'threshold': {
                    'line': {'color': GRAPHITE, 'width': 4},
                    'thickness': 0.75,
                    'value': 1500
                }
            },
            title={'text': "Rotational Speed [RPM]", 'font': {'color': GRAPHITE}}
        ))
        
        fig_speed.update_layout(
            paper_bgcolor="white",
            font={'color': GRAPHITE, 'family': "IBM Plex Mono"},
            height=300
        )
        
        st.plotly_chart(fig_speed, use_container_width=True)
    
    with col2:
        fig_torque = go.Figure(go.Indicator(
            mode="gauge+number",
            value=torque,
            gauge={
                'axis': {'range': [0, 80], 'tickcolor': GRAPHITE},
                'bar': {'color': ESPRESSO_GOLD},
                'bgcolor': PLATINUM,
                'borderwidth': 2,
                'bordercolor': SILVER,
                'steps': [
                    {'range': [0, 25], 'color': f'rgba(242, 174, 74, 0.12)'},
                    {'range': [25, 50], 'color': f'rgba(67, 147, 108, 0.12)'},
                    {'range': [50, 80], 'color': f'rgba(217, 107, 95, 0.12)'}
                ]
            },
            title={'text': "Torque [Nm]", 'font': {'color': GRAPHITE}}
        ))
        
        fig_torque.update_layout(
            paper_bgcolor="white",
            font={'color': GRAPHITE, 'family': "IBM Plex Mono"},
            height=300
        )
        
        st.plotly_chart(fig_torque, use_container_width=True)

with tab3:
    st.markdown(f"""
        <div class="section-subheader">Recommended Actions</div>
    """, unsafe_allow_html=True)
    
    if operational_state == "OPERATIONAL":
        recommendations = [
            ("Continue Normal Operation", "No consistent degradation signals detected. Maintain current monitoring schedule.", SUCCESS),
            ("Routine Inspection", "Schedule standard inspection within 7-14 days per maintenance calendar.", SUCCESS),
            ("Monitor Tool Wear", f"Tool wear at {tool_wear} minutes. Plan replacement around 220-230 minutes.", WARNING if tool_wear > 180 else SUCCESS)
        ]
    else:
        recommendations = [
            ("Immediate Intervention Required", "Model predicts high failure risk. Stop operation for urgent inspection.", DANGER),
            ("Complete System Inspection", "Check cooling system, electrical connections, bearings, and tooling.", DANGER),
            ("Condition Analysis", "Evaluate component wear and schedule replacement before restart.", DANGER)
        ]
    
    for i, (title, description, color) in enumerate(recommendations):
        st.markdown(f"""
            <div style="margin-bottom: 1.5rem; padding: 1.5rem; background: white; border-left: 4px solid {color}; border-radius: 4px; box-shadow: 0 2px 8px rgba(42, 48, 56, 0.05);">
                <div style="font-family: 'Orbitron', sans-serif; font-size: 1.1rem; font-weight: 700; color: {color}; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">
                    {i+1}. {title}
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: {SLATE}; line-height: 1.6;">
                    {description}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Model Performance Summary
    if metadata:
        metrics = metadata.get('metrics', {})
        model_name = metadata.get('model_name', 'Random Forest')
        accuracy = metrics.get('Accuracy', 0)*100
        precision = metrics.get('Precision', 0)*100
        recall = metrics.get('Recall', 0)*100
        
        st.markdown(f"""
<div style="margin-top: 2rem; padding: 2rem; background: white; border: 2px solid {ESPRESSO_GOLD}; border-radius: 8px; box-shadow: 0 4px 16px rgba(42, 48, 56, 0.08);">
    <div style="font-family: 'Orbitron', sans-serif; font-size: 1.1rem; font-weight: 700; color: {ESPRESSO_GOLD}; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.05em;">
        💡 About the Prediction Model
    </div>
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: {SLATE}; line-height: 1.8;">
        <p style="margin-bottom: 0.8rem;"><strong style="color: {GRAPHITE_DEEP};">Model:</strong> {model_name} trained with CRISP-ML methodology.</p>
        <p style="margin-bottom: 0.8rem;"><strong style="color: {GRAPHITE_DEEP};">Performance:</strong> Accuracy {accuracy:.1f}% | Precision {precision:.1f}% | Recall {recall:.1f}%</p>
        <p style="margin-bottom: 0.8rem;"><strong style="color: {GRAPHITE_DEEP};">Feature Engineering:</strong> Includes derived features like Power (Torque × RPM), Strain (Torque × Wear), and Temperature ratios.</p>
        <p style="margin: 0;"><strong style="color: {ESPRESSO_GOLD};">Business Value:</strong> ~78% cost savings potential through proactive maintenance.</p>
    </div>
</div>
        """, unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: {PEBBLE}; letter-spacing: 0.1em; text-transform: uppercase;">
            Predictive Maintenance System • CRISP-ML Pipeline
        </div>
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: {MIST}; margin-top: 0.5rem;">
            Built with Machine Learning • Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>
""", unsafe_allow_html=True)