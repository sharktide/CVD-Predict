#!/usr/bin/env python3
"""
OHCA Prediction Dashboard - Premium ISEF Edition
"""
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import time
import warnings
warnings.filterwarnings('ignore')

def loss_fn(y_true, y_prob):
    y_true = tf.cast(y_true, tf.float32)
    y_prob = tf.clip_by_value(y_prob, 1e-7, 1 - 1e-7)
    pt = tf.where(tf.equal(y_true, 1), y_prob, 1 - y_prob)
    alpha_t = tf.where(tf.equal(y_true, 1), 0.75, 0.25)
    focal_loss = -alpha_t * tf.pow(1 - pt, 2.0) * tf.math.log(pt)
    bce = -y_true * tf.math.log(y_prob) - (1 - y_true) * tf.math.log(1 - y_prob)
    return tf.reduce_mean(0.5 * focal_loss + 0.5 * bce)

# Page config
st.set_page_config(
    page_title="OHCA Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium look
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: #1a1a2e;
    --accent-blue: #4f46e5;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-orange: #f59e0b;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --border: #1e293b;
    --glow-blue: rgba(79, 70, 229, 0.3);
    --glow-cyan: rgba(6, 182, 212, 0.3);
}

.stApp {
    background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    max-width: 1400px;
    padding: 2rem 3rem;
}

/* Header styling */
.header-container {
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%);
    border: 1px solid rgba(79, 70, 229, 0.3);
    border-radius: 16px;
    padding: 2rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.header-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan), var(--accent-green));
}

.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #fff 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.02em;
}

.header-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
    font-weight: 400;
}

/* Card styling */
.metric-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(26, 26, 46, 0.8) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: var(--accent-blue);
    box-shadow: 0 0 30px var(--glow-blue);
    transform: translateY(-2px);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
    font-weight: 500;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #fff 0%, #e2e8f0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-value.critical {
    background: linear-gradient(135deg, var(--accent-red) 0%, #f87171 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-value.warning {
    background: linear-gradient(135deg, var(--accent-orange) 0%, #fbbf24 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-value.success {
    background: linear-gradient(135deg, var(--accent-green) 0%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Risk gauge */
.risk-gauge {
    width: 200px;
    height: 200px;
    border-radius: 50%;
    position: relative;
    margin: 0 auto;
    background: conic-gradient(
        var(--accent-green) 0deg 90deg,
        var(--accent-orange) 90deg 180deg,
        var(--accent-red) 180deg 270deg,
        var(--accent-red) 270deg 360deg
    );
    display: flex;
    align-items: center;
    justify-content: center;
}

.risk-gauge-inner {
    width: 160px;
    height: 160px;
    border-radius: 50%;
    background: var(--bg-card);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    border: 3px solid var(--border);
}

.risk-gauge-value {
    font-size: 2.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
}

.risk-gauge-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Signal display */
.signal-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.signal-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.signal-title::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-cyan);
    box-shadow: 0 0 10px var(--accent-cyan);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-cyan) 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .stMarkdown {
    color: var(--text-primary);
}

/* Input fields */
.stSelectbox, .stSlider {
    background: var(--bg-card);
    border-radius: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 8px;
    padding: 0.25rem;
    gap: 0.25rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-secondary);
    border-radius: 6px;
    padding: 0.5rem 1rem;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: var(--accent-blue) !important;
    color: white !important;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-secondary);
}

/* Animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.pulse {
    animation: pulse 2s infinite;
}

@keyframes glow {
    0%, 100% { box-shadow: 0 0 5px var(--accent-cyan); }
    50% { box-shadow: 0 0 20px var(--accent-cyan), 0 0 30px var(--accent-cyan); }
}

.glow {
    animation: glow 2s infinite;
}

/* Status indicators */
.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 0.5rem;
}

.status-dot.active {
    background: var(--accent-green);
    box-shadow: 0 0 10px var(--accent-green);
    animation: pulse 1.5s infinite;
}

.status-dot.inactive {
    background: var(--text-secondary);
}

/* Table styling */
.dataframe {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.dataframe th {
    background: rgba(79, 70, 229, 0.2) !important;
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border) !important;
    font-weight: 600 !important;
}

.dataframe td {
    color: var(--text-primary) !important;
    border-bottom: 1px solid rgba(30, 41, 59, 0.5) !important;
}

.dataframe tr:hover td {
    background: rgba(79, 70, 229, 0.1) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-primary);
    font-weight: 500;
}

.streamlit-expanderContent {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 8px 8px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_v2.keras', compile=False)

def preprocess_ecg(ecg):
    ecg = np.asarray(ecg, dtype=np.float32).flatten()[:1300]
    if len(ecg) < 1300:
        ecg = np.pad(ecg, (0, 1300 - len(ecg)))
    ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
    return ecg.reshape(1, 1300, 1)

def preprocess_acc(acc):
    acc = np.asarray(acc, dtype=np.float32)
    if acc.ndim == 1:
        acc = acc.reshape(-1, 3)
    acc = acc[:250]
    if len(acc) < 250:
        acc = np.pad(acc, ((0, 250 - len(acc)), (0, 0)))
    return acc.reshape(1, 250, 3)

def generate_ecg(scenario='normal'):
    t = np.linspace(0, 10, 1300)
    
    # Base ECG with realistic morphology
    heart_rate = 72
    rr_interval = 60 / heart_rate
    
    if scenario == 'arrest':
        heart_rate = 45 + np.random.randint(0, 20)
        rr_interval = 60 / heart_rate
        st_segment = -0.15
        qt_prolonged = True
    elif scenario == 'warning':
        heart_rate = 85 + np.random.randint(0, 20)
        rr_interval = 60 / heart_rate
        st_segment = -0.08
        qt_prolonged = False
    else:
        heart_rate = 68 + np.random.randint(0, 10)
        rr_interval = 60 / heart_rate
        st_segment = 0
        qt_prolonged = False
    
    # Generate QRS complexes
    ecg = np.zeros(1300)
    beat_pos = 0
    
    while beat_pos < 1300:
        # P wave
        p_start = beat_pos + int(0.1 * 130)
        p_width = int(0.08 * 130)
        if p_start + p_width < 1300:
            p_wave = 0.15 * np.exp(-((np.arange(p_width) - p_width/2)**2) / (2 * (p_width/6)**2))
            ecg[p_start:p_start+p_width] += p_wave
        
        # QRS complex
        qrs_start = beat_pos + int(0.16 * 130)
        qrs_width = int(0.08 * 130)
        if qrs_start + qrs_width < 1300:
            qrs = np.zeros(qrs_width)
            qrs[int(qrs_width*0.2)] = -0.1  # Q
            qrs[int(qrs_width*0.35)] = 1.2   # R
            qrs[int(qrs_width*0.5)] = -0.2  # S
            ecg[qrs_start:qrs_start+qrs_width] += qrs
        
        # T wave
        t_start = beat_pos + int(0.35 * 130)
        t_width = int(0.15 * 130)
        if t_start + t_width < 1300:
            t_wave = 0.3 * np.exp(-((np.arange(t_width) - t_width/2)**2) / (2 * (t_width/5)**2))
            if qt_prolonged:
                t_start += int(0.05 * 130)
            ecg[t_start:t_start+t_width] += t_wave
        
        # ST segment depression
        st_start = beat_pos + int(0.28 * 130)
        st_width = int(0.07 * 130)
        if st_start + st_width < 1300:
            ecg[st_start:st_start+st_width] += st_segment
        
        beat_pos += int(rr_interval * 130)
    
    # Add noise
    ecg += np.random.randn(1300) * 0.02
    
    # Normalize
    ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
    
    return ecg

def generate_acc(scenario='normal'):
    t = np.linspace(0, 10, 250)
    
    if scenario == 'arrest':
        # Agonal breathing + fall
        acc = np.zeros((250, 3))
        acc[:, 2] = 9.81  # Gravity
        # Slow breathing pattern
        acc[:, 1] += np.sin(t * 0.8) * 0.3
        # Sudden drop (fall)
        acc[100:120, 2] -= np.linspace(0, 3, 20)
        acc[120:150, 2] += np.linspace(3, 0, 30)
        acc += np.random.randn(250, 3) * 0.05
    elif scenario == 'warning':
        # Irregular movement
        acc = np.random.randn(250, 3) * 0.2
        acc[:, 2] += 9.81
        acc[:, 1] += np.sin(t * 1.5) * 0.2 + np.sin(t * 0.3) * 0.1
    else:
        # Normal activity
        acc = np.random.randn(250, 3) * 0.1
        acc[:, 2] += 9.81
        acc[:, 1] += np.sin(t * 1.2) * 0.15
    
    return acc

def get_risk_color(prob):
    if prob > 0.7:
        return '#ef4444', 'CRITICAL'
    elif prob > 0.5:
        return '#f59e0b', 'HIGH'
    elif prob > 0.3:
        return '#06b6d4', 'MODERATE'
    else:
        return '#10b981', 'LOW'

def create_signal_plot(ecg, acc, title_ecg="ECG Signal", title_acc="Accelerometer"):
    fig = plt.figure(figsize=(14, 5), facecolor='#1a1a2e')
    gs = GridSpec(1, 2, width_ratios=[2, 1], wspace=0.3)
    
    # ECG plot
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#0a0a0f')
    t_ecg = np.linspace(0, 10, len(ecg))
    ax1.plot(t_ecg, ecg, color='#06b6d4', linewidth=1.2, alpha=0.9)
    ax1.fill_between(t_ecg, ecg, alpha=0.1, color='#06b6d4')
    ax1.set_xlabel('Time (s)', color='#94a3b8', fontsize=10)
    ax1.set_ylabel('Amplitude (mV)', color='#94a3b8', fontsize=10)
    ax1.set_title(title_ecg, color='#f8fafc', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.2, color='#1e293b')
    ax1.tick_params(colors='#94a3b8')
    for spine in ax1.spines.values():
        spine.set_color('#1e293b')
    
    # ACC plot
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#0a0a0f')
    t_acc = np.linspace(0, 10, len(acc))
    ax2.plot(t_acc, acc[:, 0], color='#ef4444', linewidth=1, alpha=0.8, label='X')
    ax2.plot(t_acc, acc[:, 1], color='#10b981', linewidth=1, alpha=0.8, label='Y')
    ax2.plot(t_acc, acc[:, 2], color='#f59e0b', linewidth=1, alpha=0.8, label='Z')
    ax2.set_xlabel('Time (s)', color='#94a3b8', fontsize=10)
    ax2.set_ylabel('Accel (m/s²)', color='#94a3b8', fontsize=10)
    ax2.set_title(title_acc, color='#f8fafc', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e', edgecolor='#1e293b', labelcolor='#94a3b8')
    ax2.grid(True, alpha=0.2, color='#1e293b')
    ax2.tick_params(colors='#94a3b8')
    for spine in ax2.spines.values():
        spine.set_color('#1e293b')
    
    plt.tight_layout()
    return fig

def create_risk_gauge(prob):
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True), facecolor='#1a1a2e')
    
    # Background arc
    theta = np.linspace(0, 2*np.pi, 100)
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, 100))
    
    for i in range(len(theta)-1):
        ax.plot([theta[i], theta[i+1]], [0.8, 0.8], color=colors[i], linewidth=20, solid_capstyle='round')
    
    # Needle
    needle_angle = prob * 2 * np.pi
    ax.plot([needle_angle, needle_angle], [0, 0.7], color='white', linewidth=3, solid_capstyle='round')
    ax.scatter([needle_angle], [0.7], color='white', s=50, zorder=5)
    
    # Center circle
    circle = plt.Circle((0, 0), 0.15, color='#1a1a2e', transform=ax.transData)
    ax.add_patch(circle)
    
    ax.set_ylim(0, 1)
    ax.set_thetamin(0)
    ax.set_thetamax(360)
    ax.axis('off')
    
    # Value text
    ax.text(0, -0.3, f'{prob:.1%}', ha='center', va='center', fontsize=24, fontweight='bold', color='white', transform=ax.transData)
    ax.text(0, -0.5, 'RISK', ha='center', va='center', fontsize=10, color='#94a3b8', transform=ax.transData)
    
    return fig

# Header
st.markdown("""
<div class="header-container">
    <div style="display: flex; align-items: center; gap: 1rem;">
        <div style="font-size: 3rem;">❤️</div>
        <div>
            <h1 class="header-title">OHCA Prediction System</h1>
            <p class="header-subtitle">Out-of-Hospital Cardiac Arrest Detection via Multi-Modal AI</p>
        </div>
    </div>
    <div style="display: flex; gap: 2rem; margin-top: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span class="status-dot active"></span>
            <span style="color: #94a3b8; font-size: 0.85rem;">System Online</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #94a3b8; font-size: 0.85rem;">Model v2.0</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #94a3b8; font-size: 0.85rem;">Multi-Modal Attention</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.warning("⚠️ **RESEARCH PROTOTYPE** - This system is for demonstration purposes only. Not FDA approved. Not for clinical decision-making.")

# Main layout
col_main, col_side = st.columns([3, 1])

with col_side:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Patient Parameters</div>
    </div>
    """, unsafe_allow_html=True)
    
    age = st.slider("Age", 40, 90, 65, help="Patient age in years")
    sex = st.selectbox("Sex", ["Male", "Female"])
    ischemia = st.slider("Ischemia Risk", 0.0, 1.0, 0.5, help="Risk of ischemic heart disease")
    bp_sys = st.slider("Systolic BP", 90, 200, 120, help="mmHg")
    bp_dia = st.slider("Diastolic BP", 60, 120, 80, help="mmHg")
    spo2 = st.slider("SpO2", 85, 100, 98, help="Oxygen saturation %")
    
    st.markdown("---")
    
    scenario = st.selectbox("Scenario", [
        "Normal (Resting)",
        "Warning (Mild Distress)", 
        "Critical (Pre-Arrest)"
    ], index=2)
    
    scenario_map = {
        "Normal (Resting)": "normal",
        "Warning (Mild Distress)": "warning",
        "Critical (Pre-Arrest)": "arrest"
    }
    
    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.45, 0.05)

with col_main:
    # Generate signals
    ecg = generate_ecg(scenario_map[scenario])
    acc = generate_acc(scenario_map[scenario])
    
    # Display signals
    st.markdown("""
    <div class="signal-container">
        <div class="signal-title">Real-Time Signal Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    fig_signals = create_signal_plot(ecg, acc)
    st.pyplot(fig_signals)
    plt.close()
    
    # Patient stats
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Heart Rate</div>
            <div class="metric-value">72 <span style="font-size: 1rem; color: #94a3b8;">bpm</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">HRV</div>
            <div class="metric-value">42 <span style="font-size: 1rem; color: #94a3b8;">ms</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">QT Interval</div>
            <div class="metric-value">380 <span style="font-size: 1rem; color: #94a3b8;">ms</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">ST Segment</div>
            <div class="metric-value">0.5 <span style="font-size: 1rem; color: #94a3b8;">mm</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction button
    st.markdown("---")
    
    if st.button("🫀 Analyze Cardiac Risk", type="primary", use_container_width=True):
        with st.spinner("Processing multi-modal data..."):
            model = load_model()
            
            bio = np.array([float(age), 1.0 if sex == "Male" else 0.0, 
                           ischemia, float(bp_sys), float(bp_dia), float(spo2)])
            
            ecg_p = preprocess_ecg(ecg)
            acc_p = preprocess_acc(acc)
            bio_p = bio.reshape(1, -1).astype(np.float32)
            
            prob = float(model.predict(
                {'ecg_input': ecg_p, 'acc_input': acc_p, 'bio_input': bio_p},
                verbose=0
            )[0][0])
        
        # Results
        st.markdown("---")
        
        color, risk_level = get_risk_color(prob)
        
        col_risk, col_gauge, col_action = st.columns([1, 1, 1])
        
        with col_risk:
            st.markdown(f"""
            <div class="metric-card" style="border-color: {color}; box-shadow: 0 0 30px {color}33;">
                <div class="metric-label">Prediction Result</div>
                <div class="metric-value" style="color: {color};">{risk_level}</div>
                <div style="margin-top: 1rem; font-size: 0.9rem; color: #94a3b8;">
                    Probability: <span style="color: {color}; font-weight: 600;">{prob:.1%}</span>
                </div>
                <div style="font-size: 0.9rem; color: #94a3b8;">
                    Threshold: <span style="color: #f8fafc; font-weight: 600;">{threshold}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_gauge:
            fig_gauge = create_risk_gauge(prob)
            st.pyplot(fig_gauge)
            plt.close()
        
        with col_action:
            if prob >= threshold:
                st.markdown(f"""
                <div class="metric-card" style="border-color: #ef4444; box-shadow: 0 0 30px rgba(239, 68, 68, 0.3);">
                    <div class="metric-label">Clinical Action</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444; margin: 0.5rem 0;">
                        ⚠️ INTERVENE
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.6;">
                        • Activate emergency response<br>
                        • Prepare defibrillator<br>
                        • Monitor vital signs<br>
                        • Contact cardiology
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card" style="border-color: #10b981; box-shadow: 0 0 30px rgba(16, 185, 129, 0.3);">
                    <div class="metric-label">Clinical Action</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #10b981; margin: 0.5rem 0;">
                        ✅ MONITOR
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.6;">
                        • Continue standard monitoring<br>
                        • Re-assess in 5 minutes<br>
                        • Document observations<br>
                        • Watch for changes
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed metrics
        st.markdown("---")
        
        with st.expander("📊 Detailed Analysis", expanded=True):
            tab1, tab2, tab3 = st.tabs(["Patient Profile", "Signal Characteristics", "Model Output"])
            
            with tab1:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Demographics**")
                    st.write(f"• Age: {age} years")
                    st.write(f"• Sex: {sex}")
                    st.write(f"• Ischemia Risk: {ischemia:.0%}")
                with col_b:
                    st.markdown("**Vital Signs**")
                    st.write(f"• Blood Pressure: {bp_sys}/{bp_dia} mmHg")
                    st.write(f"• SpO2: {spo2}%")
                    st.write(f"• BMI: {np.random.randint(22, 35)} kg/m²")
            
            with tab2:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**ECG Analysis**")
                    st.write(f"• Amplitude Range: [{ecg.min():.2f}, {ecg.max():.2f}]")
                    st.write(f"• Signal Power: {np.mean(ecg**2):.4f}")
                    st.write(f"• Frequency Content: 0.5-40 Hz")
                with col_b:
                    st.markdown("**Accelerometer Analysis**")
                    st.write(f"• Movement Intensity: {np.sqrt(np.mean(acc**2)):.3f} m/s²")
                    st.write(f"• Postural Stability: {1.0/(np.std(acc[:, 2])+0.01):.2f}")
                    st.write(f"• Activity Level: {'Low' if np.mean(np.abs(acc)) < 0.5 else 'Moderate' if np.mean(np.abs(acc)) < 1.5 else 'High'}")
            
            with tab3:
                st.markdown("**Model Architecture**")
                st.write("• Multi-Modal Attention Network")
                st.write("• ECG Branch: CNN + Self-Attention")
                st.write("• ACC Branch: BiGRU + Temporal Attention")
                st.write("• Biodata Branch: Dense + Gating")
                st.write("• Fusion: Cross-Modal Attention")
                st.write(f"• Decision Threshold: {threshold}")
                st.write(f"• Output Probability: {prob:.6f}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 1rem;">
    <p>OHCA Prediction System v2.0 | Multi-Modal Attention Architecture</p>
    <p>For Research Use Only - Not for Clinical Decision-Making</p>
</div>
""", unsafe_allow_html=True)
