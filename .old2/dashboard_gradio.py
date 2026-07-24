#!/usr/bin/env python3
"""
OHCA Prediction Dashboard - Gradio Premium Edition
"""
import gradio as gr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Load model
print("Loading CardiacGuard AI V3 model...")
model = tf.keras.models.load_model('models_v3/best_v3.keras', compile=False)
print("Model loaded successfully!")

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
    
    if scenario == 'arrest':
        heart_rate = 45 + np.random.randint(0, 20)
        st_segment = -0.15
        qt_prolonged = True
    elif scenario == 'warning':
        heart_rate = 85 + np.random.randint(0, 20)
        st_segment = -0.08
        qt_prolonged = False
    else:
        heart_rate = 68 + np.random.randint(0, 10)
        st_segment = 0
        qt_prolonged = False
    
    rr_interval = 60 / heart_rate
    ecg = np.zeros(1300)
    beat_pos = 0
    
    while beat_pos < 1300:
        p_start = beat_pos + int(0.1 * 130)
        p_width = int(0.08 * 130)
        if p_start + p_width < 1300:
            p_wave = 0.15 * np.exp(-((np.arange(p_width) - p_width/2)**2) / (2 * (p_width/6)**2))
            ecg[p_start:p_start+p_width] += p_wave
        
        qrs_start = beat_pos + int(0.16 * 130)
        qrs_width = int(0.08 * 130)
        if qrs_start + qrs_width < 1300:
            qrs = np.zeros(qrs_width)
            qrs[int(qrs_width*0.2)] = -0.1
            qrs[int(qrs_width*0.35)] = 1.2
            qrs[int(qrs_width*0.5)] = -0.2
            ecg[qrs_start:qrs_start+qrs_width] += qrs
        
        t_start = beat_pos + int(0.35 * 130)
        t_width = int(0.15 * 130)
        if t_start + t_width < 1300:
            t_wave = 0.3 * np.exp(-((np.arange(t_width) - t_width/2)**2) / (2 * (t_width/5)**2))
            if qt_prolonged:
                t_start += int(0.05 * 130)
            ecg[t_start:t_start+t_width] += t_wave
        
        st_start = beat_pos + int(0.28 * 130)
        st_width = int(0.07 * 130)
        if st_start + st_width < 1300:
            ecg[st_start:st_start+st_width] += st_segment
        
        beat_pos += int(rr_interval * 130)
    
    ecg += np.random.randn(1300) * 0.02
    ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
    return ecg

def generate_acc(scenario='normal'):
    t = np.linspace(0, 10, 250)
    
    if scenario == 'arrest':
        acc = np.zeros((250, 3))
        acc[:, 2] = 9.81
        acc[:, 1] += np.sin(t * 0.8) * 0.3
        acc[100:120, 2] -= np.linspace(0, 3, 20)
        acc[120:150, 2] += np.linspace(3, 0, 30)
        acc += np.random.randn(250, 3) * 0.05
    elif scenario == 'warning':
        acc = np.random.randn(250, 3) * 0.2
        acc[:, 2] += 9.81
        acc[:, 1] += np.sin(t * 1.5) * 0.2 + np.sin(t * 0.3) * 0.1
    else:
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

def create_signal_plot(ecg, acc, scenario):
    fig = plt.figure(figsize=(14, 5), facecolor='#0a0a0f')
    gs = GridSpec(1, 2, width_ratios=[2, 1], wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#0a0a0f')
    t_ecg = np.linspace(0, 10, len(ecg))
    ax1.plot(t_ecg, ecg, color='#06b6d4', linewidth=1.2, alpha=0.9)
    ax1.fill_between(t_ecg, ecg, alpha=0.1, color='#06b6d4')
    ax1.set_xlabel('Time (s)', color='#94a3b8', fontsize=10)
    ax1.set_ylabel('Amplitude (mV)', color='#94a3b8', fontsize=10)
    ax1.set_title(f'ECG Signal ({scenario})', color='#f8fafc', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.2, color='#1e293b')
    ax1.tick_params(colors='#94a3b8')
    for spine in ax1.spines.values():
        spine.set_color('#1e293b')
    
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#0a0a0f')
    t_acc = np.linspace(0, 10, len(acc))
    ax2.plot(t_acc, acc[:, 0], color='#ef4444', linewidth=1, alpha=0.8, label='X')
    ax2.plot(t_acc, acc[:, 1], color='#10b981', linewidth=1, alpha=0.8, label='Y')
    ax2.plot(t_acc, acc[:, 2], color='#f59e0b', linewidth=1, alpha=0.8, label='Z')
    ax2.set_xlabel('Time (s)', color='#94a3b8', fontsize=10)
    ax2.set_ylabel('Accel (m/s²)', color='#94a3b8', fontsize=10)
    ax2.set_title('Accelerometer Data', color='#f8fafc', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', fontsize=8, facecolor='#0a0a0f', edgecolor='#1e293b', labelcolor='#94a3b8')
    ax2.grid(True, alpha=0.2, color='#1e293b')
    ax2.tick_params(colors='#94a3b8')
    for spine in ax2.spines.values():
        spine.set_color('#1e293b')
    
    plt.tight_layout()
    return fig

def create_risk_gauge(prob):
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True), facecolor='#0a0a0f')
    
    theta = np.linspace(0, 2*np.pi, 100)
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, 100))
    
    for i in range(len(theta)-1):
        ax.plot([theta[i], theta[i+1]], [0.8, 0.8], color=colors[i], linewidth=20, solid_capstyle='round')
    
    needle_angle = prob * 2 * np.pi
    ax.plot([needle_angle, needle_angle], [0, 0.7], color='white', linewidth=3, solid_capstyle='round')
    ax.scatter([needle_angle], [0.7], color='white', s=50, zorder=5)
    
    circle = plt.Circle((0, 0), 0.15, color='#0a0a0f', transform=ax.transData)
    ax.add_patch(circle)
    
    ax.set_ylim(0, 1)
    ax.set_thetamin(0)
    ax.set_thetamax(360)
    ax.axis('off')
    
    ax.text(0, -0.3, f'{prob:.1%}', ha='center', va='center', fontsize=24, fontweight='bold', color='white', transform=ax.transData)
    ax.text(0, -0.5, 'RISK', ha='center', va='center', fontsize=10, color='#94a3b8', transform=ax.transData)
    
    return fig

def analyze_cardiac_risk(age, sex, ischemia, bp_sys, bp_dia, spo2, scenario, threshold):
    """Main prediction function"""
    # Generate signals based on scenario
    scenario_map = {
        "Normal (Resting)": "normal",
        "Warning (Mild Distress)": "warning",
        "Critical (Pre-Arrest)": "arrest"
    }
    
    ecg = generate_ecg(scenario_map[scenario])
    acc = generate_acc(scenario_map[scenario])
    
    # Create signal plot
    signal_plot = create_signal_plot(ecg, acc, scenario)
    
    # Prepare inputs
    bio = np.array([float(age), 1.0 if sex == "Male" else 0.0, 
                    ischemia, float(bp_sys), float(bp_dia), float(spo2)])
    
    ecg_p = preprocess_ecg(ecg)
    acc_p = preprocess_acc(acc)
    bio_p = bio.reshape(1, -1).astype(np.float32)
    
    # Predict
    prob = float(model.predict(
        {'ecg_input': ecg_p, 'acc_input': acc_p, 'bio_input': bio_p},
        verbose=0
    )[0][0])
    
    # Create gauge
    gauge_plot = create_risk_gauge(prob)
    
    # Get risk level
    color, risk_level = get_risk_color(prob)
    
    # Determine action
    if prob >= threshold:
        action = "⚠️ INTERVENE"
        action_color = "#ef4444"
        action_details = """**Immediate Actions Required:**
- Activate emergency response
- Prepare defibrillator
- Monitor vital signs continuously
- Contact cardiology on-call
- Prepare for possible CPR"""
    else:
        action = "✅ MONITOR"
        action_color = "#10b981"
        action_details = """**Continue Standard Care:**
- Continue standard monitoring
- Re-assess in 5 minutes
- Document observations
- Watch for changes in condition"""
    
    # Create results HTML
    results_html = f"""
    <div style="font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #0a0a0f 0%, #12121a 100%); padding: 2rem; border-radius: 16px; border: 1px solid #1e293b;">
        <h2 style="color: #f8fafc; margin-bottom: 1rem; font-size: 1.5rem;">Cardiac Risk Assessment</h2>
        
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, rgba(26, 26, 46, 0.8) 100%); border: 1px solid {color}; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 0 30px {color}33;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; margin-bottom: 0.5rem;">Risk Level</div>
            <div style="font-size: 2rem; font-weight: 700; color: {color}; margin-bottom: 0.5rem;">{risk_level}</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{prob:.1%}</div>
        </div>
        
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, rgba(26, 26, 46, 0.8) 100%); border: 1px solid {action_color}; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; margin-bottom: 0.5rem;">Recommended Action</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {action_color}; margin-bottom: 1rem;">{action}</div>
            <div style="color: #94a3b8; line-height: 1.6; font-size: 0.9rem;">{action_details}</div>
        </div>
        
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, rgba(26, 26, 46, 0.8) 100%); border: 1px solid #1e293b; border-radius: 12px; padding: 1.5rem;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; margin-bottom: 0.5rem;">Patient Profile</div>
            <div style="color: #f8fafc; font-size: 0.9rem; line-height: 1.8;">
                • Age: {age} years<br>
                • Sex: {sex}<br>
                • Ischemia Risk: {ischemia:.0%}<br>
                • Blood Pressure: {bp_sys}/{bp_dia} mmHg<br>
                • SpO2: {spo2}%<br>
                • Decision Threshold: {threshold}
            </div>
        </div>
    </div>
    """
    
    return signal_plot, gauge_plot, results_html

# Custom CSS
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0a0a0f 0%, #12121a 100%) !important;
}
.gradio-container .gr-box {
    background: linear-gradient(135deg, #1a1a2e 0%, rgba(26, 26, 46, 0.8) 100%) !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
}
.gradio-container .gr-label {
    color: #94a3b8 !important;
    font-weight: 500 !important;
}
.gradio-container .gr-input {
    background: #0a0a0f !important;
    border: 1px solid #1e293b !important;
    color: #f8fafc !important;
    border-radius: 8px !important;
}
.gradio-container .gr-input:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 15px rgba(79, 70, 229, 0.3) !important;
}
.gradio-container .gr-button-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.gradio-container .gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4) !important;
}
.gradio-container .gr-title {
    color: #f8fafc !important;
}
.gradio-container .gr-markdown {
    color: #94a3b8 !important;
}
gradio-container {background: #0a0a0f !important}
"""

# Build interface
with gr.Blocks(css=custom_css, title="CardiacGuard AI - OHCA Prediction", theme=gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)) as demo:
    
    # Header
    gr.Markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%); border: 1px solid rgba(79, 70, 229, 0.3); border-radius: 16px; margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #fff 0%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">CardiacGuard AI</h1>
        <p style="font-size: 1rem; color: #94a3b8; margin-top: 0.5rem;">Out-of-Hospital Cardiac Arrest Prediction via Multi-Modal AI</p>
        <p style="font-size: 0.85rem; color: #06b6d4; margin-top: 1rem;">V3 Model | AUC: 0.998 | Accuracy: 98.1% | Sensitivity: 98.6%</p>
    </div>
    """)
    
    gr.Markdown("⚠️ **RESEARCH PROTOTYPE** - This system is for demonstration purposes only. Not FDA approved. Not for clinical decision-making.", elem_classes=["warning"])
    
    with gr.Row():
        # Left column - Controls
        with gr.Column(scale=1):
            gr.Markdown("### Patient Parameters")
            
            age = gr.Slider(minimum=40, maximum=90, value=65, step=1, label="Age")
            sex = gr.Radio(["Male", "Female"], value="Male", label="Sex")
            ischemia = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Ischemia Risk")
            bp_sys = gr.Slider(minimum=90, maximum=200, value=120, step=1, label="Systolic BP (mmHg)")
            bp_dia = gr.Slider(minimum=60, maximum=120, value=80, step=1, label="Diastolic BP (mmHg)")
            spo2 = gr.Slider(minimum=85, maximum=100, value=98, step=1, label="SpO2 (%)")
            
            gr.Markdown("### Scenario")
            scenario = gr.Radio(
                ["Normal (Resting)", "Warning (Mild Distress)", "Critical (Pre-Arrest)"],
                value="Critical (Pre-Arrest)",
                label="Clinical Scenario"
            )
            
            threshold = gr.Slider(minimum=0.1, maximum=0.9, value=0.45, step=0.05, label="Decision Threshold")
            
            analyze_btn = gr.Button("Analyze Cardiac Risk", variant="primary", size="lg")
        
        # Right column - Results
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Signal Analysis"):
                    signal_plot = gr.Plot(label="ECG & Accelerometer Signals")
                
                with gr.TabItem("Risk Assessment"):
                    gauge_plot = gr.Plot(label="Risk Gauge")
                
                with gr.TabItem("Clinical Report"):
                    results_html = gr.HTML(label="Detailed Results")
    
    # Footer
    gr.Markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 2rem; border-top: 1px solid #1e293b; margin-top: 2rem;">
        <p>CardiacGuard AI V3 | Multi-Modal Attention Architecture</p>
        <p>For Research Use Only - Not for Clinical Decision-Making</p>
        <p style="color: #06b6d4;">Validated on synthetic data with 98.1% accuracy and 0.998 AUC-ROC</p>
    </div>
    """)
    
    # Connect button
    analyze_btn.click(
        fn=analyze_cardiac_risk,
        inputs=[age, sex, ischemia, bp_sys, bp_dia, spo2, scenario, threshold],
        outputs=[signal_plot, gauge_plot, results_html]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
