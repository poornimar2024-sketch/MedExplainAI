import os
import io
import re
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(
    page_title="MedExplainAI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model("pneumonia_model.h5")

model = load_ai_model()

def describe_heatmap(heatmap):
    h, w = heatmap.shape

    left_half = heatmap[:, :w // 2].mean()
    right_half = heatmap[:, w // 2:].mean()

    upper_half = heatmap[:h // 2, :].mean()
    lower_half = heatmap[h // 2:, :].mean()

    if abs(left_half - right_half) < 0.03:
        side_desc = "both lungs fairly evenly"
    elif right_half > left_half:
        side_desc = "the right side more strongly"
    else:
        side_desc = "the left side more strongly"

    if abs(upper_half - lower_half) < 0.03:
        region_desc = "across both upper and lower lung regions"
    elif lower_half > upper_half:
        region_desc = "mainly in the lower lung region"
    else:
        region_desc = "mainly in the upper lung region"

    high_attention_ratio = (heatmap > 0.6).mean()
    if high_attention_ratio < 0.12:
        pattern_desc = "a relatively localized focus"
    elif high_attention_ratio < 0.25:
        pattern_desc = "a moderately spread abnormal focus"
    else:
        pattern_desc = "a more diffuse attention pattern"

    return (
        f"The model focuses on {side_desc}, {region_desc}, with {pattern_desc}. "
        f"This is an AI-generated interpretability summary and not a clinical report."
    )

def generate_report(prediction):
    if prediction > 0.90:
        return [
            "High probability of pneumonia pattern detected.",
            "Model output suggests strong abnormal lung-related features.",
            "Further clinical evaluation or radiologist review is recommended."
        ]
    elif prediction > 0.70:
        return [
            "Moderate abnormal pattern detected.",
            "The findings are suggestive but not definitive.",
            "Additional expert review or imaging may be helpful."
        ]
    else:
        return [
            "No strong pneumonia pattern detected.",
            "The scan appears relatively more consistent with a normal chest X-ray.",
            "Clinical interpretation should still be based on expert review."
        ]

def get_result_label(prediction):
    if prediction > 0.95:
        return "Pneumonia Detected", "bad"
    elif prediction > 0.70:
        return "Possible Pneumonia", "warn"
    else:
        return "Likely Normal", "good"

def safe_filename(name):
    cleaned = name.strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", cleaned)
    return cleaned or "Patient"

def create_pdf_report(name, age, prediction, pneumonia_percent, normal_percent, result_label, report_items):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("MedExplainAI Screening Report", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Patient Name: {name}", styles["Normal"]))
    content.append(Paragraph(f"Patient Age: {age}", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Prediction Score: {prediction:.4f}", styles["Normal"]))
    content.append(Paragraph(f"Pneumonia Probability: {pneumonia_percent:.1f}%", styles["Normal"]))
    content.append(Paragraph(f"Normal Probability: {normal_percent:.1f}%", styles["Normal"]))
    content.append(Paragraph(f"Final Result: {result_label}", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("AI Screening Summary", styles["Heading2"]))
    content.append(Spacer(1, 6))

    for item in report_items:
        content.append(Paragraph(f"- {item}", styles["Normal"]))

    content.append(Spacer(1, 12))
    content.append(Paragraph(
        "Note: This is an AI-generated educational screening result and not a clinical diagnosis.",
        styles["Italic"]
    ))

    doc.build(content)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at top left, rgba(30, 64, 175, 0.35), transparent 30%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.18), transparent 30%),
        linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
    color: #f8fafc;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #08101f 0%, #0b1220 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
.hero {
    padding: 1.6rem 1.7rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(37,99,235,0.22), rgba(14,165,233,0.10));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 20px 50px rgba(0,0,0,0.28);
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.4rem;
    color: #f8fafc;
}
.hero-subtitle {
    color: #cbd5e1;
    font-size: 1.05rem;
    line-height: 1.6;
}
.hero-pill {
    display: inline-block;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.08);
    color: #dbeafe;
    font-size: 0.86rem;
    margin-right: 0.45rem;
    margin-top: 0.5rem;
}
.glass-card {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 22px;
    padding: 1.2rem 1.25rem;
    box-shadow: 0 14px 35px rgba(0,0,0,0.24);
    margin-bottom: 1rem;
}
.card-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
    color: #f8fafc;
}
.small-note {
    color: #94a3b8;
    font-size: 0.92rem;
    line-height: 1.6;
}
.result-badge-good,
.result-badge-warn,
.result-badge-bad {
    display: inline-block;
    padding: 0.55rem 0.95rem;
    border-radius: 14px;
    font-weight: 700;
    font-size: 1rem;
    margin: 0.4rem 0 0.8rem 0;
}
.result-badge-good {
    color: #dcfce7;
    background: rgba(34,197,94,0.16);
    border: 1px solid rgba(34,197,94,0.28);
}
.result-badge-warn {
    color: #fef3c7;
    background: rgba(245,158,11,0.14);
    border: 1px solid rgba(245,158,11,0.28);
}
.result-badge-bad {
    color: #fee2e2;
    background: rgba(239,68,68,0.16);
    border: 1px solid rgba(239,68,68,0.28);
}
.metric-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1rem;
    text-align: center;
    min-height: 110px;
}
.metric-label {
    color: #94a3b8;
    font-size: 0.88rem;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-size: 1.35rem;
    font-weight: 800;
    color: #f8fafc;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border: 1px dashed rgba(255,255,255,0.18);
    border-radius: 18px;
    padding: 0.8rem;
}
img {
    border-radius: 18px !important;
}
.caption-box {
    color: #94a3b8;
    font-size: 0.88rem;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🫁 MedExplainAI")
    st.write("AI-powered chest X-ray screening interface")
    st.markdown("---")
    st.markdown("### Controls")
    show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=False)
    use_sample = st.checkbox("Use sample demo image", value=False)
    show_debug = st.checkbox("Show raw debug info", value=False)
    st.markdown("---")
    st.markdown("### Workflow")
    st.write("1. Enter patient details")
    st.write("2. Upload or load a sample chest X-ray")
    st.write("3. Let the model analyze it")
    st.write("4. Review prediction and confidence")
    st.write("5. Download PDF report")
    st.markdown("---")
    st.warning("Educational use only. Not for clinical diagnosis.")

st.markdown("""
<div class="hero">
    <div class="hero-title">MedExplainAI</div>
    <div class="hero-subtitle">
        A visually guided AI screening tool for pneumonia detection from chest X-ray images,
        powered by MobileNetV2 transfer learning and optional Grad-CAM interpretability.
    </div>
    <div>
        <span class="hero-pill">Deep Learning</span>
        <span class="hero-pill">Chest X-ray Analysis</span>
        <span class="hero-pill">Explainable AI</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">👤 Patient Information</div>', unsafe_allow_html=True)

p1, p2 = st.columns(2)
with p1:
    patient_name = st.text_input("Patient Name")
with p2:
    patient_age = st.text_input("Patient Age")

st.markdown('</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

sample_path = "sample.jpg"
image_source = None

if use_sample and os.path.exists(sample_path):
    image_source = Image.open(sample_path).convert("RGB")
elif uploaded_file is not None:
    image_source = Image.open(uploaded_file).convert("RGB")

if image_source is None:
    c1, c2 = st.columns([1.15, 0.85], gap="large")

    with c1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">Welcome</div>
            <p>This application analyzes chest X-ray images and produces an AI-based pneumonia screening result.</p>
            <p>You can use it to demonstrate:</p>
            <ul>
                <li>Binary chest X-ray classification</li>
                <li>Confidence-based result reporting</li>
                <li>Optional Grad-CAM interpretability</li>
                <li>Readable AI-generated abnormality summaries</li>
                <li>Mini screening report generation</li>
                <li>Downloadable PDF report</li>
            </ul>
            <div class="small-note">Enter patient details and upload an image or enable sample demo mode to begin analysis.</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">Model Overview</div>
            <p><b>Architecture:</b> MobileNetV2</p>
            <p><b>Classification Type:</b> Binary</p>
            <p><b>Target Classes:</b> NORMAL / PNEUMONIA</p>
            <p><b>Input Format:</b> 224 × 224 RGB</p>
            <div class="small-note">Grad-CAM remains optional because interpretability maps may not always localize pathology perfectly.</div>
        </div>
        """, unsafe_allow_html=True)

else:
    original_img = image_source
    display_img = original_img.copy()
    model_img = original_img.resize((224, 224))

    img_array = np.array(model_img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Uploaded X-ray</div>', unsafe_allow_html=True)
        st.image(display_img, use_container_width=True)
        st.markdown(
            '<div class="caption-box">Original image displayed for visual review.</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        with st.spinner("🔍 Analyzing lung patterns using AI..."):
            prediction = model.predict(img_array, verbose=0)[0][0]

        pneumonia_percent = float(prediction * 100)
        normal_percent = float((1 - prediction) * 100)

        result_label, result_type = get_result_label(prediction)

        if result_type == "bad":
            label_html = '<div class="result-badge-bad">🦠 Pneumonia Detected</div>'
            result_text = "The model predicts a high probability of pneumonia."
            confidence_display = pneumonia_percent
        elif result_type == "warn":
            label_html = '<div class="result-badge-warn">⚠️ Possible Pneumonia</div>'
            result_text = "The model predicts a moderate probability of pneumonia."
            confidence_display = pneumonia_percent
        else:
            label_html = '<div class="result-badge-good">✅ Likely Normal</div>'
            result_text = "The model predicts that the scan is more consistent with a normal chest X-ray."
            confidence_display = normal_percent

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">AI Screening Result</div>', unsafe_allow_html=True)
        st.markdown(label_html, unsafe_allow_html=True)
        st.success("✔️ AI analysis complete")
        st.write(result_text)
        st.write(f"**Raw prediction score:** `{prediction:.4f}`")
        st.metric("Confidence", f"{confidence_display:.1f}%")

        if confidence_display > 90:
            st.caption("High confidence prediction")
        elif confidence_display > 70:
            st.caption("Moderate confidence prediction")
        else:
            st.caption("Low confidence — interpret cautiously")

        st.progress(float(min(max(confidence_display / 100, 0.0), 1.0)))
        st.info("The model may rely on visual patterns rather than true pathology, highlighting the importance of dataset quality in medical AI.")
        st.markdown(
            '<div class="small-note">This is an AI-based educational screening result and may produce false positives or false negatives.</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3, gap="large")
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pneumonia Score</div>
            <div class="metric-value">{pneumonia_percent:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Normal Score</div>
            <div class="metric-value">{normal_percent:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value">MobileNetV2</div>
        </div>
        """, unsafe_allow_html=True)

    report_items = generate_report(prediction)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧾 AI Screening Report</div>', unsafe_allow_html=True)
    for item in report_items:
        st.write(f"- {item}")

    if patient_name and patient_age:
        pdf_data = create_pdf_report(
            patient_name,
            patient_age,
            prediction,
            pneumonia_percent,
            normal_percent,
            result_label,
            report_items
        )

        file_name = f"{safe_filename(patient_name)}_Report.pdf"

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_data,
            file_name=file_name,
            mime="application/pdf"
        )
    else:
        st.warning("Please enter patient name and age to enable PDF report download.")

    st.markdown('</div>', unsafe_allow_html=True)

    ex1, ex2 = st.columns(2, gap="large")

    with ex1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🧠 Why this prediction?</div>', unsafe_allow_html=True)
        st.write("""
- The model analyzes patterns such as lung opacity, texture irregularities, and density changes.
- It does not understand disease the way a doctor does.
- It learns visual patterns from training data and uses them to estimate whether pneumonia is likely.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with ex2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">⚙️ Model Details</div>', unsafe_allow_html=True)
        st.write("""
- Architecture: MobileNetV2 (Transfer Learning)
- Task: Binary classification
- Input size: 224 × 224 RGB
- Classes: NORMAL vs PNEUMONIA
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">⚠️ Limitations</div>', unsafe_allow_html=True)
    st.write("""
- The model may produce false positives or false negatives.
- Performance depends heavily on the training dataset.
- External images from Google may differ from the training distribution.
- This tool is not suitable for real clinical diagnosis without expert review.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    if show_debug:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🛠 Raw Debug Info</div>', unsafe_allow_html=True)
        st.write("Prediction:", float(prediction))
        st.write("Pneumonia %:", pneumonia_percent)
        st.write("Normal %:", normal_percent)
        st.markdown('</div>', unsafe_allow_html=True)

    if show_gradcam:
        st.write("")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Grad-CAM Interpretability View</div>', unsafe_allow_html=True)

        try:
            base_model = model.layers[0]
            last_conv_layer = base_model.get_layer("Conv_1")

            grad_model = tf.keras.models.Model(
                inputs=base_model.input,
                outputs=[last_conv_layer.output, base_model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, base_features = grad_model(img_array, training=False)
                tape.watch(conv_outputs)

                x = model.layers[1](base_features)
                preds = model.layers[2](x)
                loss = preds[:, 0]

            grads = tape.gradient(loss, conv_outputs)

            if grads is None:
                st.warning("Grad-CAM could not be generated for this image.")
            else:
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_outputs = conv_outputs[0]
                heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

                heatmap = np.maximum(heatmap.numpy(), 0)
                heatmap = heatmap / (np.max(heatmap) + 1e-8)

                abnormality_description = describe_heatmap(heatmap)

                heatmap_resized = cv2.resize(heatmap, (224, 224))
                heatmap_uint8 = np.uint8(255 * heatmap_resized)
                heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

                img_cv = cv2.cvtColor(np.array(model_img), cv2.COLOR_RGB2BGR)
                superimposed_img = cv2.addWeighted(img_cv, 0.82, heatmap_colored, 0.18, 0)
                superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

                g1, g2 = st.columns(2, gap="large")

                with g1:
                    st.markdown("**Model Input (224 × 224)**")
                    st.image(model_img, use_container_width=True)

                with g2:
                    st.markdown("**Grad-CAM Heatmap**")
                    st.image(superimposed_img, use_container_width=True)

                st.markdown('<div class="glass-card" style="margin-top: 1rem; margin-bottom: 0;">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">AI Description of Abnormality Pattern</div>', unsafe_allow_html=True)
                st.info(abnormality_description)
                st.caption("Grad-CAM is an experimental interpretability view and may not always localize pathology accurately.")
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception:
            st.warning("Grad-CAM is unavailable for this model configuration.")

        st.markdown('</div>', unsafe_allow_html=True)