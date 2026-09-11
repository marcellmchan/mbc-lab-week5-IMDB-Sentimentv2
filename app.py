import os
import streamlit as st
import numpy as np
import pickle
import re
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import onnxruntime as ort

# ============================================================
# KONFIGURASI
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
H5_MODEL_PATH = os.path.join(BASE_DIR, "sentiment_GRU_cfg1_seq100.h5")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "sentiment_gru.onnx")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
MAX_LEN = 100

st.set_page_config(page_title="IMDB Sentiment Classifier v2", page_icon="🎬", layout="centered")

st.markdown("""
<style>
h1 { color: #1A1A2E; }
.stButton>button { background: #16213E; color: #FFFFFF; border: none; }
.stButton>button:hover { background: #0F3460; color: #FFFFFF; }
div[role="radiogroup"] label { border: 1px solid #DCE3D0; border-radius: 4px; padding: 0.2rem 0.6rem; margin-right: 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_keras_model():
    return load_model(H5_MODEL_PATH)

@st.cache_resource
def load_onnx_model():
    return ort.InferenceSession(ONNX_MODEL_PATH)

@st.cache_resource
def get_tokenizer():
    with open(TOKENIZER_PATH, 'rb') as f:
        return pickle.load(f)

def preprocess_text(text, tokenizer):
    text = text.lower()
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

def predict_onnx(session, processed):
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: processed.astype(np.float32)})
    return result[0][0][0]

# ============================================================
# SESSION STATE UNTUK RIWAYAT PREDIKSI
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# UI
# ============================================================
st.title("IMDB Movie Review Sentiment Classifier")
st.caption("Versi 2")
st.write(
    "Masukkan review film untuk diprediksi sentimennya. "
    "Versi ini menambahkan breakdown probabilitas kedua kelas, "
    "riwayat prediksi, dan opsi model teroptimasi (ONNX Runtime)."
)

# ============================================================
# INFO OPTIMASI MODEL
# ============================================================
with st.expander("📊 Info Optimasi Model (ONNX Runtime)"):
    col1, col2, col3 = st.columns(3)
    if os.path.exists(H5_MODEL_PATH):
        h5_mb = os.path.getsize(H5_MODEL_PATH) / (1024 * 1024)
        col1.metric("Model Original (.h5)", f"{h5_mb:.2f} MB")
    if os.path.exists(ONNX_MODEL_PATH):
        onnx_mb = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024)
        col2.metric("Model ONNX (.onnx)", f"{onnx_mb:.2f} MB")
        if os.path.exists(H5_MODEL_PATH):
            saving = (1 - onnx_mb / h5_mb) * 100
            col3.metric("Penghematan", f"{saving:.1f}%" if saving > 0 else "Lihat kecepatan")

    st.markdown("""
    **ONNX Runtime (Open Neural Network Exchange):**
    - Format model universal yang dapat berjalan lebih cepat di berbagai platform.
    - Inferensi dioptimalkan oleh ONNX Runtime engine, bukan TensorFlow.
    - GRU dan semua arsitektur RNN **fully supported** oleh ONNX.

    > *"ONNX Runtime: Memungkinkan model berjalan lebih cepat di berbagai platform"* — Materi Week 5 HDT
    """)

# ============================================================
# PILIHAN MODEL (SEPERTI APPLE VS ORANGE V2)
# ============================================================
model_choice = st.radio(
    "Pilih model inferensi",
    ["Model asli (.h5)", "Model teroptimasi (.onnx - ONNX Runtime)"],
    horizontal=True,
)

# ============================================================
# INPUT TEKS
# ============================================================
user_input = st.text_area(
    "Masukkan review film (Bahasa Inggris):",
    "This movie was absolutely fantastic! The storytelling was superb.",
    help="Tulis review dalam Bahasa Inggris untuk hasil terbaik.",
)

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Review tidak boleh kosong.")
    else:
        with st.spinner("Model sedang memproses..."):
            tokenizer = get_tokenizer()
            processed = preprocess_text(user_input, tokenizer)
            start = time.time()

            if model_choice.startswith("Model asli"):
                keras_model = load_keras_model()
                pred = keras_model.predict(processed, verbose=0)[0][0]
                model_label = "H5 (Keras)"
            else:
                onnx_session = load_onnx_model()
                pred = predict_onnx(onnx_session, processed)
                model_label = "ONNX Runtime"

            elapsed = time.time() - start

        prob_positive = float(pred)
        prob_negative = 1 - prob_positive
        pred_class = "Positive" if prob_positive > 0.5 else "Negative"
        confidence = max(prob_positive, prob_negative)

        st.success(
            f"Prediksi: **{pred_class}** ({confidence * 100:.2f}% confidence) "
            f"| ⏱ Inferensi: {elapsed * 1000:.1f}ms | Model: {model_label}"
        )

        st.write("Breakdown probabilitas:")
        st.write(f"🌟 Positive: {prob_positive * 100:.2f}%")
        st.progress(prob_positive)
        st.write(f"💔 Negative: {prob_negative * 100:.2f}%")
        st.progress(prob_negative)

        st.session_state.history.append({
            "Review (50 karakter pertama)": user_input[:50] + "...",
            "Prediksi": pred_class,
            "Confidence": f"{confidence * 100:.2f}%",
            "Model": model_label,
            "Waktu (ms)": f"{elapsed * 1000:.1f}",
        })

if st.session_state.history:
    st.markdown("---")
    st.subheader("Riwayat Prediksi (sesi ini)")
    st.table(st.session_state.history)

st.markdown("---")
st.caption("Model GRU dilatih pada dataset IMDB 50K Reviews. Opsi ONNX Runtime tersedia untuk inferensi yang lebih cepat.")
