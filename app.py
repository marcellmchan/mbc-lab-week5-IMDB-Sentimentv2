import streamlit as st
import numpy as np
import pickle
import re
import os
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="IMDB Sentiment - V2 (Optimized)", page_icon="⚡")

st.title("⚡ IMDB Movie Review Sentiment Analysis")
st.write("**Versi 2 (Optimasi TFLite):** Implementasi Quantization & TensorFlow Lite sesuai materi MLOps.")
st.markdown("---")

H5_PATH = 'sentiment_GRU_cfg1_seq100.h5'
TFLITE_PATH = 'sentiment_gru_optimized.tflite'

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def get_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

def preprocess_text(text, tokenizer, max_len=100):
    text = text.lower()
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post').astype(np.float32)

# ============================================================
# Info Optimasi Model
# ============================================================
if os.path.exists(H5_PATH) and os.path.exists(TFLITE_PATH):
    h5_mb = os.path.getsize(H5_PATH) / (1024 * 1024)
    tfl_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    saving = (1 - tfl_mb / h5_mb) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Ukuran Original (.h5)", f"{h5_mb:.2f} MB")
    col2.metric("Ukuran Optimasi (.tflite)", f"{tfl_mb:.2f} MB")
    col3.metric("Penghematan Ukuran", f"{saving:.1f}%")

    st.markdown("---")

with st.expander("📖 Teknik Optimasi yang Digunakan"):
    st.markdown("""
    **Dynamic Range Quantization (TFLite):**
    - Bobot model dikonversi dari **Float32 → Int8**, mengurangi ukuran model secara drastis.
    - Model tetap menghasilkan output Float32 saat inferensi.
    - Cocok untuk deployment di server resource-terbatas (Streamlit Community Cloud, Edge Devices).

    > *"TensorFlow Lite adalah framework ringan dari TensorFlow yang dirancang untuk menjalankan model ML di edge devices"* — Materi Week 5
    """)

st.markdown("---")

# ============================================================
# Load model TFLite
# ============================================================
interpreter = load_tflite_model()
tokenizer = get_tokenizer()

user_input = st.text_area(
    "Masukkan review film (Bahasa Inggris):",
    "This movie was absolutely fantastic! The acting was superb."
)

if st.button("Analisis Sentimen"):
    if user_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        with st.spinner("Menganalisis dengan TFLite..."):
            processed = preprocess_text(user_input, tokenizer)

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], processed)
            start = time.time()
            interpreter.invoke()
            elapsed = time.time() - start
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        st.subheader("Hasil Prediksi:")
        if prediction > 0.5:
            st.success(f"🌟 **POSITIF** — Confidence: {prediction:.2%} | Waktu inferensi: {elapsed*1000:.1f}ms")
        else:
            st.error(f"💔 **NEGATIF** — Confidence: {(1-prediction):.2%} | Waktu inferensi: {elapsed*1000:.1f}ms")
