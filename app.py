import streamlit as st
import numpy as np
import pickle
import re
import os
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="IMDB Sentiment - V2 (Optimized)", page_icon="⚡")

st.title("⚡ IMDB Movie Review Sentiment Analysis")
st.write("**Versi 2 (Optimasi TFLite):** Implementasi Quantization & TensorFlow Lite sesuai materi MLOps.")
st.markdown("---")

H5_PATH = 'sentiment_GRU_cfg1_seq100.h5'
TFLITE_PATH = 'sentiment_gru_optimized.tflite'

# ============================================================
# Fungsi konversi model ke TFLite (dijalankan sekali)
# ============================================================
def try_convert_tflite():
    if os.path.exists(TFLITE_PATH):
        return True
    if not os.path.exists(H5_PATH):
        return False
    try:
        keras_model = load_model(H5_PATH)
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        with open(TFLITE_PATH, 'wb') as f:
            f.write(tflite_model)
        return True
    except Exception as e:
        st.warning(f"Konversi TFLite gagal (akan pakai model .h5): {e}")
        return False

# ============================================================
# Load model dengan fallback ke H5 jika TFLite gagal
# ============================================================
@st.cache_resource
def load_resources():
    tokenizer = None
    if os.path.exists('tokenizer.pkl'):
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)

    use_tflite = False
    model = None
    interpreter = None

    # Coba TFLite dulu
    converted = try_convert_tflite()
    if converted and os.path.exists(TFLITE_PATH):
        try:
            interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
            interp.allocate_tensors()
            interpreter = interp
            use_tflite = True
        except Exception:
            pass

    # Fallback ke H5 jika TFLite gagal
    if not use_tflite and os.path.exists(H5_PATH):
        model = load_model(H5_PATH)

    return tokenizer, interpreter, model, use_tflite

def preprocess_text(text, tokenizer, max_len=100):
    text = text.lower()
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# ============================================================
# Load resources
# ============================================================
with st.spinner("Memuat model..."):
    tokenizer, interpreter, keras_model, use_tflite = load_resources()

# ============================================================
# Info Optimasi
# ============================================================
if os.path.exists(H5_PATH):
    h5_mb = os.path.getsize(H5_PATH) / (1024 * 1024)

    if os.path.exists(TFLITE_PATH):
        tfl_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
        saving = (1 - tfl_mb / h5_mb) * 100
        st.success(
            f"✅ **Model berhasil dioptimasi ke TFLite!** "
            f"Ukuran asli: `{h5_mb:.2f} MB` → Setelah Quantization: `{tfl_mb:.2f} MB` "
            f"(Hemat **{saving:.1f}%** ukuran)"
        )
        st.info(f"🔧 Mode inferensi: **{'TFLite (Optimized)' if use_tflite else 'Keras H5 (Fallback)'}**")
    else:
        st.info(f"ℹ️ Ukuran model H5: `{h5_mb:.2f} MB` | TFLite: Belum terkonversi.")

# ============================================================
# Tampilkan Penjelasan Teknik Optimasi
# ============================================================
with st.expander("📖 Teknik Optimasi yang Digunakan (Klik untuk baca)"):
    st.markdown("""
    **Dynamic Range Quantization (TFLite):**
    - Bobot model dikonversi dari **Float32 → Int8**, mengurangi ukuran model secara drastis.
    - Model tetap menghasilkan output Float32 saat inferensi.
    - Cocok untuk deployment di server resource-terbatas atau edge devices.
    
    > *"TensorFlow Lite adalah framework ringan dari TensorFlow yang dirancang untuk menjalankan model ML di edge devices"* — Materi Week 5
    """)

# ============================================================
# Prediksi
# ============================================================
if tokenizer is None or (interpreter is None and keras_model is None):
    st.error("❌ Model atau Tokenizer tidak ditemukan!")
else:
    user_input = st.text_area("Masukkan review film (Bahasa Inggris):", "This movie was fantastic!")

    if st.button("Analisis Sentimen"):
        if user_input.strip() == "":
            st.warning("Teks tidak boleh kosong.")
        else:
            with st.spinner("Menganalisis..."):
                processed = preprocess_text(user_input, tokenizer)
                start = time.time()

                if use_tflite and interpreter is not None:
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    interpreter.set_tensor(input_details[0]['index'], processed.astype(np.float32))
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
                else:
                    prediction = keras_model.predict(processed, verbose=0)[0][0]

                elapsed = time.time() - start

            st.subheader("Hasil Prediksi:")
            if prediction > 0.5:
                st.success(f"🌟 **POSITIF** — Confidence: {prediction:.2%} _(Waktu inferensi: {elapsed*1000:.1f}ms)_")
            else:
                st.error(f"💔 **NEGATIF** — Confidence: {(1-prediction):.2%} _(Waktu inferensi: {elapsed*1000:.1f}ms)_")
