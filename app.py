import streamlit as st
import numpy as np
import pickle
import re
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="IMDB Sentiment - V2 (Optimized)", page_icon="⚡")

st.title("⚡ IMDB Movie Review Sentiment Analysis")
st.write("**Versi 2 (Optimasi TFLite):** Implementasi Quantization & TensorFlow Lite untuk mengurangi ukuran model sesuai kriteria MLOps.")
st.markdown("---")

H5_PATH = 'sentiment_GRU_cfg1_seq100.h5'
TFLITE_PATH = 'sentiment_gru_optimized.tflite'

@st.cache_resource
def get_optimized_model():
    # Convert on the fly if TFLite doesn't exist
    if not os.path.exists(TFLITE_PATH) and os.path.exists(H5_PATH):
        with st.spinner("Mengoptimasi model ke TFLite (Hanya berjalan sekali)..."):
            model = tf.keras.models.load_model(H5_PATH)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            with open(TFLITE_PATH, 'wb') as f:
                f.write(tflite_model)
                
    if os.path.exists(TFLITE_PATH):
        interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        interpreter.allocate_tensors()
        return interpreter
    return None

@st.cache_resource
def get_tokenizer():
    if os.path.exists('tokenizer.pkl'):
        with open('tokenizer.pkl', 'rb') as handle:
            return pickle.load(handle)
    return None

def preprocess_text(text, tokenizer, max_len=100):
    text = text.lower()
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post').astype(np.float32)

interpreter = get_optimized_model()
tokenizer = get_tokenizer()

# Tampilkan metrik optimasi
if os.path.exists(H5_PATH) and os.path.exists(TFLITE_PATH):
    h5_size = os.path.getsize(H5_PATH) / (1024 * 1024)
    tflite_size = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    st.info(f"💡 **Info Optimasi Model:** Ukuran asli `.h5` ({h5_size:.2f} MB) ➔ Ukuran optimasi `.tflite` ({tflite_size:.2f} MB). Hemat ruang penyimpanan hingga {(1 - tflite_size/h5_size):.2%}!")

if interpreter is None or tokenizer is None:
    st.error("Model atau Tokenizer tidak ditemukan!")
else:
    user_input = st.text_area("Masukkan review film (Bahasa Inggris):", "This movie was fantastic!")
    
    if st.button("Analisis Sentimen"):
        if user_input.strip() != "":
            with st.spinner("Menganalisis dengan TFLite..."):
                processed_text = preprocess_text(user_input, tokenizer)
                
                # Inferensi TFLite
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                interpreter.set_tensor(input_details[0]['index'], processed_text)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
                
                st.subheader("Hasil Prediksi:")
                if prediction > 0.5:
                    st.success(f"🌟 POSITIF (Confidence: {prediction:.2%})")
                else:
                    st.error(f"💔 NEGATIF (Confidence: {(1-prediction):.2%})")
