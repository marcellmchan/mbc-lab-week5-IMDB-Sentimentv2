import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import pickle
import re
import os

st.set_page_config(page_title="Versi 2 - Multi Model", page_icon="🚀", layout="wide")

@st.cache_resource
def load_sentiment_model():
    if os.path.exists('sentiment_GRU_cfg1_seq100.h5'): return load_model('sentiment_GRU_cfg1_seq100.h5')
    return None

@st.cache_resource
def load_image_model():
    if os.path.exists('model_pretrained_mobilenetv2.h5'): return load_model('model_pretrained_mobilenetv2.h5')
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
    return pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

st.sidebar.title("Navigasi Versi 2")
menu = st.sidebar.radio("Pilih Halaman:", ["Klasifikasi Gambar Apple vs Orange", "Analisis Sentimen IMDB"])

if menu == "Klasifikasi Gambar Apple vs Orange":
    st.title("🍎🍊 Klasifikasi Gambar: Apple vs Orange")
    img_model = load_image_model()
    
    if img_model is None:
        st.error("Model gambar tidak ditemukan.")
    else:
        uploaded_file = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            with st.spinner("Menganalisis..."):
                img_resized = image.resize((128, 128))
                img_array = preprocess_input(np.expand_dims(img_to_array(img_resized), axis=0))
                prediction = img_model.predict(img_array)[0][0]
                if prediction > 0.5:
                    st.success(f"🍊 JERUK (Orange) - {prediction:.2%}")
                else:
                    st.info(f"🍎 APEL (Apple) - {(1-prediction):.2%}")

elif menu == "Analisis Sentimen IMDB":
    st.title("🎬 Analisis Sentimen Ulasan Film (IMDB)")
    sentiment_model = load_sentiment_model()
    tokenizer = get_tokenizer()

    if sentiment_model is None or tokenizer is None:
        st.error("Model/Tokenizer tidak ditemukan.")
    else:
        user_input = st.text_area("Masukkan ulasan film:", "This movie was fantastic!")
        if st.button("Analisis Sentimen"):
            processed_text = preprocess_text(user_input, tokenizer)
            prediction = sentiment_model.predict(processed_text)[0][0]
            if prediction > 0.5:
                st.success(f"🌟 POSITIF ({prediction:.2%})")
            else:
                st.error(f"💔 NEGATIF ({(1-prediction):.2%})")
