import streamlit as st
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import sys
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))

from src.preprocessing import clean_text
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "fake_news_model.pkl")
VEC_PATH   = os.path.join(BASE_DIR, "..", "models", "tfidf_vectorizer.pkl")
from src.utils import load_model_and_vectorizer

model, vectorizer = load_model_and_vectorizer()
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)
stop_words = set(ENGLISH_STOP_WORDS)

def predict_news(news_text):
    if len(news_text.split()) < 80:
        st.warning("⚠️ Please enter at least 80 words for better prediction.")
        st.stop()
    cleaned = clean_text(news_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    fake_prob = float(proba[0])
    real_prob = float(proba[1])

    if fake_prob >= 0.70:
        label = "FAKE"
    elif real_prob >= 0.70:
        label = "REAL"
    else:
        label = "UNCERTAIN"

    confidence = max(fake_prob, real_prob)

    return label, confidence, fake_prob, real_prob

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.info("Note: This model predicts based on patterns learned from training data. It may not guarantee real-world truth.")
st.write("Paste a news article or headline below and click **Predict**.")

news_input = st.text_area("Enter news text:", height=200)

if st.button("Predict"):
    if len(news_input.split()) < 80:
        st.warning("⚠️ Please enter at least 30 words for better prediction.")
        st.stop()

    if news_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        label, confidence, fake_prob, real_prob = predict_news(news_input)
        st.subheader("Prediction Result:")
        st.write(f"### {label}")
        st.write(f"Confidence: **{confidence*100:.2f}%**")
        st.write(f"Fake Probability: **{fake_prob*100:.2f}%**")
        st.write(f"Real Probability: **{real_prob*100:.2f}%**")