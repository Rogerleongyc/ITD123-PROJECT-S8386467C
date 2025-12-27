import streamlit as st
import tensorflow as tf
from tensorflow import keras

st.set_page_config(page_title="Kindle Sentiment Classifier", page_icon="📚")
st.title("📚 Kindle Reviews Sentiment Classifier")
st.write("Paste a review and the model will predict **Positive / Negative / Uncertain**.")

@st.cache_resource
def load_model():
    return keras.models.load_model("kindle_sentiment_model.keras")

model = load_model()

pos_threshold = st.slider("Positive threshold", 0.50, 0.95, 0.60, 0.01)
neg_threshold = st.slider("Negative threshold", 0.05, 0.50, 0.40, 0.01)

review = st.text_area("Enter review text:", height=160)

def predict_one(text: str):
    x = tf.constant([text], dtype=tf.string)
    x = tf.expand_dims(x, axis=1)
    prob = float(model.predict(x, verbose=0)[0][0])

    if prob >= pos_threshold:
        label = "✅ Positive"
    elif prob <= neg_threshold:
        label = "❌ Negative"
    else:
        label = "😶 Uncertain"
    return label, prob

if st.button("Predict"):
    if not review.strip():
        st.warning("Please paste a review first 🙂")
    else:
        label, prob = predict_one(review.strip())
        st.subheader(f"Prediction: {label}")
        st.write(f"Probability (positive): **{prob:.3f}**")

