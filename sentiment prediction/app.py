# app.py
from flask import Flask, render_template, request
import pickle
import re
import numpy as np

app = Flask(__name__)

# ============================
# Load Models
# ============================
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
sentiment_model = pickle.load(open("model/sentiment_model.pkl", "rb"))
age_model = pickle.load(open("model/age_model.pkl", "rb"))
sentiment_enc = pickle.load(open("model/sentiment_encoder.pkl", "rb"))
age_enc = pickle.load(open("model/age_encoder.pkl", "rb"))

feature_names = vectorizer.get_feature_names_out()

# ============================
# Cleaning
# ============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ============================
# Explainability (STABLE)
# ============================
def explain_sentiment(text, top_n=8):
    X = vectorizer.transform([clean_text(text)])
    pred = sentiment_model.predict(X)[0]

    coef = sentiment_model.coef_[pred]
    top_idx = np.argsort(coef)[-top_n:]

    return [feature_names[i] for i in top_idx]

# ============================
# Route
# ============================
@app.route("/", methods=["GET", "POST"])
def index():
    result = {}

    if request.method == "POST":
        text = request.form["text"]
        X = vectorizer.transform([clean_text(text)])

        # Sentiment
        s_probs = sentiment_model.predict_proba(X)[0]
        s_idx = np.argmax(s_probs)

        # Age
        a_probs = age_model.predict_proba(X)[0]
        a_idx = np.argmax(a_probs)

        result = {
            "sentiment": sentiment_enc.inverse_transform([s_idx])[0],
            "sentiment_conf": round(s_probs[s_idx] * 100, 2),
            "age_group": age_enc.inverse_transform([a_idx])[0],
            "age_conf": round(a_probs[a_idx] * 100, 2),
            "explanation": explain_sentiment(text),
            "text": text
        }

    return render_template("index.html", **result)

if __name__ == "__main__":
    app.run(debug=True)
