from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Load saved models
type_model = joblib.load("type_model.pkl")
priority_model = joblib.load("priority_model.pkl")
type_encoder = joblib.load("type_encoder.pkl")
priority_encoder = joblib.load("priority_encoder.pkl")


# SIMPLE CLEANING (NO NLTK)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        subject = request.form["subject"]
        description = request.form["description"]

        combined = subject + " " + description
        cleaned = clean_text(combined)

        type_pred = type_model.predict([cleaned])[0]
        priority_pred = priority_model.predict([cleaned])[0]

        category = type_encoder.inverse_transform([type_pred])[0]
        priority = priority_encoder.inverse_transform([priority_pred])[0]

        return render_template(
            "index.html",
            category=category,
            priority=priority
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)