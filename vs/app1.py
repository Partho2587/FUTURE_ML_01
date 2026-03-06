from flask import Flask, request, render_template_string
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load models and vectorizers
with open('model.pkl', 'rb') as f:
    models = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizers = pickle.load(f)

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# HTML Template with input choice
html = r'''
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fake News Detector</title>
    <style>
        body {
            font-family: 'Georgia', serif;
            background-color: #eeeeee;
            padding: 40px;
            margin: 0;
        }
        .container {
            max-width: 800px;
            margin: auto;
            background: #ffffff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
        h2 {
            text-align: center;
            font-size: 28px;
            color: #000000;
        }
        label {
            font-size: 18px;
        }
        textarea {
            width: 100%;
            padding: 12px;
            margin-top: 10px;
            border-radius: 8px;
            border: 1px solid #ccc;
            resize: none;
            background-color: #ffffff;
            color: red;
            font-size: 16px;
        }
        .radio-group {
            margin: 15px 0;
        }
        .radio-group label {
            margin-right: 20px;
            font-weight: bold;
        }
        input[type=submit], .refresh-btn {
            padding: 12px 20px;
            font-size: 16px;
            margin-top: 20px;
            margin-right: 10px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            background-color: #2c3e50;
            color: #fff;
        }
        .refresh-btn {
            background-color: #7f8c8d;
            text-decoration: none;
        }
        .result {
            margin-top: 25px;
            font-size: 20px;
            text-align: center;
            font-weight: bold;
            background-color: #dfe6e9;
            padding: 15px;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>📰 Fake News Detection App</h2>
        <form method="post">
            <div class="radio-group">
                <label>
                    <input type="radio" name="choice" value="title"
                    {% if request.form.choice == 'title' %}checked{% endif %}> Title
                </label>
                <label>
                    <input type="radio" name="choice" value="text"
                    {% if request.form.choice == 'text' %}checked{% endif %}> Text
                </label>
            </div>

            <textarea name="input_text" placeholder="Enter news content..." rows="10">{{ request.form.input_text or '' }}</textarea><br>

            <input type="submit" value="Predict">
            <a href="/" class="refresh-btn">Refresh</a>
        </form>

        {% if prediction %}
            <div class="result">🔍 Prediction: {{ prediction }}</div>
        {% endif %}
    </div>
</body>
</html>

'''
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        choice = request.form['choice']
        user_input = request.form['input_text']
        cleaned = clean_text(user_input)
        vec = vectorizers[choice].transform([cleaned])
        result = models[choice].predict(vec)[0]
        prediction = "REAL ✅" if result == 1 else "FAKE ❌"
    return render_template_string(html, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True,port=5001)

