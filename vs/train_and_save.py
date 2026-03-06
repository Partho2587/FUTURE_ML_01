# train_and_save.py
import pandas as pd
import string
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
nltk.download('wordnet')

# Load data
df1 = pd.read_csv(r"C:\Users\parth\OneDrive\Documents\Fake.csv")
df2 = pd.read_csv(r"C:\Users\parth\OneDrive\Documents\True.csv")
df = pd.concat([df1, df2], ignore_index=True)
df['result'] = ['fake' if i <= 23503 else 'real' for i in range(len(df))]

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['text'] = df['text'].apply(clean_text)
df['title'] = df['title'].apply(clean_text)
df['label'] = df['result'].map({'fake': 0, 'real': 1})

# Train 2 models and vectorizers: one for title, one for text
models = {}
vectorizers = {}

for field in ['title', 'text']:
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df[field])
    y = df['label']
    
    model = LogisticRegression()
    model.fit(X, y)

    models[field] = model
    vectorizers[field] = vectorizer

# Save both
with open("model.pkl", "wb") as f:
    pickle.dump(models, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizers, f)

print("✅ Models and vectorizers saved for both title and text.")
