# model.py
import pandas as pd
import re
import pickle
import os
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# ============================
# Load Dataset
# ============================
df = pd.read_csv(
    r"C:\Users\parth\OneDrive\Desktop\sentiment prediction\train.csv",
    encoding="latin1"
)

df = df.drop(columns=[
    'textID',
    'selected_text',
    'Time of Tweet',
    'Population -2020',
    'Land Area (Km²)',
    'Density (P/Km²)'
]).dropna().reset_index(drop=True)

# ============================
# Age Processing
# ============================
def extract_age(age):
    nums = re.findall(r'\d+', str(age))
    if not nums:
        return None
    if len(nums) == 1:
        return int(nums[0])
    return int((int(nums[0]) + int(nums[1])) / 2)

df['age_numeric'] = df['Age of User'].apply(extract_age)
df['age_numeric'].fillna(df['age_numeric'].median(), inplace=True)

def age_group(age):
    if age < 18:
        return "Minor"
    elif age <= 60:
        return "Major"
    return "Senior Citizen"

df['age_group'] = df['age_numeric'].apply(age_group)
df.drop(columns=['Age of User', 'age_numeric'], inplace=True)

# ============================
# Text Cleaning
# ============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

df['clean_text'] = df['text'].apply(clean_text)

# ============================
# Encoding
# ============================
sentiment_enc = LabelEncoder()
age_enc = LabelEncoder()

y_sentiment = sentiment_enc.fit_transform(df['sentiment'])
y_age = age_enc.fit_transform(df['age_group'])

# ============================
# Vectorization
# ============================
vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 6),
    min_df=3,
    max_features=40000
)

X = vectorizer.fit_transform(df['clean_text'])

# ============================
# Train-Test Split
# ============================
X_train, X_test, yS_train, yS_test = train_test_split(
    X, y_sentiment, test_size=0.2, stratify=y_sentiment, random_state=42
)

_, _, yA_train, yA_test = train_test_split(
    X, y_age, test_size=0.2, stratify=y_age, random_state=42
)

# ============================
# MODELS
# ============================
sentiment_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

age_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

sentiment_model.fit(X_train, yS_train)
age_model.fit(X_train, yA_train)

print("Sentiment Accuracy:", accuracy_score(yS_test, sentiment_model.predict(X_test)))
print("Age Accuracy:", accuracy_score(yA_test, age_model.predict(X_test)))

# ============================
# Save
# ============================
os.makedirs("model", exist_ok=True)

pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
pickle.dump(sentiment_model, open("model/sentiment_model.pkl", "wb"))
pickle.dump(age_model, open("model/age_model.pkl", "wb"))
pickle.dump(sentiment_enc, open("model/sentiment_encoder.pkl", "wb"))
pickle.dump(age_enc, open("model/age_encoder.pkl", "wb"))

print("✅ Models saved successfully")
