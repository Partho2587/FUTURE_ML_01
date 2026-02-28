import pandas as pd
import re
import nltk
import joblib
import warnings
warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# NLTK
try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except:
    nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load dataset
df = pd.read_csv("customer_support_tickets.csv")

df["text"] = (
    df["Ticket Subject"].astype(str) + " " +
    df["Ticket Description"].astype(str)
)

df = df[["text", "Ticket Type", "Ticket Priority"]].dropna()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    words = text.split()
    words = [
        lemmatizer.lemmatize(w)
        for w in words
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

type_encoder = LabelEncoder()
priority_encoder = LabelEncoder()

df["type_label"] = type_encoder.fit_transform(df["Ticket Type"])
df["priority_label"] = priority_encoder.fit_transform(df["Ticket Priority"])

X_train, X_test, y_type_train, y_type_test, y_priority_train, y_priority_test = train_test_split(
    df["clean_text"],
    df["type_label"],
    df["priority_label"],
    test_size=0.2,
    random_state=42
)

tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

type_pipeline = Pipeline([
    ("tfidf", tfidf),
    ("model", LinearSVC(class_weight="balanced"))
])

priority_pipeline = Pipeline([
    ("tfidf", tfidf),
    ("model", LinearSVC(class_weight="balanced"))
])

type_pipeline.fit(X_train, y_type_train)
priority_pipeline.fit(X_train, y_priority_train)

# Save models
joblib.dump(type_pipeline, "type_model.pkl")
joblib.dump(priority_pipeline, "priority_model.pkl")
joblib.dump(type_encoder, "type_encoder.pkl")
joblib.dump(priority_encoder, "priority_encoder.pkl")

print("Models saved successfully!")