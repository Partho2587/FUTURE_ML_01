# ==========================================
# SMART RESUME SCREENING SYSTEM
# ==========================================

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# LOAD DATASET
# ==========================================
df = pd.read_csv(r"C:\Users\parth\Downloads\archive (25)\Resume\Resume.csv")

# Print columns once to verify
print("Available Columns:", df.columns)

# ==========================================
# JOB DESCRIPTION
# ==========================================
job_description = """
Looking for a Machine Learning Engineer with experience in Python,
Scikit-learn, NLP, Data Analysis, Deep Learning, and Flask.
"""

# ==========================================
# TF-IDF VECTORIZATION
# ==========================================
tfidf = TfidfVectorizer(stop_words='english')

# 🔥 FIXED COLUMN NAME
documents = list(df['Resume_str']) + [job_description]

tfidf_matrix = tfidf.fit_transform(documents)

# ==========================================
# COSINE SIMILARITY
# ==========================================
cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

df['Similarity Score'] = cosine_sim.flatten()

# ==========================================
# SORT TOP CANDIDATES
# ==========================================
top_candidates = df.sort_values(by='Similarity Score', ascending=False).head(5)

# ==========================================
# OUTPUT RESULTS
# ==========================================
print("\n🏆 TOP 5 CANDIDATES")
print("=" * 40)

for index, row in top_candidates.iterrows():
    print(f"📄 Category: {row['Category']}")
    print(f"📊 Similarity Score: {round(row['Similarity Score'], 4)}")
    print("-" * 40)

print("✅ Screening Completed Successfully!")