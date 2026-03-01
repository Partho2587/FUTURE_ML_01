from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

nltk.download('stopwords')

app = Flask(__name__)

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
job_description = pickle.load(open("job_desc.pkl", "rb"))
jd_skills = pickle.load(open("jd_skills.pkl", "rb"))

stop_words = set(stopwords.words('english'))

SKILLS_DB = [
    "python", "machine learning", "deep learning", "nlp",
    "sql", "data analysis", "pandas", "numpy",
    "tensorflow", "keras", "scikit-learn",
    "excel", "power bi", "tableau",
    "flask", "django", "aws", "docker",
    "communication", "leadership"
]

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def extract_skills(text):
    text = text.lower()
    found_skills = []
    for skill in SKILLS_DB:
        if skill in text:
            found_skills.append(skill)
    return list(set(found_skills))

def extract_pdf_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['resume']
    
    if file.filename.endswith(".pdf"):
        resume_text = extract_pdf_text(file)
    else:
        resume_text = file.read().decode("utf-8")

    cleaned_resume = clean_text(resume_text)

    resume_vector = vectorizer.transform([cleaned_resume])
    jd_vector = vectorizer.transform([clean_text(job_description)])

    similarity_score = cosine_similarity(jd_vector, resume_vector)[0][0]

    resume_skills = extract_skills(resume_text)
    matched_skills = list(set(resume_skills).intersection(set(jd_skills)))
    missing_skills = list(set(jd_skills) - set(resume_skills))

    skill_score = len(matched_skills) / len(jd_skills)

    final_score = 0.7 * similarity_score + 0.3 * skill_score

    if final_score > 0.5:
        result = "✅ Strong Match"
    elif final_score > 0.3:
        result = "⚠ Moderate Match"
    else:
        result = "❌ Weak Match"

    explanation = f"""
    This resume matches {round(final_score*100,2)}% of the job requirements.
    It covers {len(matched_skills)} out of {len(jd_skills)} key skills.
    """

    return render_template("index.html",
                           result=result,
                           score=round(final_score,2),
                           matched=matched_skills,
                           missing=missing_skills,
                           explanation=explanation)

if __name__ == "__main__":
    app.run(debug=True)
