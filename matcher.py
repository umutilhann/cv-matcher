import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import streamlit as st

nlp = spacy.load("en_core_web_sm")

# Yaygın teknik beceri listesi (eşleşen becerileri bulmak için)
SKILL_KEYWORDS = [
    "python", "sql", "excel", "power bi", "tableau", "r", "java", "javascript",
    "machine learning", "deep learning", "nlp", "tensorflow", "pytorch", "keras",
    "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn",
    "etl", "data cleaning", "data visualization", "data analysis", "data modeling",
    "statistics", "regression", "classification", "clustering",
    "aws", "azure", "gcp", "docker", "git", "linux",
    "mysql", "postgresql", "mongodb", "spark", "hadoop",
    "communication", "teamwork", "project management", "agile", "scrum",
]

@st.cache_data
def load_and_prepare_data():
    """Veriyi yükle ve TF-IDF matrisini hazırla (cache'lenir, bir kez çalışır)"""
    df = pd.read_csv("data.csv")
    df["Description"] = df["Description"].fillna("")

    descriptions_clean = [clean_text(d) for d in df["Description"].tolist()]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(descriptions_clean)

    return df, vectorizer, tfidf_matrix, descriptions_clean

def clean_text(text):
    """spaCy ile metni temizle ve lemmatize et"""
    doc = nlp(text.lower()[:10000])  # çok uzun metinleri kırp
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 2
    ]
    return " ".join(tokens)

def extract_matching_skills(cv_text, job_description):
    """CV ile iş ilanı arasındaki ortak becerileri bul"""
    cv_lower = cv_text.lower()
    job_lower = job_description.lower()
    matched = [skill for skill in SKILL_KEYWORDS
               if skill in cv_lower and skill in job_lower]
    return matched

def find_matches(cv_text, top_n=5):
    """CV metnini analiz et ve en uygun ilanları döndür"""
    df, vectorizer, tfidf_matrix, _ = load_and_prepare_data()

    cv_clean = clean_text(cv_text)
    cv_vector = vectorizer.transform([cv_clean])

    scores = cosine_similarity(cv_vector, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_n]

    results = []
    for i in top_indices:
        matched_skills = extract_matching_skills(cv_text, df.iloc[i]["Description"])
        results.append({
            "title": df.iloc[i]["Job Title"],
            "score": round(float(scores[i]) * 100, 1),
            "description": df.iloc[i]["Description"][:400] + "...",
            "matched_skills": matched_skills,
        })
    return results