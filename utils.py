import re
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_resume(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = text.lower()
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def ats_match(resume_text, jd_text):
    resume_words = set(clean_resume(resume_text).split())
    jd_words = set(clean_resume(jd_text).split())

    matched = resume_words.intersection(jd_words)
    score = (len(matched) / len(jd_words)) * 100 if jd_words else 0

    return round(score, 2), list(matched)
