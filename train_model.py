import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

# ---------------- Load Dataset ----------------
df = pd.read_csv(
    "data/UpdatedResumeDataSet.csv",
    encoding="utf-8",
    sep=",",
    engine="python",
    quoting=3,          # Ignore broken quotes in resume text
    on_bad_lines="skip"
)

# ---------------- Cleaning Function ----------------
def cleanResume(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub('[^a-zA-Z ]', ' ', text)
    text = text.lower()
    return text

df["cleaned_resume"] = df["Resume"].apply(cleanResume)

# ---------------- Encode Labels ----------------
le = LabelEncoder()
df["Category"] = le.fit_transform(df["Category"])

# ---------------- TF-IDF ----------------
tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(df["cleaned_resume"])
y = df["Category"]

# ---------------- Model ----------------
model = OneVsRestClassifier(KNeighborsClassifier())
model.fit(X, y)

# ---------------- Save Model ----------------
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))
pickle.dump(le, open("model/label_encoder.pkl", "wb"))

print(" Model trained and saved successfully")
