import streamlit as st
import pickle
from utils import clean_resume, extract_text_from_pdf, ats_match

# ---------------- Load Models ----------------
model = pickle.load(open("model/model.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
le = pickle.load(open("model/label_encoder.pkl", "rb"))

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>

/* ===== REMOVE EMPTY BOX UNDER TABS ===== */
div[data-testid="stTabs"] + div {
    display: none !important;
}

/* Remove tab underline & spacing */
div[data-testid="stTabs"] hr {
    display: none;
}
div[data-testid="stTabs"] > div {
    padding-top: 0rem;
}

/* ===== BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #020617, #020617);
}

/* ===== CENTER TITLE ===== */
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    color: #facc15;
    margin-bottom: 20px;
}

/* ===== GLOBAL TEXT COLOR ===== */
.stApp, p, span, label, div {
    color: #f8fafc !important;
}

/* Headings */
h1, h2, h3, h4 {
    color: #56b1fc !important;
}

/* ===== TEXT AREAS ===== */
textarea {
    background-color: #020617 !important;
    color: white !important;
    border-radius: 12px;
    border: 1px solid #334155;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] section {
    background-color: #020617 !important;
    border-radius: 12px;
    border: 1px solid #334155;
}

[data-testid="stFileUploader"] button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}

[data-testid="stFileUploader"] button:hover {
    background-color: #1e40af !important;
}

/* ===== ACTION BUTTONS ===== */
.stButton > button {
    background-color: #b50202;
    color: black;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 22px;
    border: none;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #16a34a;
}

/* ===== CARD LAYOUT ===== */
.card {
    background-color: #020617;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #334155;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown(
    '<div class="main-title">📄 RESUME SCREENING SYSTEM </div>',
    unsafe_allow_html=True
)

# ---------------- Tabs ----------------
tab1, tab2 = st.tabs(["👤 Candidate View", "🧑‍💼 HR / Recruiter View"])

# =====================================================
# TAB 1: Candidate View
# =====================================================
with tab1:

    st.subheader("🔍 Resume Category Prediction")

    resume_text = st.text_area("✍️ Paste Resume Text", height=200)
    resume_file = st.file_uploader(
        "📂 OR Upload Resume (.txt / .pdf)",
        type=["txt", "pdf"]
    )

    final_text = ""

    if resume_file:
        if resume_file.type == "application/pdf":
            final_text = extract_text_from_pdf(resume_file)
        else:
            final_text = resume_file.read().decode("utf-8")
    elif resume_text:
        final_text = resume_text

    if st.button("🔎 Predict Job Category"):
        if final_text.strip() == "":
            st.warning("Please paste or upload a resume.")
        else:
            cleaned = clean_resume(final_text)
            vector = tfidf.transform([cleaned])
            prediction = model.predict(vector)
            category = le.inverse_transform(prediction)[0]

            st.success(f"Predicted Resume Category: **{category}**")

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# TAB 2: HR / Recruiter View
# =====================================================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 ATS Resume Matching")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📝 Job Description")
        jd_text = st.text_area("Paste Job Description", height=180)
        jd_file = st.file_uploader(
            "OR Upload JD (.txt / .pdf)",
            type=["txt", "pdf"],
            key="jd"
        )

    with col2:
        st.markdown("### 📄 Candidate Resume")
        resume_hr_text = st.text_area("Paste Resume", height=180)
        resume_hr_file = st.file_uploader(
            "OR Upload Resume (.txt / .pdf)",
            type=["txt", "pdf"],
            key="resume_hr"
        )

    if st.button("📈 Check ATS Match"):
        # ---- Job Description ----
        if jd_file:
            if jd_file.type == "application/pdf":
                jd_final = extract_text_from_pdf(jd_file)
            else:
                jd_final = jd_file.read().decode("utf-8")
        else:
            jd_final = jd_text

        # ---- Resume ----
        if resume_hr_file:
            if resume_hr_file.type == "application/pdf":
                resume_final = extract_text_from_pdf(resume_hr_file)
            else:
                resume_final = resume_hr_file.read().decode("utf-8")
        else:
            resume_final = resume_hr_text

        if jd_final.strip() == "" or resume_final.strip() == "":
            st.warning("Please provide both Job Description and Resume.")
        else:
            score, keywords = ats_match(resume_final, jd_final)

            st.success("✅ ATS Matching Completed")
            st.progress(score / 100)

            st.markdown(
                f"<h2 style='color:#22c55e;'>ATS Match Score: {score}%</h2>",
                unsafe_allow_html=True
            )

            st.markdown("### ⭐ Matched Keywords")
            st.write(", ".join(keywords[:40]))

    st.markdown('</div>', unsafe_allow_html=True)
