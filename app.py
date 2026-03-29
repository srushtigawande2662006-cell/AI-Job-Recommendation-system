import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import PyPDF2
import json
import os

# ------------------ Page Config ------------------
st.set_page_config(page_title="Smart AI Job Recommender", layout="wide")

# ------------------ CSS ------------------
st.markdown("""
<style>
.card {
    background: linear-gradient(135deg, #e3f2fd, #ffffff);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
    transition: 0.3s;
}
.card:hover { transform: scale(1.02); }

.dark-card {
    background: linear-gradient(135deg, #232526, #414345);
    color: white;
    border-radius: 15px;
    padding: 20px;
}

.tag-green {
    background-color: #4CAF50;
    color: white;
    padding: 4px 10px;
    border-radius: 6px;
    margin: 2px;
}
.tag-red {
    background-color: #f44336;
    color: white;
    padding: 4px 10px;
    border-radius: 6px;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ User DB ------------------
USER_DB = "users.json"

def load_users():
    if not os.path.exists(USER_DB):
        return {}
    with open(USER_DB, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f)

users = load_users()

# ------------------ Session Init ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Login"

# ------------------ Sidebar Navigation ------------------
if not st.session_state.logged_in:
    st.sidebar.title("🔐 Account")
    st.session_state.page = st.sidebar.radio("Select", ["Login", "Signup"])

# ------------------ LOGIN / SIGNUP ------------------
if not st.session_state.logged_in:

    # -------- Signup --------
    if st.session_state.page == "Signup":
        st.title("📝 Create Account")
        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")

        if st.button("Signup"):
            if new_user == "" or new_pass == "":
                st.warning("Please fill all fields")
            elif new_user in users:
                st.warning("User already exists")
            else:
                users[new_user] = new_pass
                save_users(users)
                st.success("Account created! Now login")

    # -------- Login --------
    elif st.session_state.page == "Login":
        st.title("🔐 Login")
        user = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if user in users and users[user] == password:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.success(f"Welcome {user} 🎉")
                st.rerun()
            else:
                st.error("Invalid username or password")

    st.stop()

# ------------------ Logout ------------------
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ------------------ Logo ------------------
try:
    st.image(Image.open("logo.jpg"), width=200)
except:
    st.warning("Add logo.jpg")

# ------------------ Dark Mode ------------------
theme = st.sidebar.checkbox("🌙 Dark Mode")

if theme:
    st.markdown("<style>body{background:#0e1117;color:white;}</style>", unsafe_allow_html=True)

# ------------------ Inputs ------------------
location = st.sidebar.selectbox("Location", ["Nagpur","Mumbai","Pune","Remote"])
skills_input = st.sidebar.text_input("Skills")
file = st.sidebar.file_uploader("Resume", type=["pdf","txt"])

# ------------------ Resume ------------------
resume_text = ""
if file:
    try:
        if file.type == "application/pdf":
            pdf = PyPDF2.PdfReader(file)
            for p in pdf.pages:
                resume_text += p.extract_text()
        else:
            resume_text = str(file.read(), "utf-8")
    except:
        st.warning("Resume error")

user_skills = skills_input + " " + resume_text

# ------------------ Data ------------------
df = pd.read_csv("jobs.csv")
df['skills'] = df['skills'].fillna('')

# ------------------ State ------------------
if "saved" not in st.session_state:
    st.session_state.saved = []

# ------------------ Title ------------------
st.title("💼 Smart AI Job Recommender")

# ------------------ Recommendation ------------------
if st.button("Find Jobs"):

    tfidf = TfidfVectorizer()
    mat = tfidf.fit_transform(df['skills'])
    user_vec = tfidf.transform([user_skills])

    sim = cosine_similarity(user_vec, mat)
    scores = sorted(list(enumerate(sim[0])), key=lambda x:x[1], reverse=True)

    for i in scores[:3]:
        idx = i[0]
        job = df.iloc[idx]['title']
        skills = df.iloc[idx]['skills']

        user_set = set(user_skills.lower().split(","))
        job_set = set(skills.lower().split(","))

        matched = job_set & user_set
        missing = job_set - user_set

        match_html = " ".join([f"<span class='tag-green'>{s}</span>" for s in matched])
        miss_html = " ".join([f"<span class='tag-red'>{s}</span>" for s in missing])

        st.markdown(f"""
        <div class="card">
        <h3>{job}</h3>
        <p><b>Matched:</b> {match_html}</p>
        <p><b>Missing:</b> {miss_html}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"[Apply](https://www.google.com/search?q= {job}+jobs+in+{location})")

        if st.button(f"Save {job}"):
            st.session_state.saved.append(job)

# ------------------ Saved ------------------
if st.session_state.saved:
    st.subheader("Saved Jobs")
    st.write(st.session_state.saved)