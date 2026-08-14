"""Interactive Streamlit demo for Student Performance Prediction.

Run:  streamlit run app/app.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

st.title("🎓 Student Performance Predictor")
st.caption("Predict a student's final grade (G3) and pass/fail outcome from social & school attributes — "
           "trained on the UCI Student Performance dataset (382 students, no prior-grade leakage).")

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "classification_rf.joblib")

pipe = load_model()

with st.sidebar:
    st.header("Student profile")
    school = st.selectbox("School", ["GP", "MS"])
    sex = st.selectbox("Sex", ["M", "F"])
    age = st.slider("Age", 15, 22, 17)
    address = st.selectbox("Home address type", ["U" if s == "Urban" else "R" for s in ["Urban", "Rural"]])
    famsize = st.selectbox("Family size", ["GT3", "LE3"])
    pstatus = st.selectbox("Parent's cohabitation status", ["A", "T"])
    medu = st.slider("Mother's education (0=none, 4=higher)", 0, 4, 2)
    fedu = st.slider("Father's education (0=none, 4=higher)", 0, 4, 2)
    mjob = st.selectbox("Mother's job", ["teacher", "health", "services", "at_home", "other"])
    fjob = st.selectbox("Father's job", ["teacher", "health", "services", "at_home", "other"])
    reason = st.selectbox("Reason to choose school", ["home", "reputation", "course", "other"])
    traveltime = st.slider("Travel time to school (1=<15min ... 4=>1h)", 1, 4, 2)
    studytime = st.slider("Weekly study time (1=<2h ... 4=>10h)", 1, 4, 2)
    failures = st.slider("Past class failures", 0, 4, 0)
    schoolsup = st.selectbox("Extra educational support", ["yes", "no"])
    famsup = st.selectbox("Family educational support", ["yes", "no"])
    paid = st.selectbox("Extra paid classes", ["yes", "no"])
    activities = st.selectbox("Extra-curricular activities", ["yes", "no"])
    nursery = st.selectbox("Attended nursery school", ["yes", "no"])
    higher = st.selectbox("Wants to take higher education", ["yes", "no"])
    internet = st.selectbox("Internet access at home", ["yes", "no"])
    romantic = st.selectbox("In a romantic relationship", ["yes", "no"])
    famrel = st.slider("Family relationship quality (1=very bad, 5=excellent)", 1, 5, 4)
    freetime = st.slider("Free time after school (1=very low, 5=very high)", 1, 5, 3)
    goout = st.slider("Going out with friends (1=very low, 5=very high)", 1, 5, 3)
    dalc = st.slider("Weekday alcohol consumption (1=very low, 5=very high)", 1, 5, 1)
    walc = st.slider("Weekend alcohol consumption (1=very low, 5=very high)", 1, 5, 2)
    health = st.slider("Current health status (1=very bad, 5=very good)", 1, 5, 3)
    absences = st.slider("Number of school absences", 0, 93, 4)
    predict = st.button("Predict", type="primary", use_container_width=True)

row = {
    "school": school, "sex": sex, "age": age, "address": address, "famsize": famsize,
    "Pstatus": pstatus, "Medu": medu, "Fedu": fedu, "Mjob": mjob, "Fjob": fjob,
    "reason": reason, "traveltime": traveltime, "studytime": studytime, "failures": failures,
    "schoolsup": schoolsup, "famsup": famsup, "paid": paid, "activities": activities,
    "nursery": nursery, "higher": higher, "internet": internet, "romantic": romantic,
    "famrel": famrel, "freetime": freetime, "goout": goout, "Dalc": dalc, "Walc": walc,
    "health": health, "absences": absences,
}

if predict:
    X = pd.DataFrame([row])
    prob = pipe.predict_proba(X)[0, 1]
    pred = int(pipe.predict(X)[0])

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Prediction", "✅ Likely to PASS" if pred == 1 else "❌ Likely to FAIL",
                  delta=None)
    with c2:
        st.metric("Probability of passing", f"{prob:.1%}")
    st.progress(float(prob))

    if prob > 0.7:
        st.success("This student profile looks strong — keep it up!")
    elif prob > 0.4:
        st.warning("Borderline profile. More study time and fewer absences would help.")
    else:
        st.error("At-risk profile — intervention recommended (study time, attendance, support).")
