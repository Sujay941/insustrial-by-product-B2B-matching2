import streamlit as st
import joblib

model = joblib.load(
    "model.pkl"
)

vectorizer = joblib.load(
    "vectorizer.pkl"
)

st.title("🤖 Fraud Detection")

job_text = st.text_area(
    "Paste Job Description"
)

if st.button("Check Fraud"):

    vec = vectorizer.transform(
        [job_text]
    )

    prediction = model.predict(
        vec
    )[0]

    if prediction == 0:

        st.success(
            "Likely Genuine Job Posting"
        )

    else:

        st.error(
            "Potential Fake Job Posting"
        )
