import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Fake Job Detection Dashboard",
    page_icon="💼",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main{
    background:#F8FAFC;
}

[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:15px;
    box-shadow:0 0 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("fake_job_postings.csv.zip")

try:
    df = load_data()
except Exception as e:
    st.error(f"Dataset Error: {e}")
    st.stop()

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("💼 Fake Job Posting Detection Dashboard")
st.markdown("Analyze genuine and fraudulent job postings using interactive analytics.")

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

total_jobs = len(df)

fake_jobs = int(df["fraudulent"].sum()) if "fraudulent" in df.columns else 0

real_jobs = total_jobs - fake_jobs

companies = (
    df["company_profile"].notna().sum()
    if "company_profile" in df.columns
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Jobs", f"{total_jobs:,}")
col2.metric("Fake Jobs", f"{fake_jobs:,}")
col3.metric("Real Jobs", f"{real_jobs:,}")
col4.metric("Companies", f"{companies:,}")

st.divider()

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

st.subheader("Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# FRAUD DISTRIBUTION
# --------------------------------------------------

if "fraudulent" in df.columns:

    st.subheader("Fraud Distribution")

    fraud_df = df["fraudulent"].value_counts().reset_index()
    fraud_df.columns = ["Status", "Count"]

    fraud_df["Status"] = fraud_df["Status"].replace({
        0: "Real",
        1: "Fake"
    })

    fig = px.pie(
        fraud_df,
        names="Status",
        values="Count",
        hole=0.4
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# FRAUD BY EMPLOYMENT TYPE
# --------------------------------------------------

if "employment_type" in df.columns and "fraudulent" in df.columns:

    st.subheader("Fraud by Employment Type")

    fig = px.histogram(
        df,
        x="employment_type",
        color="fraudulent",
        barmode="group"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# FRAUD BY EXPERIENCE LEVEL
# --------------------------------------------------

if "required_experience" in df.columns and "fraudulent" in df.columns:

    st.subheader("Fraud by Experience Level")

    fig = px.histogram(
        df,
        x="required_experience",
        color="fraudulent"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# TOP INDUSTRIES
# --------------------------------------------------

if "industry" in df.columns:

    st.subheader("Top Industries")

    industry = (
        df["industry"]
        .fillna("Unknown")
        .value_counts()
        .head(15)
        .reset_index()
    )

    industry.columns = ["Industry", "Count"]

    fig = px.bar(
        industry,
        x="Industry",
        y="Count",
        color="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# TOP FUNCTIONS
# --------------------------------------------------

if "function" in df.columns:

    st.subheader("Top Job Functions")

    function = (
        df["function"]
        .fillna("Unknown")
        .value_counts()
        .head(15)
        .reset_index()
    )

    function.columns = ["Function", "Count"]

    fig = px.bar(
        function,
        x="Function",
        y="Count",
        color="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------

numeric_df = df.select_dtypes(include="number")

if len(numeric_df.columns) > 1:

    st.subheader("Correlation Heatmap")

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# WORD CLOUD
# --------------------------------------------------

if "fraudulent" in df.columns and "title" in df.columns:

    st.subheader("Most Common Words in Fake Jobs")

    text = " ".join(
        df[df["fraudulent"] == 1]["title"]
        .fillna("")
        .astype(str)
    )

    if text.strip():

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")

        st.pyplot(fig)

# --------------------------------------------------
# FRAUD DETECTION MODEL
# --------------------------------------------------

if all(col in df.columns for col in [
    "title",
    "description",
    "requirements",
    "benefits",
    "fraudulent"
]):

    st.subheader("Fraud Detection")

    train_df = df.copy()

    train_df["text"] = (
        train_df["title"].fillna("") + " " +
        train_df["description"].fillna("") + " " +
        train_df["requirements"].fillna("") + " " +
        train_df["benefits"].fillna("")
    )

    X = train_df["text"]
    y = train_df["fraudulent"]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_vec, y)

    job_text = st.text_area(
        "Paste Job Description Here"
    )

    if st.button("Check Fraud"):

        if job_text.strip() == "":
            st.warning("Please enter job description.")
        else:

            text_vec = vectorizer.transform([job_text])

            prediction = model.predict(text_vec)[0]

            if prediction == 0:
                st.success("Likely Genuine Job Posting")
            else:
                st.error("Potential Fake Job Posting")

# --------------------------------------------------
# SUMMARY STATISTICS
# --------------------------------------------------

st.subheader("Summary Statistics")

try:
    st.dataframe(df.describe(include="all"))
except:
    st.write("Statistics not available.")
