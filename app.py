import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Fake Job Posting Detection Dashboard")

st.markdown("""
### Features

- Overview Dashboard
- Advanced Analytics
- Fraud Detection using Machine Learning
- Business Insights
- Interactive Visualizations

Select a page from the left sidebar.
""")

# --------------------------------------------------
# Page Config
# --------------------------------------------------



# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

[data-testid="metric-container"] {
    background-color: white;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}

h1 {
    color: #2563eb;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/fake_job_postings.csv")

try:
    df = load_data()
except:
    st.error("Dataset not found. Place fake_job_postings.csv inside data folder.")
    st.stop()

# --------------------------------------------------
# Dashboard Header
# --------------------------------------------------
st.title("💼 Fake Job Posting Detection Dashboard")
st.markdown("Analyze genuine and fraudulent job postings using interactive analytics.")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Filters")

if "employment_type" in df.columns:

    employment_filter = st.sidebar.multiselect(
        "Employment Type",
        df["employment_type"].dropna().unique()
    )

    if employment_filter:
        df = df[df["employment_type"].isin(employment_filter)]

# --------------------------------------------------
# KPI Section
# --------------------------------------------------
total_jobs = len(df)

fake_jobs = (
    int(df["fraudulent"].sum())
    if "fraudulent" in df.columns
    else 0
)

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
# Dataset Preview
# --------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Fraud Distribution
# --------------------------------------------------
if "fraudulent" in df.columns:

    st.subheader("Fraud Distribution")

    fraud_df = (
        df["fraudulent"]
        .value_counts()
        .reset_index()
    )

    fraud_df.columns = ["Status", "Count"]

    fraud_df["Status"] = fraud_df["Status"].replace({
        0: "Real",
        1: "Fake"
    })

    fig = px.pie(
        fraud_df,
        names="Status",
        values="Count",
        hole=0.4,
        title="Real vs Fake Jobs"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Employment Type Analysis
# --------------------------------------------------
if "employment_type" in df.columns:

    st.subheader("Employment Type Analysis")

    emp = (
        df["employment_type"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )

    emp.columns = ["Employment Type", "Count"]

    fig = px.bar(
        emp,
        x="Employment Type",
        y="Count",
        color="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Experience Level Analysis
# --------------------------------------------------
if "required_experience" in df.columns:

    st.subheader("Experience Level Analysis")

    exp = (
        df["required_experience"]
        .fillna("Unknown")
        .value_counts()
        .head(10)
        .reset_index()
    )

    exp.columns = ["Experience", "Count"]

    fig = px.bar(
        exp,
        x="Experience",
        y="Count",
        color="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Education Analysis
# --------------------------------------------------
if "required_education" in df.columns:

    st.subheader("Education Requirement Analysis")

    edu = (
        df["required_education"]
        .fillna("Unknown")
        .value_counts()
        .head(10)
        .reset_index()
    )

    edu.columns = ["Education", "Count"]

    fig = px.bar(
        edu,
        x="Education",
        y="Count",
        color="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Top Industries
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
# Top Functions
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
# Telecommuting Analysis
# --------------------------------------------------
if "telecommuting" in df.columns:

    st.subheader("Remote Job Analysis")

    tele = (
        df["telecommuting"]
        .value_counts()
        .reset_index()
    )

    tele.columns = ["Remote", "Count"]

    fig = px.bar(
        tele,
        x="Remote",
        y="Count",
        color="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Correlation Heatmap
# --------------------------------------------------
numeric_df = df.select_dtypes(include=["int64", "float64"])

if len(numeric_df.columns) > 1:

    st.subheader("Correlation Heatmap")

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Summary Statistics
# --------------------------------------------------
st.subheader("Summary Statistics")
st.dataframe(df.describe(include="all"))
Advanced Analytics Page

Add these charts:

1. Fraud by Employment Type
px.histogram(
    df,
    x="employment_type",
    color="fraudulent",
    barmode="group"
)
2. Fraud by Experience Level
px.histogram(
    df,
    x="required_experience",
    color="fraudulent"
)
3. Fraud by Industry
industry = df["industry"].value_counts().head(15)
4. Fraud by Function
function = df["function"].value_counts().head(15)
5. Correlation Heatmap
numeric = df.select_dtypes(include="number")

corr = numeric.corr()

fig = px.imshow(
    corr,
    text_auto=True
)
6. Word Cloud

Most common words in fake jobs:

from wordcloud import WordCloud

Use:

df[df["fraudulent"]==1]["title"]
Fraud Detection Page

Train model:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

Features:

title
description
requirements
benefits

Model:

model = LogisticRegression()

Prediction form:

job_text = st.text_area(
    "Paste Job Description"
)

if st.button("Check Fraud"):

    prediction = model.predict(...)

Output:

st.success("Likely Genuine")

or

st.error("Potential Fake Job")
Streamlit UI Enhancements

Add:

st.markdown("""
<style>

.main {
    background:#F8FAFC;
}

[data-testid="metric-container"]{
    border-radius:12px;
    padding:15px;
    box-shadow:0 0 10px rgba(0,0,0,0.1);
}

</style>
""",unsafe_allow_html=True)
