import streamlit as st
import pandas as pd

df = pd.read_csv(
    "data/fake_job_postings.csv"
)

st.title("📋 Overview")

col1,col2,col3 = st.columns(3)

col1.metric(
    "Total Jobs",
    len(df)
)

col2.metric(
    "Fake Jobs",
    int(df["fraudulent"].sum())
)

col3.metric(
    "Real Jobs",
    len(df)-int(df["fraudulent"].sum())
)

st.dataframe(df.head())
