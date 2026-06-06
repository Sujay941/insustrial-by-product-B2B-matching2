import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv(
    "data/fake_job_postings.csv"
)

st.title("📊 Analytics")

# Fraud by Employment Type

fig = px.histogram(
    df,
    x="employment_type",
    color="fraudulent",
    barmode="group"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# Fraud by Experience

fig = px.histogram(
    df,
    x="required_experience",
    color="fraudulent"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# Industry

industry = (
    df["industry"]
    .fillna("Unknown")
    .value_counts()
    .head(15)
)

fig = px.bar(
    x=industry.index,
    y=industry.values
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# Function

function = (
    df["function"]
    .fillna("Unknown")
    .value_counts()
    .head(15)
)

fig = px.bar(
    x=function.index,
    y=function.values
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# Heatmap

numeric = df.select_dtypes(
    include="number"
)

corr = numeric.corr()

fig = px.imshow(
    corr,
    text_auto=True
)

st.plotly_chart(
    fig,
    use_container_width=True
)
