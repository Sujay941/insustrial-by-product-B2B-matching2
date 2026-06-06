import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/fake_job_postings.csv"
)

st.title("📈 Insights")

fake_jobs = df[
    df["fraudulent"] == 1
]

text = " ".join(
    fake_jobs["title"]
    .fillna("")
)

wordcloud = WordCloud(
    width=800,
    height=400
).generate(text)

fig, ax = plt.subplots()

ax.imshow(wordcloud)

ax.axis("off")

st.pyplot(fig)

st.subheader(
    "Key Insights"
)

st.write("""
• Fake jobs are concentrated in specific industries.

• Some employment types have higher fraud rates.

• Word cloud highlights common scam keywords.

• Machine learning model can classify jobs as genuine or fraudulent.
""")
