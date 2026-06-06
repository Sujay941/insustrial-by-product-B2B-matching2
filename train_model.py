import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load Dataset
df = pd.read_csv("data/fake_job_postings.csv")

# Create Text Feature
df["text"] = (
    df["title"].fillna("") +
    " " +
    df["description"].fillna("") +
    " " +
    df["requirements"].fillna("") +
    " " +
    df["benefits"].fillna("")
)

X = df["text"]
y = df["fraudulent"]

# Vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000
)

X_vec = vectorizer.fit_transform(X)

# Model
model = LogisticRegression(
    max_iter=1000
)

model.fit(X_vec, y)

# Save Files
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved Successfully")
