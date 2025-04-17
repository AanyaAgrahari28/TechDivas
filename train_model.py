import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import joblib
import os


df = pd.read_csv("dataset/job_descriptions.csv")
df.columns = df.columns.str.strip()

print("👉 Column names:", df.columns)


required_columns = ['skills', 'Job Title']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset")

df = df.dropna(subset=required_columns)


print("Available columns:", df.columns)
df = df.dropna(subset=['skills', 'Job Title'])

vectorizer = TfidfVectorizer()
X_skills = df[('skills')]
X_skills_transformed = vectorizer.fit_transform(X_skills)


core_skills = ['Python', 'Java', 'SQL', 'C++', 'Machine Learning', 'Data Analysis', 'Communication', 'Leadership']

for skill in core_skills:
    df[skill] = df['skills'].apply(lambda x: 1 if skill.lower() in x.lower() else 0)


X = df[core_skills]
y = df['Job Title']


le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/job_model.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print(" Model trained and saved!")
import os
import joblib


os.makedirs("model", exist_ok=True)

# Save the model and vectorizer
joblib.dump(model, "model/job_model.pkl")
joblib.dump(vectorizer, "model/skill_vectorizer.pkl")
