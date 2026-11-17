"""
import kagglehub

# Download latest version
path = kagglehub.dataset_download("steve1215rogg/student-lifestyle-dataset")

print("Path to dataset files:", path)
"""
from pathlib import Path
from importlib.metadata import version
import sys
import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

df = pd.read_csv("student_lifestyle_dataset.csv")

features = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
]
target = "GPA"

print("Shape:", df.shape)
print(df[features + [target]].head())
print("Missing values:\n", df[features + [target]].isna().sum())
print("Duplicate rows:", df.duplicated().sum())

data = df.drop_duplicates().dropna(subset=[target])

X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target], test_size=0.2, random_state=42
)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    LinearRegression()
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Test MAE:", mean_absolute_error(y_test, predictions))

joblib.dump(model, "model.joblib")

packages = ["pandas", "numpy", "scipy", "scikit-learn", "joblib"]
lines = ["streamlit"] + [f"{p}=={version(p)}" for p in packages]
Path("requirements.txt").write_text("\n".join(lines) + "\n")

print("Training Python version:", sys.version.split()[0])