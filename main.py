import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("data.csv")

# Features
X = df["text"]
y = df["sentiment"]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = X, X, y, y

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print(classification_report(y_test, y_pred, zero_division=1))

# Test input
text = ["I hate this app"]
text_vec = vectorizer.transform(text)
print("\nPrediction:", model.predict(text_vec))