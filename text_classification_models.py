# -*- coding: utf-8 -*-
"""Text classification with Linear Regression, Logistic Regression,
Random Forest, and SVM using the 20 Newsgroups dataset.

Note: Linear Regression is included to compare with the original script,
but Logistic Regression, Random Forest, and SVM are more appropriate
choices for a classification task.
"""

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# 1. Load the same two categories as the original script
categories = ["rec.sport.baseball", "sci.space"]

data = fetch_20newsgroups(
    subset="all",
    categories=categories,
    remove=("headers", "footers", "quotes"),
)

X = data.data
y = data.target

print("Total documents:", len(X))
print("Classes:", data.target_names)

# 2. Split the data while preserving the class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. TF-IDF turns text into numeric features.
# Fit the vectorizer only on training text to avoid data leakage.
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)

# 4. Compare several models on the same TF-IDF features.
# Linear Regression is retained to match the original code; it is not
# generally recommended for classification.
models = {
    "Linear Regression (baseline)": LinearRegression(),
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
    "Linear SVM": LinearSVC(random_state=42, class_weight="balanced"),
}

results = []
fitted_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)

    if name == "Linear Regression (baseline)":
        # Convert continuous regression output to class labels.
        y_pred = (model.predict(X_test_tfidf) >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "F1 Score (%)": round(f1 * 100, 2),
    })
    fitted_models[name] = model

    print(f"\n{name} Performance")
    print("-" * (len(name) + 12))
    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, target_names=data.target_names, zero_division=0
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# 5. Show a side-by-side comparison of model accuracy and other metrics.
results_df = pd.DataFrame(results).sort_values(
    by="Accuracy (%)", ascending=False
)
print("\n\nMODEL COMPARISON")
print("================")
print(results_df.to_string(index=False))

# 6. Predict the class of new text using the selected model.
# Change SELECTED_MODEL to any key from the models dictionary.
SELECTED_MODEL = "Linear SVM"
selected_model = fitted_models[SELECTED_MODEL]

new_text = [
    "The team played an excellent baseball game",
    "NASA launched a spacecraft into orbit",
]

new_text_tfidf = vectorizer.transform(new_text)
predicted_classes = selected_model.predict(new_text_tfidf)

print(f"\nNEW TEXT PREDICTIONS (using {SELECTED_MODEL})")
print("=" * 45)
for text, prediction in zip(new_text, predicted_classes):
    print("\nText:", text)
    print("Predicted class:", data.target_names[int(prediction)])
