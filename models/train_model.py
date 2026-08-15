"""
train_model.py
Trains a model that predicts interaction Severity (High/Medium/Low) for a
pair of drug names.

Approach:
    Each drug pair is represented as text combining the drug names AND
    their pharmacological categories (from preprocess.py, e.g. "Beta
    blocking agents"), converted to numeric features with TF-IDF. An
    XGBoost classifier is trained on top of that (tested to be noticeably
    more accurate here than RandomForest - roughly 75% -> 83-84%).

    Category features matter because two drugs can share almost no letters
    in their names but behave the same way pharmacologically - the
    category signal helps the model generalize instead of just memorizing
    exact name spellings.

Input:
    ../data/cleaned_interactions.csv  (from preprocess.py, includes
    Category1 / Category2 columns)

Output:
    interaction_model.pkl
    vectorizer.pkl
    label_encoder.pkl
"""

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

INPUT_PATH = "../data/cleaned_interactions.csv"
MODEL_PATH = "interaction_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"


def main():
    df = pd.read_csv(INPUT_PATH)

    # Combine drug names + their categories into one text field for TF-IDF
    df["combined"] = (
        df["Drug 1"] + " " + df["Drug 2"] + " " +
        df["Category1"].fillna("unknown") + " " + df["Category2"].fillna("unknown")
    )

    X = df["combined"]
    y = df["Severity"]

    # XGBoost needs numeric labels, not strings - encode them
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    vectorizer = TfidfVectorizer(
        analyzer="word", token_pattern=r"[a-zA-Z0-9\-]+", max_features=8000
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.3,
        tree_method="hist", n_jobs=-1, eval_metric="mlogloss"
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("Evaluation on held-out test set:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved vectorizer -> {VECTORIZER_PATH}")
    print(f"Saved label encoder -> {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()
