"""
interaction_checker.py
Given two medicine names, finds out whether they interact and how severe
that interaction is.

Strategy (in order):
    1. Try to find the exact pair (in either order) in the cleaned
       interactions dataset - this gives the most trustworthy answer,
       including the real description text.
    2. If the pair isn't in the dataset, fall back to the trained ML model
       (interaction_model.pkl) to estimate a severity.
    3. Medicine names are first resolved to their generic/composition name
       via medicine_lookup.py, since interactions are recorded by generic
       name, not brand name.
"""

import pandas as pd
import joblib
import re
import os
from medicine_lookup import find_medicine

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTERACTIONS_PATH = os.path.join(_BASE_DIR, "..", "data", "cleaned_interactions.csv")
MODEL_PATH = os.path.join(_BASE_DIR, "..", "models", "interaction_model.pkl")
VECTORIZER_PATH = os.path.join(_BASE_DIR, "..", "models", "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(_BASE_DIR, "..", "models", "label_encoder.pkl")
CATEGORY_MAP_PATH = os.path.join(_BASE_DIR, "..", "data", "drug_category_map.csv")

_interactions_df = None
_model = None
_vectorizer = None
_label_encoder = None
_category_map = None


def _load():
    global _interactions_df, _model, _vectorizer, _label_encoder, _category_map
    if _interactions_df is None:
        _interactions_df = pd.read_csv(INTERACTIONS_PATH)
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
        _label_encoder = joblib.load(LABEL_ENCODER_PATH)
    if _category_map is None:
        _category_map = pd.read_csv(CATEGORY_MAP_PATH).set_index("drug_name")["category"].to_dict()
    return _interactions_df, _model, _vectorizer, _label_encoder, _category_map


def _to_generic_name(medicine_name: str) -> str:
    """Resolves a brand name (e.g. 'Augmentin 625 Duo Tablet') to its
    generic composition name (e.g. 'amoxycillin'), which is what the
    interactions dataset actually uses."""
    match = find_medicine(medicine_name)
    if match:
        # Composition can contain multiple drugs, e.g. "Amoxycillin (500mg) + Clavulanic Acid (125mg)"
        # Take the first ingredient name as the primary generic name.
        composition = match["composition"]
        first_ingredient = re.split(r"\+", composition)[0]
        first_ingredient = re.sub(r"\(.*?\)", "", first_ingredient).strip().lower()
        return first_ingredient
    return medicine_name.strip().lower()


def check_interaction(medicine_a: str, medicine_b: str) -> dict:
    df, model, vectorizer, label_encoder, category_map = _load()

    generic_a = _to_generic_name(medicine_a)
    generic_b = _to_generic_name(medicine_b)

    # 1. Exact pair lookup (either order)
    match = df[
        ((df["Drug 1"] == generic_a) & (df["Drug 2"] == generic_b)) |
        ((df["Drug 1"] == generic_b) & (df["Drug 2"] == generic_a))
    ]
    if not match.empty:
        row = match.iloc[0]
        return {
            "medicine_a": medicine_a,
            "medicine_b": medicine_b,
            "interaction_found": True,
            "severity": row["Severity"],
            "description": row["Interaction Description"],
            "source": "dataset",
        }

    # 2. ML model fallback (uses drug names + their pharmacological categories)
    cat_a = category_map.get(generic_a, "unknown")
    cat_b = category_map.get(generic_b, "unknown")
    combined = f"{generic_a} {generic_b} {cat_a} {cat_b}"

    vec = vectorizer.transform([combined])
    predicted_encoded = model.predict(vec)[0]
    predicted_severity = label_encoder.inverse_transform([predicted_encoded])[0]
    confidence = model.predict_proba(vec).max()

    return {
        "medicine_a": medicine_a,
        "medicine_b": medicine_b,
        "interaction_found": None,  # unknown - not in dataset
        "severity": predicted_severity,
        "description": "This exact pair was not found in the verified dataset. "
                        "This severity is an ML estimate based on similar drugs.",
        "source": "model_estimate",
        "confidence": round(float(confidence), 2),
    }


if __name__ == "__main__":
    # quick manual test
    result = check_interaction("Trioxsalen", "Verteporfin")
    print(result)

    result2 = check_interaction("Augmentin 625 Duo Tablet", "Telmikind Beta 40mg")
    print(result2)
