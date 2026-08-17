"""
medicine_lookup.py
Looks up full details for a medicine name typed/spoken/OCR'd by the user.

Matching strategy (in order):
    1. Exact match on Medicine Name.
    2. Partial match on Medicine Name (handles dosage variants like
       "Telmikind Beta 40mg" vs "Telmikind Beta 20mg").
    3. Composition/generic-name match - many brand names are missing from
       the dataset, but their generic composition (e.g. "Telmisartan") is
       usually present in some other brand's Composition field, so this is
       the fallback that gives the widest coverage.

If nothing matches, the caller (explainer.py) can fall back to an
LLM-generated general answer, clearly labelled as not dataset-verified.
"""

import pandas as pd
import re
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDICINES_PATH = os.path.join(_BASE_DIR, "..", "data", "cleaned_medicines.csv")

_df = None  # lazy-loaded so importing this module doesn't hit disk immediately


def _load():
    global _df
    if _df is None:
        _df = pd.read_csv(MEDICINES_PATH)
        _df["Medicine Name"] = _df["Medicine Name"].astype(str)
        _df["Composition"] = _df["Composition"].astype(str)
    return _df


def _strip_dosage(name: str) -> str:
    """Removes numbers/mg/tablet/injection etc. so 'Telmikind Beta 40mg
    Tablet' becomes 'telmikind beta' for looser matching."""
    name = name.lower()
    name = re.sub(r"\d+(\.\d+)?\s*(mg|ml|mcg|g)?", "", name)
    name = re.sub(r"\b(tablet|injection|capsule|syrup|drop)s?\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


def find_medicine(query: str) -> dict | None:
    """Returns a dict with medicine details, or None if nothing is found."""
    df = _load()
    query_clean = _strip_dosage(query)

    # 1. Exact match
    exact = df[df["Medicine Name"].str.lower() == query.lower()]
    if not exact.empty:
        return _row_to_dict(exact.iloc[0], match_type="exact")

    # 2. Partial / dosage-insensitive match on medicine name
    partial = df[df["Medicine Name"].apply(_strip_dosage).str.contains(
        re.escape(query_clean), na=False
    )]
    if not partial.empty:
        return _row_to_dict(partial.iloc[0], match_type="partial_name")

    # 3. Composition/generic-name match (widest net)
    if query_clean:
        composition_match = df[df["Composition"].str.lower().str.contains(
            re.escape(query_clean), na=False
        )]
        if not composition_match.empty:
            return _row_to_dict(composition_match.iloc[0], match_type="composition")

    return None


def _row_to_dict(row, match_type: str) -> dict:
    return {
        "name": row["Medicine Name"],
        "composition": row["Composition"],
        "uses": row.get("Uses", ""),
        "side_effects": row.get("Side_effects", ""),
        "manufacturer": row.get("Manufacturer", ""),
        "match_type": match_type,  # exact / partial_name / composition
    }


if __name__ == "__main__":
    # quick manual test
    for test_query in ["Augmentin 625 Duo Tablet", "Telmikind Beta 40mg", "Bevacizumab"]:
        result = find_medicine(test_query)
        print(f"\nQuery: {test_query}")
        print(result if result else "No match found")
