"""
preprocess.py
Cleans the raw datasets and prepares them for model training.

Inputs (expected in the same folder):
    - db_drug_interactions.csv
    - Medicine_Details.csv
    - medicines.csv

Outputs:
    - cleaned_interactions.csv
    - cleaned_medicines.csv
"""

import pandas as pd
import re

# ---------------------------------------------------------------------------
# 1. Drug-drug interactions -> derive severity from description text
# ---------------------------------------------------------------------------

# NOTE: this dataset uses DrugBank-style phrasing (e.g. "The risk or severity
# of adverse effects can be increased...", "The serum concentration of X can
# be decreased...") rather than plain words like "severe" or "avoid" - so the
# keyword lists below are tuned to match how THIS dataset actually talks.
HIGH_RISK_KEYWORDS = [
    "risk or severity of adverse effects", "qtc-prolonging",
    "arrhythmogenic", "central nervous system depressant", "hypotensive"
]
MEDIUM_RISK_KEYWORDS = [
    "serum concentration", "metabolism", "excretion rate",
    "therapeutic efficacy", "absorption"
]
# anything that doesn't match the above falls back to "Low"


def assign_severity(description: str) -> str:
    """Keyword-based severity rule. Checked in priority order: High > Medium > Low."""
    text = description.lower()

    if any(keyword in text for keyword in HIGH_RISK_KEYWORDS):
        return "High"
    if any(keyword in text for keyword in MEDIUM_RISK_KEYWORDS):
        return "Medium"
    return "Low"


def process_interactions(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    df = df.dropna(subset=["Drug 1", "Drug 2", "Interaction Description"])
    df = df.drop_duplicates()

    # Normalize drug names (trim spaces, consistent casing for matching later)
    df["Drug 1"] = df["Drug 1"].str.strip().str.lower()
    df["Drug 2"] = df["Drug 2"].str.strip().str.lower()

    df["Severity"] = df["Interaction Description"].apply(assign_severity)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} cleaned interaction rows -> {output_path}")
    print(df["Severity"].value_counts())


# ---------------------------------------------------------------------------
# 2. Medicine details -> clean structured medicine info
# ---------------------------------------------------------------------------

def process_medicine_details(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    df = df.dropna(subset=["Medicine Name", "Composition"])
    df = df.drop_duplicates(subset=["Medicine Name"])

    df["Medicine Name"] = df["Medicine Name"].str.strip()
    df["Composition"] = df["Composition"].str.strip()

    # Keep only the columns the app actually needs
    keep_cols = ["Medicine Name", "Composition", "Uses", "Side_effects", "Manufacturer"]
    df = df[keep_cols]

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} cleaned medicine rows -> {output_path}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    process_interactions(
        input_path="db_drug_interactions.csv",
        output_path="cleaned_interactions.csv"
    )

    process_medicine_details(
        input_path="Medicine_Details.csv",
        output_path="cleaned_medicines.csv"
    )
