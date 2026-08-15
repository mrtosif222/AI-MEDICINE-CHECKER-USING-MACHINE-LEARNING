"""
preprocess.py
Cleans the raw datasets and prepares them for model training.

Inputs (expected in the same folder):
    - db_drug_interactions.csv
    - Medicine_Details.csv
    - medicines.csv
    - who_atc_ddd.csv       (WHO ATC/DDD drug classification)
    - drug_finder.csv       (Generic Name -> Drug Class reference)

Outputs:
    - cleaned_interactions.csv   (now includes Category1 / Category2 columns)
    - cleaned_medicines.csv
    - drug_category_map.csv      (drug name -> pharmacological category)
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


# ---------------------------------------------------------------------------
# 1b. Build a drug -> pharmacological category map (ATC + Drug Finder)
# ---------------------------------------------------------------------------

def build_category_map(atc_path: str, drug_finder_path: str, output_path: str) -> dict:
    """Builds a lookup of generic drug name -> category name.

    Source 1 (ATC): level-5 codes (7 characters) are individual chemical
    substances; their level-4 parent code (5 characters) gives the
    pharmacological subgroup name (e.g. "Beta blocking agents").

    Source 2 (Drug Finder): a smaller reference that gives an explicit
    "Drug Class" per generic name, used as a fallback where ATC has no match.
    """
    atc = pd.read_csv(atc_path)
    atc["code_len"] = atc["atc_code"].str.len()

    level5 = atc[atc["code_len"] == 7][["atc_code", "atc_name"]].copy()
    level5["drug_name"] = level5["atc_name"].str.lower().str.strip()
    level5["level4_code"] = level5["atc_code"].str[:5]

    level4 = atc[atc["code_len"] == 5][["atc_code", "atc_name"]].rename(
        columns={"atc_code": "level4_code", "atc_name": "category"}
    )
    atc_map = level5.merge(level4, on="level4_code", how="left")[["drug_name", "category"]]
    atc_map = atc_map.drop_duplicates(subset="drug_name")

    finder = pd.read_csv(drug_finder_path)
    finder["drug_name"] = finder["Generic Name"].str.lower().str.strip()
    finder_map = finder[["drug_name", "Drug Class"]].rename(columns={"Drug Class": "category"})
    finder_map = finder_map.drop_duplicates(subset="drug_name")

    # ATC first (broader coverage), Drug Finder fills in anything ATC missed
    combined = pd.concat([atc_map, finder_map]).drop_duplicates(subset="drug_name", keep="first")
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} drug -> category mappings -> {output_path}")

    return combined.set_index("drug_name")["category"].to_dict()


def process_interactions(input_path: str, output_path: str, category_map: dict) -> None:
    df = pd.read_csv(input_path)

    df = df.dropna(subset=["Drug 1", "Drug 2", "Interaction Description"])
    df = df.drop_duplicates()

    # Normalize drug names (trim spaces, consistent casing for matching later)
    df["Drug 1"] = df["Drug 1"].str.strip().str.lower()
    df["Drug 2"] = df["Drug 2"].str.strip().str.lower()

    df["Severity"] = df["Interaction Description"].apply(assign_severity)

    # Pharmacological category for each drug (helps the model generalize
    # beyond just memorizing exact drug-name spellings)
    df["Category1"] = df["Drug 1"].map(category_map).fillna("unknown")
    df["Category2"] = df["Drug 2"].map(category_map).fillna("unknown")

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} cleaned interaction rows -> {output_path}")
    print(df["Severity"].value_counts())
    matched = (df["Category1"] != "unknown").sum()
    print(f"Category matched for Drug 1 in {matched}/{len(df)} rows")


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
    category_map = build_category_map(
        atc_path="who_atc_ddd.csv",
        drug_finder_path="drug_finder.csv",
        output_path="drug_category_map.csv"
    )

    process_interactions(
        input_path="db_drug_interactions.csv",
        output_path="cleaned_interactions.csv",
        category_map=category_map
    )

    process_medicine_details(
        input_path="Medicine_Details.csv",
        output_path="cleaned_medicines.csv"
    )
