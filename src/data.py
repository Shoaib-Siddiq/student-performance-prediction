"""Data loading and feature engineering for the Student Performance dataset.

Uses the classic UCI Student Performance dataset (Cortez & Silva, 2008):
- student-mat.csv  : math course (395 students)
- student-por.csv  : Portuguese course (649 students)

The two files are integrated (inner join on the 30 non-grade attributes,
averaging the duplicated grade columns), reproducing the canonical merge
from the original study -> 382 unique students.
"""
from pathlib import Path

import pandas as pd

# Merge key from the original study (student-merge.R):
# "school","sex","age","address","famsize","Pstatus","Medu","Fedu",
#  "Mjob","Fjob","reason","nursery","internet"  ->  382 unique students
MERGE_KEY = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "nursery", "internet",
]

GRADE_COLS = ["G1", "G2", "G3"]


def load_raw(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean the raw math and Portuguese course CSVs."""
    data_dir = Path(data_dir)

    def _read(name: str) -> pd.DataFrame:
        df = pd.read_csv(data_dir / name, sep=";", quotechar='"')
        # grades arrive as strings; coerce to numeric
        for col in GRADE_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    math = _read("student-mat.csv")
    por = _read("student-por.csv")
    return math, por


def integrate(math: pd.DataFrame, por: pd.DataFrame) -> pd.DataFrame:
    """Merge math + Portuguese datasets on the key attributes.

    Mirrors the canonical merge in the original study (student-merge.R):
    grade pairs for the same student are averaged, keeping one row per
    student. All other attributes come from the math record.
    """
    merged = pd.merge(math, por, on=MERGE_KEY, suffixes=("_math", "_por"))
    df = merged[MERGE_KEY].copy()
    for col in GRADE_COLS:
        df[col] = (merged[f"{col}_math"] + merged[f"{col}_por"]) / 2
    # keep the remaining (non-key) attributes from the math file
    other = [c for c in math.columns if c not in MERGE_KEY + GRADE_COLS]
    for col in other:
        df[col] = merged[f"{col}_math"]
    return df


def add_targets(df: pd.DataFrame, pass_threshold: int = 10) -> pd.DataFrame:
    """Add derived targets: final grade G3 and a pass/fail binary label."""
    df = df.copy()
    df["G3"] = df["G3"].round(0)
    df["passed"] = (df["G3"] >= pass_threshold).astype(int)
    return df


def load_dataset(data_dir: str | Path = "data", pass_threshold: int = 10) -> pd.DataFrame:
    """End-to-end loader: raw files -> integrated, target-ready dataset."""
    math, por = load_raw(data_dir)
    return add_targets(integrate(math, por), pass_threshold)
