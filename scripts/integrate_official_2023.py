"""
Prepare an official 2023 NCRB file for drop-in use by the dashboard backend.

Usage:
    python scripts/integrate_official_2023.py --input path\\to\\official_file.csv

Output:
    Creates ./official-crime-data-2023.csv in the project root.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "STATE": "state_name",
        "State": "state_name",
        "STATE_NAME": "state_name",
        "DISTRICT": "district_name",
        "District": "district_name",
        "DISTRICT_NAME": "district_name",
        "YEAR": "year",
        "Year": "year",
        "UT": "state_name",
    }
    df = df.rename(columns=rename_map).copy()
    df.columns = [str(col).strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns]
    return df


def read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize official NCRB 2023 file for CrimeScope.")
    parser.add_argument("--input", required=True, help="Path to official 2023 CSV/XLSX")
    parser.add_argument("--output", default="official-crime-data-2023.csv", help="Output CSV path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = normalize_columns(read_input(input_path))
    required = {"state_name", "district_name", "year"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"] == 2023].copy()
    if df.empty:
        raise ValueError("No 2023 rows found in the provided file.")

    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(f"Saved normalized official 2023 data to: {output_path.resolve()}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
