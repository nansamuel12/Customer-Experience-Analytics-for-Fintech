from pathlib import Path

import pandas as pd


RAW_PATH = Path("data") / "raw" / "bank_reviews_raw.csv"
PROCESSED_DIR = Path("data") / "processed"
OUTPUT_PATH = PROCESSED_DIR / "bank_reviews_clean.csv"


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw reviews file not found at {RAW_PATH}. Run scrape_reviews.py first."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    original_rows = len(df)

    # Drop duplicate reviews using the most specific identifier available
    subset_cols = [
        col
        for col in ["reviewId", "content", "score", "at"]
        if col in df.columns
    ]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols)
    else:
        df = df.drop_duplicates()

    # Map raw columns to standardized names
    if "content" in df.columns:
        df["review"] = df["content"]

    if "score" in df.columns:
        df["rating"] = df["score"]

    if "at" in df.columns:
        df["date"] = pd.to_datetime(df["at"], errors="coerce").dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Ensure required columns exist
    required_cols = ["review", "rating", "date", "bank"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns after mapping: {missing_required}")

    # Handle missing data by dropping rows with critical nulls
    before_missing = len(df)
    df = df.dropna(subset=required_cols)
    after_missing = len(df)

    removed_missing = before_missing - after_missing
    missing_rate = (
        (removed_missing / before_missing) * 100 if before_missing > 0 else 0.0
    )

    # Normalize date format to YYYY-MM-DD (as string)
    df["date"] = df["date"].astype(str)

    # Ensure source column exists
    if "source" not in df.columns:
        df["source"] = "google_play"

    final_cols = ["review", "rating", "date", "bank", "source"]
    df_out = df[final_cols].copy()

    df_out.to_csv(OUTPUT_PATH, index=False)

    print("Preprocessing summary:")
    print(f"Original rows: {original_rows}")
    print(f"After deduplication and missing handling: {len(df_out)}")
    print(f"Rows removed due to missing data: {removed_missing} ({missing_rate:.2f}% of deduplicated set)")
    print("Rows per bank (cleaned):")
    print(df_out["bank"].value_counts())


if __name__ == "__main__":
    main()
