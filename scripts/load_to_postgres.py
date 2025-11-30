import os
from pathlib import Path
from typing import Dict

import nltk
import pandas as pd
from dotenv import load_dotenv
from nltk.sentiment import SentimentIntensityAnalyzer
from sqlalchemy import create_engine, text


BASE_DIR = Path(__file__).resolve().parents[1]
CLEANED_PATH = BASE_DIR / "data" / "processed" / "bank_reviews_clean.csv"


BANK_META = {
    "CBE": {
        "bank_name": "Commercial Bank of Ethiopia",
        "app_name": "Commercial Bank of Ethiopia",
        "db_url_env": "DATABASE_URL_CBE",
    },
    "BOA": {
        "bank_name": "Bank of Abyssinia",
        "app_name": "BoA Mobile",
        "db_url_env": "DATABASE_URL_BOA",
    },
    "DASHEN": {
        "bank_name": "Dashen Bank",
        "app_name": "Dashen Bank",
        "db_url_env": "DATABASE_URL_DASHEN",
    },
}


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    scores = df["review"].fillna("").astype(str).apply(
        lambda txt: sia.polarity_scores(txt)["compound"]
    )
    df["sentiment_score"] = scores

    def label(compound: float) -> str:
        if compound > 0.05:
            return "positive"
        if compound < -0.05:
            return "negative"
        return "neutral"

    df["sentiment_label"] = df["sentiment_score"].apply(label)
    return df


def load_bank_to_db(bank_code: str, df_bank: pd.DataFrame, db_url: str) -> None:
    """Load a single bank's reviews into its dedicated database."""
    meta = BANK_META[bank_code]
    bank_name = meta["bank_name"]
    app_name = meta["app_name"]

    print(f"\nLoading {len(df_bank)} reviews for {bank_name} into {db_url.split('/')[-1]}...")

    engine = create_engine(db_url)

    with engine.begin() as conn:
        # Insert bank metadata (single row)
        banks_df = pd.DataFrame([{"bank_name": bank_name, "app_name": app_name}])
        banks_df.to_sql("banks", conn, if_exists="append", index=False)

        # Get the bank_id
        banks_lookup = pd.read_sql(
            text("SELECT bank_id, bank_name FROM banks WHERE bank_name = :name"),
            conn,
            params={"name": bank_name},
        )
        if banks_lookup.empty:
            raise RuntimeError(f"Failed to insert bank: {bank_name}")

        bank_id = banks_lookup.iloc[0]["bank_id"]

        # Prepare reviews
        reviews_df = df_bank[
            ["review", "rating", "date", "sentiment_label", "sentiment_score", "source"]
        ].copy()
        reviews_df["bank_id"] = bank_id
        reviews_df.rename(
            columns={"review": "review_text", "date": "review_date"}, inplace=True
        )
        reviews_df["review_date"] = pd.to_datetime(reviews_df["review_date"]).dt.date

        reviews_df.to_sql("reviews", conn, if_exists="append", index=False)

        # Verification
        count = pd.read_sql(
            text("SELECT COUNT(*) as count FROM reviews WHERE bank_id = :bid"),
            conn,
            params={"bid": bank_id},
        )
        avg_rating = pd.read_sql(
            text("SELECT AVG(rating) as avg_rating FROM reviews WHERE bank_id = :bid"),
            conn,
            params={"bid": bank_id},
        )
        print(f"  Inserted {count.iloc[0]['count']} reviews")
        print(f"  Average rating: {avg_rating.iloc[0]['avg_rating']:.2f}")


def main() -> None:
    load_dotenv()

    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned reviews file not found at {CLEANED_PATH}. Run preprocess_reviews.py first."
        )

    df = pd.read_csv(CLEANED_PATH)
    if df.empty:
        raise ValueError("Cleaned reviews dataframe is empty.")

    # Normalize bank codes
    df["bank"] = df["bank"].astype(str).str.upper()

    # Add sentiment analysis
    df = add_sentiment(df)

    # Process each bank separately
    for bank_code, meta in BANK_META.items():
        db_url_env = meta["db_url_env"]
        db_url = os.getenv(db_url_env)

        if not db_url:
            print(f"WARNING: {db_url_env} not set in .env, skipping {bank_code}")
            continue

        df_bank = df[df["bank"] == bank_code].copy()
        if df_bank.empty:
            print(f"WARNING: No reviews found for {bank_code}, skipping")
            continue

        try:
            load_bank_to_db(bank_code, df_bank, db_url)
        except Exception as exc:
            print(f"ERROR loading {bank_code}: {exc}")
            continue

    print("\n=== Load complete ===")


if __name__ == "__main__":
    main()
