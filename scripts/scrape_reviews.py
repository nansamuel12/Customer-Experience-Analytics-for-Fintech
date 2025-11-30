from pathlib import Path
from typing import Dict, List

import pandas as pd
from google_play_scraper import Sort, reviews


APPS: List[Dict[str, str]] = [
    {
        "bank_code": "CBE",
        "bank_name": "Commercial Bank of Ethiopia",
        "app_id": "com.combanketh.mobilebanking",
        "app_name": "Commercial Bank of Ethiopia",
    },
    {
        "bank_code": "BOA",
        "bank_name": "Bank of Abyssinia",
        "app_id": "com.boa.boaMobileBanking",
        "app_name": "BoA Mobile",
    },
    {
        "bank_code": "DASHEN",
        "bank_name": "Dashen Bank",
        "app_id": "com.dashen.dashensuperapp",
        "app_name": "Dashen Bank",
    },
]

# Target at least 400 reviews per bank; request a bit more to allow for drops
REVIEWS_PER_APP = 500


def fetch_reviews_for_app(app_config: Dict[str, str], count: int) -> pd.DataFrame:
    """Fetch reviews for a single Google Play app using google-play-scraper."""
    print(
        f"Fetching up to {count} reviews for {app_config['bank_name']} (app_id={app_config['app_id']})..."
    )

    all_reviews, _ = reviews(
        app_config["app_id"],
        lang="en",  # language of reviews
        country="et",  # Ethiopian Play Store
        sort=Sort.NEWEST,
        count=count,
    )

    df = pd.DataFrame(all_reviews)
    if df.empty:
        print(f"No reviews fetched for {app_config['bank_name']}.")
        return df

    df["bank"] = app_config["bank_code"]
    df["bank_name"] = app_config["bank_name"]
    df["app_id"] = app_config["app_id"]
    df["app_name"] = app_config["app_name"]
    df["source"] = "google_play"

    return df


def main() -> None:
    raw_dir = Path("data") / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []

    for app in APPS:
        try:
            df_app = fetch_reviews_for_app(app, REVIEWS_PER_APP)
        except Exception as exc:  # network / scraping issues
            print(f"Error while fetching reviews for {app['bank_name']}: {exc}")
            continue

        if df_app.empty:
            continue

        frames.append(df_app)
        per_bank_path = raw_dir / f"{app['bank_code'].lower()}_reviews_raw.csv"
        df_app.to_csv(per_bank_path, index=False)
        print(f"Saved {len(df_app)} raw reviews to {per_bank_path}")

    if not frames:
        print("No reviews fetched for any bank.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined_path = raw_dir / "bank_reviews_raw.csv"
    combined.to_csv(combined_path, index=False)

    print("\nCombined raw dataset summary:")
    print(f"Total rows: {len(combined)}")
    print("Rows per bank:")
    print(combined["bank"].value_counts())


if __name__ == "__main__":
    main()
