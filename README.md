# Customer Experience Analytics for Fintech (Ethiopian Banks)

This project analyzes customer satisfaction with mobile banking apps for three Ethiopian banks using Google Play Store reviews:

- Commercial Bank of Ethiopia (CBE)
- Bank of Abyssinia (BoA)
- Dashen Bank

The workflow covers:

- Task 1: Data collection and preprocessing from Google Play
- Task 2–3: Storing cleaned data in PostgreSQL with a clear schema
- Task 4: Insights and visualizations (sentiment, rating distributions, keyword clouds)

## Project Structure

- `data/`
  - `raw/` – raw scraped reviews (CSV)
  - `processed/` – cleaned dataset (CSV)
- `scripts/`
  - `scrape_reviews.py` – scrape Google Play reviews for the three banks
  - `preprocess_reviews.py` – clean and normalize review data
  - `load_to_postgres.py` – load cleaned data into PostgreSQL
- `sql/`
  - `schema.sql` – PostgreSQL schema (database + tables)
- `analysis/`
  - `insights_and_visuals.py` – sentiment, insights, and visualizations

## Google Play App IDs

- CBE: `com.combanketh.mobilebanking`
- BoA: `com.boa.boaMobileBanking`
- Dashen: `com.dashen.dashensuperapp`

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run Task 1 (data collection + preprocessing) on the `task-1` branch:

   ```bash
   python scripts/scrape_reviews.py
   python scripts/preprocess_reviews.py
   ```

4. Set up PostgreSQL (creates 3 databases: `cbe_reviews`, `boa_reviews`, `dashen_reviews`):

   ```bash
   # Create databases and apply schema
   python scripts/setup_databases.py
   
   # Copy .env.example to .env (already configured with password 123)
   copy .env.example .env
   
   # Load data into databases
   python scripts/load_to_postgres.py
   ```

5. Run Task 4 (insights & visuals) on the `task-4` branch:

   ```bash
   python analysis/insights_and_visuals.py
   ```

See in-code docstrings and comments for more details on parameters and behavior.
