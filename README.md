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
  - `load_to_postgres.py` – load cleaned data into PostgreSQL (3 separate databases)
  - `setup_databases.py` – automated database creation and schema setup
  - `nlp_preprocessing.py` – NLP preprocessing (tokenization, lemmatization, stop-word removal)
  - `keyword_extraction.py` – keyword/n-gram extraction using TF-IDF and spaCy
- `sql/`
  - `schema.sql` – PostgreSQL schema (database + tables)
- `analysis/`
  - `theme_analysis.py` – comprehensive theme, pain point, and driver analysis
  - `insights_and_visuals.py` – sentiment, insights, and visualizations
  - `NLP_MODULE_README.md` – detailed NLP module documentation

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

5. Download spaCy language model for NLP analysis:

   ```bash
   python -m spacy download en_core_web_sm
   ```

6. Run NLP theme analysis to identify pain points and positive drivers:

   ```bash
   python analysis/theme_analysis.py
   ```

   This generates detailed reports in `analysis/reports/` including:
   - Individual bank theme analysis reports
   - Cross-bank comparison report
   - CSV exports of keywords, pain points, and positive drivers

7. Run Task 4 (insights & visuals):

   ```bash
   python analysis/insights_and_visuals.py
   ```

## NLP Analysis Features

The project includes advanced NLP capabilities:

- **Tokenization & Preprocessing**: Clean text, remove stop-words, lemmatize tokens
- **TF-IDF Keyword Extraction**: Identify statistically significant terms (unigrams, bigrams, trigrams)
- **N-gram Analysis**: Extract common 2-word and 3-word phrases
- **spaCy Noun Phrases**: Linguistic pattern-based phrase extraction
- **Pain Point Detection**: Analyze negative reviews (rating ≤ 3) to identify problems
- **Positive Driver Detection**: Analyze positive reviews (rating ≥ 4) to identify strengths

See `analysis/NLP_MODULE_README.md` for detailed documentation and usage examples.

See in-code docstrings and comments for more details on parameters and behavior.
