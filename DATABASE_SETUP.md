# Database Setup Guide

## Overview

This project uses **3 separate PostgreSQL databases** (one per bank) with password `123`:

- `cbe_reviews` - Commercial Bank of Ethiopia
- `boa_reviews` - Bank of Abyssinia  
- `dashen_reviews` - Dashen Bank

## Connection Strings

### For Python Scripts (SQLAlchemy format)

In your `.env` file:

```env
DATABASE_URL_CBE=postgresql+psycopg2://postgres:123@localhost:5432/cbe_reviews
DATABASE_URL_BOA=postgresql+psycopg2://postgres:123@localhost:5432/boa_reviews
DATABASE_URL_DASHEN=postgresql+psycopg2://postgres:123@localhost:5432/dashen_reviews
```

### For Flask (if using Flask-SQLAlchemy)

```python
# For CBE
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:123@localhost:5432/cbe_reviews'

# For BOA
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:123@localhost:5432/boa_reviews'

# For Dashen
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:123@localhost:5432/dashen_reviews'
```

## Database Schema

Each database has the same structure:

### `banks` table
- `bank_id` (SERIAL PRIMARY KEY)
- `bank_name` (TEXT)
- `app_name` (TEXT)

### `reviews` table
- `review_id` (BIGSERIAL PRIMARY KEY)
- `bank_id` (INTEGER, FOREIGN KEY → banks.bank_id)
- `review_text` (TEXT)
- `rating` (INTEGER, 1-5)
- `review_date` (DATE)
- `sentiment_label` (TEXT: positive/negative/neutral)
- `sentiment_score` (DOUBLE PRECISION)
- `source` (TEXT: 'google_play')

## Setup Steps

### 1. Create Databases (Automated)

```bash
python scripts/setup_databases.py
```

This will:
- Create all 3 databases
- Apply the schema from `sql/schema.sql` to each

### 2. Manual Setup (Alternative)

If the automated script doesn't work:

```bash
# Create databases
createdb -U postgres cbe_reviews
createdb -U postgres boa_reviews
createdb -U postgres dashen_reviews

# Apply schema to each
psql -U postgres -d cbe_reviews -f sql/schema.sql
psql -U postgres -d boa_reviews -f sql/schema.sql
psql -U postgres -d dashen_reviews -f sql/schema.sql
```

### 3. Load Data

```bash
# First scrape and preprocess
python scripts/scrape_reviews.py
python scripts/preprocess_reviews.py

# Then load into databases
python scripts/load_to_postgres.py
```

The loader script will:
- Read `data/processed/bank_reviews_clean.csv`
- Split by bank code (CBE, BOA, DASHEN)
- Load each bank's reviews into its dedicated database
- Add sentiment analysis (VADER)
- Print verification stats

## Verification Queries

Connect to any database and run:

```sql
-- Count reviews
SELECT COUNT(*) FROM reviews;

-- Average rating
SELECT AVG(rating) FROM reviews;

-- Sentiment distribution
SELECT sentiment_label, COUNT(*) 
FROM reviews 
GROUP BY sentiment_label;

-- Reviews by date
SELECT review_date, COUNT(*) 
FROM reviews 
GROUP BY review_date 
ORDER BY review_date DESC;
```

## Flask Integration Example

If you want to query all 3 databases from Flask:

```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Define multiple database binds
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:123@localhost:5432/cbe_reviews'
app.config['SQLALCHEMY_BINDS'] = {
    'cbe': 'postgresql://postgres:123@localhost:5432/cbe_reviews',
    'boa': 'postgresql://postgres:123@localhost:5432/boa_reviews',
    'dashen': 'postgresql://postgres:123@localhost:5432/dashen_reviews',
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Models with bind_key
class CBEReview(db.Model):
    __bind_key__ = 'cbe'
    __tablename__ = 'reviews'
    review_id = db.Column(db.BigInteger, primary_key=True)
    # ... other columns

class BOAReview(db.Model):
    __bind_key__ = 'boa'
    __tablename__ = 'reviews'
    review_id = db.Column(db.BigInteger, primary_key=True)
    # ... other columns

class DashenReview(db.Model):
    __bind_key__ = 'dashen'
    __tablename__ = 'reviews'
    review_id = db.Column(db.BigInteger, primary_key=True)
    # ... other columns
```

## Troubleshooting

### Connection refused
- Ensure PostgreSQL is running: `pg_ctl status`
- Check port 5432 is not blocked

### Authentication failed
- Verify password is `123` for user `postgres`
- Update password: `ALTER USER postgres PASSWORD '123';`

### Database does not exist
- Run `python scripts/setup_databases.py`
- Or manually create with `createdb`

### Permission denied
- Ensure postgres user has CREATE DATABASE privilege
- Grant if needed: `ALTER USER postgres CREATEDB;`
