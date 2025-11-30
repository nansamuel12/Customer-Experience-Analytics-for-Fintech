-- PostgreSQL schema for separate bank review databases
-- Create three databases manually (one per bank):
--   createdb cbe_reviews
--   createdb boa_reviews
--   createdb dashen_reviews
-- Then run this schema on each database:
--   psql -d cbe_reviews -f schema.sql
--   psql -d boa_reviews -f schema.sql
--   psql -d dashen_reviews -f schema.sql

CREATE TABLE IF NOT EXISTS banks (
    bank_id   SERIAL PRIMARY KEY,
    bank_name TEXT NOT NULL,
    app_name  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id        BIGSERIAL PRIMARY KEY,
    bank_id          INTEGER NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    review_text      TEXT NOT NULL,
    rating           INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    review_date      DATE NOT NULL,
    sentiment_label  TEXT,
    sentiment_score  DOUBLE PRECISION,
    source           TEXT NOT NULL
);

-- Example integrity checks:
-- Count reviews per bank
-- SELECT b.bank_name, COUNT(r.review_id) AS review_count
-- FROM reviews r
-- JOIN banks b ON r.bank_id = b.bank_id
-- GROUP BY b.bank_name
-- ORDER BY b.bank_name;

-- Average rating per bank
-- SELECT b.bank_name, AVG(r.rating) AS avg_rating
-- FROM reviews r
-- JOIN banks b ON r.bank_id = b.bank_id
-- GROUP BY b.bank_name
-- ORDER BY b.bank_name;
