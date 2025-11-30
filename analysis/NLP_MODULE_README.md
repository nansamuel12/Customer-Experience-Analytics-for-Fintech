# NLP Analysis Module

This module provides comprehensive NLP-based analysis of customer reviews to identify themes, pain points, and positive drivers for each bank.

## Features

### 1. **NLP Preprocessing** (`scripts/nlp_preprocessing.py`)

Advanced text preprocessing pipeline including:

- **Tokenization**: Breaks text into individual words/tokens
- **Stop-word Removal**: Removes common words that don't add meaning (e.g., "the", "is", "and")
- **Lemmatization**: Reduces words to their base form (e.g., "running" → "run")
- **Text Cleaning**: Removes URLs, emails, extra whitespace
- **Custom Banking Stop-words**: Filters out generic banking terms
- **spaCy Integration**: Extracts noun phrases and named entities

**Example Usage:**

```python
from nlp_preprocessing import NLPPreprocessor

preprocessor = NLPPreprocessor()

text = "The mobile banking app crashes frequently!"
tokens = preprocessor.preprocess(text)
# Output: ['mobile', 'crash', 'frequently']

noun_phrases = preprocessor.extract_noun_phrases_spacy(text)
# Output: ['mobile banking app']
```

### 2. **Keyword Extraction** (`scripts/keyword_extraction.py`)

Multiple methods for extracting key themes:

- **TF-IDF (Term Frequency-Inverse Document Frequency)**
  - Identifies important words/phrases based on statistical significance
  - Supports unigrams, bigrams, and trigrams
  
- **N-gram Analysis**
  - Extracts common 2-word and 3-word phrases
  - Helps identify multi-word concepts (e.g., "customer service", "login problem")

- **spaCy Noun Phrase Extraction**
  - Identifies meaningful noun phrases using linguistic patterns
  - Better at capturing domain-specific terms

- **Pain Point Detection**
  - Analyzes negative reviews (rating ≤ 3) to identify problems
  - Combines TF-IDF and noun phrases for comprehensive coverage

- **Positive Driver Detection**
  - Analyzes positive reviews (rating ≥ 4) to identify strengths
  - Helps understand what customers appreciate

**Example Usage:**

```python
from keyword_extraction import KeywordExtractor

extractor = KeywordExtractor()

# Extract top keywords using TF-IDF
keywords = extractor.extract_tfidf_keywords(
    documents=reviews,
    top_n=20,
    ngram_range=(1, 2)  # unigrams and bigrams
)

# Extract pain points from negative reviews
pain_points = extractor.extract_pain_points(
    documents=reviews,
    ratings=ratings,
    negative_threshold=3,
    top_n=15
)
```

### 3. **Theme Analysis** (`analysis/theme_analysis.py`)

Comprehensive analysis script that:

- Analyzes each bank separately
- Generates detailed text reports
- Creates CSV exports for further analysis
- Produces cross-bank comparison reports

**Outputs:**

For each bank:
- Top 20 keywords (TF-IDF)
- Top 15 bigrams (2-word phrases)
- Top 10 trigrams (3-word phrases)
- Top 20 noun phrases
- Top 15 pain points (from negative reviews)
- Top 15 positive drivers (from positive reviews)

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

## Usage

### Quick Start

```bash
# Run comprehensive theme analysis
python analysis/theme_analysis.py
```

This will:
1. Load cleaned review data
2. Analyze each bank (CBE, BOA, Dashen)
3. Generate individual reports in `analysis/reports/`
4. Create a cross-bank comparison report
5. Export detailed CSV files

### Output Files

After running the analysis, you'll find:

```
analysis/reports/
├── cbe_theme_analysis.txt          # CBE detailed report
├── boa_theme_analysis.txt          # BOA detailed report
├── dashen_theme_analysis.txt       # Dashen detailed report
├── bank_comparison_report.txt      # Cross-bank comparison
├── cbe_keywords.csv                # CBE keywords (CSV)
├── cbe_pain_points.csv             # CBE pain points (CSV)
├── cbe_positive_drivers.csv        # CBE positive drivers (CSV)
├── boa_keywords.csv                # BOA keywords (CSV)
├── boa_pain_points.csv             # BOA pain points (CSV)
├── boa_positive_drivers.csv        # BOA positive drivers (CSV)
├── dashen_keywords.csv             # Dashen keywords (CSV)
├── dashen_pain_points.csv          # Dashen pain points (CSV)
└── dashen_positive_drivers.csv     # Dashen positive drivers (CSV)
```

### Programmatic Usage

```python
import pandas as pd
from keyword_extraction import analyze_bank_themes

# Load your data
df = pd.read_csv('data/processed/bank_reviews_clean.csv')

# Analyze a specific bank
df_cbe = df[df['bank'] == 'CBE']
results = analyze_bank_themes(
    df_cbe,
    bank_code='CBE',
    text_column='review',
    rating_column='rating'
)

# Access results
print(results['top_keywords'])
print(results['pain_points'])
print(results['positive_drivers'])
```

## Methodology

### TF-IDF Scoring

TF-IDF measures how important a word is to a document in a collection:

- **TF (Term Frequency)**: How often a word appears in a document
- **IDF (Inverse Document Frequency)**: How rare/common a word is across all documents
- **TF-IDF = TF × IDF**: High score = important and distinctive

### Pain Point Detection

1. Filter reviews with rating ≤ 3 (negative reviews)
2. Apply NLP preprocessing (tokenization, lemmatization, stop-word removal)
3. Extract keywords using TF-IDF (unigrams, bigrams, trigrams)
4. Extract noun phrases using spaCy
5. Combine and rank by score
6. Return top N pain points

### Positive Driver Detection

Same process as pain points, but:
1. Filter reviews with rating ≥ 4 (positive reviews)
2. Extract what customers appreciate

## Customization

### Adjust Stop Words

```python
from nlp_preprocessing import NLPPreprocessor

custom_stops = ['specific', 'words', 'to', 'ignore']
preprocessor = NLPPreprocessor(custom_stopwords=custom_stops)
```

### Change N-gram Range

```python
from keyword_extraction import KeywordExtractor

extractor = KeywordExtractor()

# Extract only trigrams (3-word phrases)
trigrams = extractor.extract_tfidf_keywords(
    documents=reviews,
    ngram_range=(3, 3)
)
```

### Adjust Rating Thresholds

```python
# Consider rating ≤ 2 as negative (more strict)
pain_points = extractor.extract_pain_points(
    documents=reviews,
    ratings=ratings,
    negative_threshold=2
)

# Consider rating ≥ 5 as positive (only excellent reviews)
drivers = extractor.extract_positive_drivers(
    documents=reviews,
    ratings=ratings,
    positive_threshold=5
)
```

## Technical Details

### Dependencies

- **spaCy**: Industrial-strength NLP library for noun phrase extraction
- **NLTK**: Tokenization, stop-words, lemmatization
- **scikit-learn**: TF-IDF vectorization and n-gram extraction
- **pandas**: Data manipulation and analysis

### Performance

- Preprocessing: ~100-500 reviews/second (depends on text length)
- TF-IDF extraction: ~1000 reviews/second
- spaCy noun phrases: ~50-200 reviews/second (slower but more accurate)

### Memory Usage

- Small datasets (<10K reviews): <500MB RAM
- Medium datasets (10K-100K reviews): 500MB-2GB RAM
- Large datasets (>100K reviews): 2GB+ RAM

## Troubleshooting

### spaCy model not found

```bash
python -m spacy download en_core_web_sm
```

### NLTK data not found

The script automatically downloads required NLTK data, but you can manually download:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### Out of memory errors

For large datasets, process banks separately:

```python
for bank in ['CBE', 'BOA', 'DASHEN']:
    df_bank = df[df['bank'] == bank]
    results = analyze_bank_themes(df_bank, bank)
    # Process results...
```

## Future Enhancements

Potential improvements:

- Topic modeling (LDA, NMF) for automatic theme discovery
- Sentiment-specific keyword extraction
- Temporal analysis (how themes change over time)
- Aspect-based sentiment analysis
- Multi-language support for Amharic reviews
