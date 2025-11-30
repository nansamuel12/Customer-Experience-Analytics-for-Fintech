# Requirements Checklist - NLP Module Implementation

## ✅ All Requirements Met

### 1. ✅ VADER-based Sentiment Analysis
**Location:** `scripts/load_to_postgres.py`

```python
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
```

**Status:** ✅ Integrated into PostgreSQL loading pipeline
- Sentiment scores calculated for all reviews
- Labels: positive, negative, neutral
- Stored in database `sentiment_score` and `sentiment_label` columns

---

### 2. ✅ NLP Preprocessing Module
**Location:** `scripts/nlp_preprocessing.py`

#### Features Implemented:

**a) Tokenization** ✅
```python
def tokenize(self, text: str) -> List[str]:
    return word_tokenize(text)
```
- Uses NLTK's `word_tokenize`
- Breaks text into individual words/tokens

**b) Stop-word Removal** ✅
```python
self.stop_words = set(stopwords.words('english'))
# Add banking-specific stop words
banking_stopwords = {
    'app', 'bank', 'banking', 'mobile', 'application',
    'please', 'thank', 'thanks', 'would', 'could', 'also'
}
self.stop_words.update(banking_stopwords)
```
- Removes common English stop words
- Custom banking-specific stop words
- Configurable with custom stop words

**c) Lemmatization** ✅
```python
self.lemmatizer = WordNetLemmatizer()
token = self.lemmatizer.lemmatize(token)
```
- Uses NLTK's WordNetLemmatizer
- Reduces words to base forms (e.g., "crashes" → "crash")

**d) Additional Preprocessing** ✅
- URL removal
- Email removal
- Punctuation removal
- Number filtering
- Minimum token length filtering
- Text cleaning and normalization

**Full Pipeline:**
```python
preprocessor = NLPPreprocessor()
tokens = preprocessor.preprocess(text)
# Returns: cleaned, tokenized, lemmatized, stop-word-free tokens
```

---

### 3. ✅ Keyword/N-gram Extraction
**Location:** `scripts/keyword_extraction.py`

#### a) TF-IDF Keyword Extraction ✅
```python
def extract_tfidf_keywords(
    self,
    documents: List[str],
    top_n: int = 20,
    max_features: int = 1000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.8,
    )
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    # Returns top keywords with TF-IDF scores
```

**Features:**
- Supports unigrams, bigrams, and trigrams
- Configurable n-gram range
- Statistical significance scoring
- Filters rare and overly common terms

#### b) N-gram Analysis ✅
```python
def extract_ngrams(
    self,
    documents: List[str],
    n: int = 2,
    top_n: int = 20,
) -> pd.DataFrame:
    vectorizer = CountVectorizer(
        ngram_range=(n, n),
        min_df=2,
        max_df=0.8,
    )
    # Returns most common n-grams with frequencies
```

**Supports:**
- Bigrams (2-word phrases)
- Trigrams (3-word phrases)
- Frequency-based ranking

#### c) spaCy Noun Phrase Extraction ✅
```python
def extract_noun_phrases_spacy(self, text: str) -> List[str]:
    doc = nlp(text)
    noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
    # Filters short phrases and stop-word-only phrases
    return filtered_phrases
```

**Features:**
- Linguistic pattern-based extraction
- Identifies meaningful multi-word concepts
- Automatic filtering of low-quality phrases

---

### 4. ✅ Theme and Pain Point Analysis Per Bank
**Location:** `analysis/theme_analysis.py`

#### Comprehensive Analysis Function:
```python
def analyze_bank_themes(
    df: pd.DataFrame,
    bank_code: str,
    text_column: str = 'review',
    rating_column: str = 'rating',
) -> Dict:
```

**For Each Bank, Extracts:**

1. **Top Keywords (TF-IDF)** - Top 20 statistically significant terms
2. **Top Bigrams** - Top 15 two-word phrases
3. **Top Trigrams** - Top 10 three-word phrases
4. **Noun Phrases** - Top 20 spaCy-extracted phrases
5. **Pain Points** ✅ - Top 15 issues from negative reviews (rating ≤ 3)
6. **Positive Drivers** ✅ - Top 15 strengths from positive reviews (rating ≥ 4)

#### Pain Point Detection:
```python
def extract_pain_points(
    self,
    documents: List[str],
    ratings: List[int] = None,
    negative_threshold: int = 3,
    top_n: int = 15,
) -> pd.DataFrame:
    # Filter negative reviews
    negative_docs = [
        doc for doc, rating in zip(documents, ratings)
        if rating <= negative_threshold
    ]
    # Extract keywords from negative reviews using TF-IDF + noun phrases
```

#### Positive Driver Detection:
```python
def extract_positive_drivers(
    self,
    documents: List[str],
    ratings: List[int] = None,
    positive_threshold: int = 4,
    top_n: int = 15,
) -> pd.DataFrame:
    # Filter positive reviews
    positive_docs = [
        doc for doc, rating in zip(documents, ratings)
        if rating >= positive_threshold
    ]
    # Extract what customers appreciate
```

---

## 📊 Output Reports Generated

### Individual Bank Reports:
- `analysis/reports/cbe_theme_analysis.txt`
- `analysis/reports/boa_theme_analysis.txt`
- `analysis/reports/dashen_theme_analysis.txt`

**Each report includes:**
- Total reviews analyzed
- Average rating
- Top 15 keywords (TF-IDF)
- Top 15 bigrams
- Top 10 trigrams
- Top 15 noun phrases
- **Top 15 pain points** (from negative reviews)
- **Top 15 positive drivers** (from positive reviews)
- Key insights and recommendations

### Cross-Bank Comparison:
- `analysis/reports/bank_comparison_report.txt`

**Includes:**
- Overview table (reviews, ratings per bank)
- Pain points comparison across all banks
- Positive drivers comparison across all banks

### CSV Exports:
For each bank:
- `{bank}_keywords.csv`
- `{bank}_pain_points.csv`
- `{bank}_positive_drivers.csv`

---

## 🚀 Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run comprehensive analysis
python analysis/theme_analysis.py
```

---

## 📦 Dependencies Added

Updated `requirements.txt`:
```
spacy              # Noun phrase extraction, NER
scikit-learn       # TF-IDF vectorization
textblob           # Additional NLP support
nltk               # Tokenization, lemmatization, stop-words
```

---

## ✅ Summary

All requirements have been fully implemented:

1. ✅ **VADER sentiment analysis** - Integrated in data loading pipeline
2. ✅ **Tokenization** - NLTK word_tokenize
3. ✅ **Stop-word removal** - English + custom banking stop-words
4. ✅ **Lemmatization** - NLTK WordNetLemmatizer
5. ✅ **TF-IDF keyword extraction** - Unigrams, bigrams, trigrams
6. ✅ **N-gram extraction** - Frequency-based bigrams and trigrams
7. ✅ **spaCy noun phrases** - Linguistic pattern extraction
8. ✅ **Pain point detection** - Per bank, from negative reviews
9. ✅ **Positive driver detection** - Per bank, from positive reviews
10. ✅ **Theme analysis** - Comprehensive reports for each bank

**All code committed and pushed to main branch.**

Git commit: `78b088e - Add comprehensive NLP module: tokenization, lemmatization, TF-IDF, n-gram extraction, and theme analysis for pain points and positive drivers`
