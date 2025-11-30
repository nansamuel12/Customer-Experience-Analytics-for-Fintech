"""
Keyword and N-gram Extraction Module
Extracts key themes, topics, and pain points from customer reviews using:
- TF-IDF (Term Frequency-Inverse Document Frequency)
- N-gram analysis (bigrams, trigrams)
- spaCy noun phrase extraction
"""

from collections import Counter
from typing import List, Tuple, Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nlp_preprocessing import NLPPreprocessor


class KeywordExtractor:
    """Extract keywords and themes from text using various methods."""

    def __init__(self, preprocessor: NLPPreprocessor = None):
        """
        Initialize keyword extractor.

        Args:
            preprocessor: NLPPreprocessor instance (creates default if None)
        """
        self.preprocessor = preprocessor or NLPPreprocessor()

    def extract_tfidf_keywords(
        self,
        documents: List[str],
        top_n: int = 20,
        max_features: int = 1000,
        ngram_range: Tuple[int, int] = (1, 2),
    ) -> pd.DataFrame:
        """
        Extract top keywords using TF-IDF.

        Args:
            documents: List of text documents
            top_n: Number of top keywords to return
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams (min, max)

        Returns:
            DataFrame with keywords and their TF-IDF scores
        """
        # Preprocess documents
        processed_docs = [
            ' '.join(self.preprocessor.preprocess(doc)) for doc in documents
        ]

        # Remove empty documents
        processed_docs = [doc for doc in processed_docs if doc.strip()]

        if not processed_docs:
            return pd.DataFrame(columns=['keyword', 'tfidf_score'])

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
        )

        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(processed_docs)

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Calculate average TF-IDF score for each term across all documents
        avg_tfidf = tfidf_matrix.mean(axis=0).A1

        # Create DataFrame
        df_tfidf = pd.DataFrame({
            'keyword': feature_names,
            'tfidf_score': avg_tfidf
        })

        # Sort by TF-IDF score and get top N
        df_tfidf = df_tfidf.sort_values('tfidf_score', ascending=False).head(top_n)

        return df_tfidf.reset_index(drop=True)

    def extract_ngrams(
        self,
        documents: List[str],
        n: int = 2,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Extract most common n-grams.

        Args:
            documents: List of text documents
            n: N-gram size (2 for bigrams, 3 for trigrams)
            top_n: Number of top n-grams to return

        Returns:
            DataFrame with n-grams and their frequencies
        """
        # Preprocess documents
        processed_docs = [
            ' '.join(self.preprocessor.preprocess(doc)) for doc in documents
        ]

        # Remove empty documents
        processed_docs = [doc for doc in processed_docs if doc.strip()]

        if not processed_docs:
            return pd.DataFrame(columns=['ngram', 'frequency'])

        # Create CountVectorizer for n-grams
        vectorizer = CountVectorizer(
            ngram_range=(n, n),
            min_df=2,
            max_df=0.8,
        )

        # Fit and transform
        ngram_matrix = vectorizer.fit_transform(processed_docs)

        # Get feature names and counts
        feature_names = vectorizer.get_feature_names_out()
        ngram_counts = ngram_matrix.sum(axis=0).A1

        # Create DataFrame
        df_ngrams = pd.DataFrame({
            'ngram': feature_names,
            'frequency': ngram_counts
        })

        # Sort by frequency and get top N
        df_ngrams = df_ngrams.sort_values('frequency', ascending=False).head(top_n)

        return df_ngrams.reset_index(drop=True)

    def extract_noun_phrases(
        self,
        documents: List[str],
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Extract most common noun phrases using spaCy.

        Args:
            documents: List of text documents
            top_n: Number of top noun phrases to return

        Returns:
            DataFrame with noun phrases and their frequencies
        """
        all_phrases = []

        for doc in documents:
            phrases = self.preprocessor.extract_noun_phrases_spacy(doc)
            all_phrases.extend(phrases)

        if not all_phrases:
            return pd.DataFrame(columns=['noun_phrase', 'frequency'])

        # Count phrase frequencies
        phrase_counts = Counter(all_phrases)

        # Create DataFrame
        df_phrases = pd.DataFrame(
            phrase_counts.most_common(top_n),
            columns=['noun_phrase', 'frequency']
        )

        return df_phrases

    def extract_pain_points(
        self,
        documents: List[str],
        ratings: List[int] = None,
        negative_threshold: int = 3,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """
        Extract pain points from negative reviews.

        Args:
            documents: List of text documents
            ratings: List of ratings (1-5) corresponding to documents
            negative_threshold: Reviews with rating <= this are considered negative
            top_n: Number of top pain points to return

        Returns:
            DataFrame with pain points and their frequencies
        """
        # Filter negative reviews
        if ratings:
            negative_docs = [
                doc for doc, rating in zip(documents, ratings)
                if rating <= negative_threshold
            ]
        else:
            negative_docs = documents

        if not negative_docs:
            return pd.DataFrame(columns=['pain_point', 'frequency', 'type'])

        # Extract keywords from negative reviews
        tfidf_keywords = self.extract_tfidf_keywords(
            negative_docs,
            top_n=top_n,
            ngram_range=(1, 3)
        )

        # Extract noun phrases
        noun_phrases = self.extract_noun_phrases(negative_docs, top_n=top_n)

        # Combine and deduplicate
        pain_points = []

        for _, row in tfidf_keywords.iterrows():
            pain_points.append({
                'pain_point': row['keyword'],
                'score': row['tfidf_score'],
                'type': 'tfidf'
            })

        for _, row in noun_phrases.iterrows():
            pain_points.append({
                'pain_point': row['noun_phrase'],
                'score': row['frequency'],
                'type': 'noun_phrase'
            })

        df_pain_points = pd.DataFrame(pain_points)

        # Remove duplicates and sort
        df_pain_points = df_pain_points.drop_duplicates(subset=['pain_point'])
        df_pain_points = df_pain_points.sort_values('score', ascending=False).head(top_n)

        return df_pain_points.reset_index(drop=True)

    def extract_positive_drivers(
        self,
        documents: List[str],
        ratings: List[int] = None,
        positive_threshold: int = 4,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """
        Extract positive drivers from positive reviews.

        Args:
            documents: List of text documents
            ratings: List of ratings (1-5) corresponding to documents
            positive_threshold: Reviews with rating >= this are considered positive
            top_n: Number of top drivers to return

        Returns:
            DataFrame with positive drivers and their frequencies
        """
        # Filter positive reviews
        if ratings:
            positive_docs = [
                doc for doc, rating in zip(documents, ratings)
                if rating >= positive_threshold
            ]
        else:
            positive_docs = documents

        if not positive_docs:
            return pd.DataFrame(columns=['driver', 'frequency', 'type'])

        # Extract keywords from positive reviews
        tfidf_keywords = self.extract_tfidf_keywords(
            positive_docs,
            top_n=top_n,
            ngram_range=(1, 3)
        )

        # Extract noun phrases
        noun_phrases = self.extract_noun_phrases(positive_docs, top_n=top_n)

        # Combine
        drivers = []

        for _, row in tfidf_keywords.iterrows():
            drivers.append({
                'driver': row['keyword'],
                'score': row['tfidf_score'],
                'type': 'tfidf'
            })

        for _, row in noun_phrases.iterrows():
            drivers.append({
                'driver': row['noun_phrase'],
                'score': row['frequency'],
                'type': 'noun_phrase'
            })

        df_drivers = pd.DataFrame(drivers)

        # Remove duplicates and sort
        df_drivers = df_drivers.drop_duplicates(subset=['driver'])
        df_drivers = df_drivers.sort_values('score', ascending=False).head(top_n)

        return df_drivers.reset_index(drop=True)


def analyze_bank_themes(
    df: pd.DataFrame,
    bank_code: str,
    text_column: str = 'review',
    rating_column: str = 'rating',
) -> Dict:
    """
    Analyze themes, pain points, and drivers for a specific bank.

    Args:
        df: DataFrame containing reviews for the bank
        bank_code: Bank identifier (e.g., 'CBE', 'BOA', 'DASHEN')
        text_column: Name of column containing review text
        rating_column: Name of column containing ratings

    Returns:
        Dictionary containing analysis results
    """
    extractor = KeywordExtractor()

    documents = df[text_column].fillna('').tolist()
    ratings = df[rating_column].tolist() if rating_column in df.columns else None

    results = {
        'bank': bank_code,
        'total_reviews': len(df),
        'avg_rating': df[rating_column].mean() if rating_column in df.columns else None,
    }

    # Extract overall keywords
    print(f"\nExtracting keywords for {bank_code}...")
    results['top_keywords'] = extractor.extract_tfidf_keywords(
        documents, top_n=20, ngram_range=(1, 2)
    )

    # Extract bigrams
    print(f"Extracting bigrams for {bank_code}...")
    results['top_bigrams'] = extractor.extract_ngrams(documents, n=2, top_n=15)

    # Extract trigrams
    print(f"Extracting trigrams for {bank_code}...")
    results['top_trigrams'] = extractor.extract_ngrams(documents, n=3, top_n=10)

    # Extract noun phrases
    print(f"Extracting noun phrases for {bank_code}...")
    results['noun_phrases'] = extractor.extract_noun_phrases(documents, top_n=20)

    # Extract pain points (from negative reviews)
    print(f"Extracting pain points for {bank_code}...")
    results['pain_points'] = extractor.extract_pain_points(
        documents, ratings, negative_threshold=3, top_n=15
    )

    # Extract positive drivers (from positive reviews)
    print(f"Extracting positive drivers for {bank_code}...")
    results['positive_drivers'] = extractor.extract_positive_drivers(
        documents, ratings, positive_threshold=4, top_n=15
    )

    return results


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    # Load sample data
    data_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "bank_reviews_clean.csv"

    if data_path.exists():
        df = pd.read_csv(data_path)

        # Analyze each bank
        for bank in df['bank'].unique():
            df_bank = df[df['bank'] == bank]
            results = analyze_bank_themes(df_bank, bank)

            print(f"\n{'='*60}")
            print(f"Analysis for {bank}")
            print(f"{'='*60}")
            print(f"Total reviews: {results['total_reviews']}")
            print(f"Average rating: {results['avg_rating']:.2f}")

            print("\nTop Keywords (TF-IDF):")
            print(results['top_keywords'].head(10))

            print("\nTop Bigrams:")
            print(results['top_bigrams'].head(10))

            print("\nPain Points:")
            print(results['pain_points'].head(10))

            print("\nPositive Drivers:")
            print(results['positive_drivers'].head(10))
    else:
        print(f"Data file not found at {data_path}")
        print("Please run scrape_reviews.py and preprocess_reviews.py first.")
