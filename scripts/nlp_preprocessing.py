"""
NLP Preprocessing Module
Provides tokenization, stop-word removal, lemmatization, and text cleaning
for customer review analysis.
"""

import re
import string
from typing import List, Optional

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Download required NLTK data
def download_nltk_resources():
    """Download required NLTK resources if not already present."""
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)


# Initialize resources
download_nltk_resources()

# Load spaCy model (use small English model)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy English model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


class NLPPreprocessor:
    """
    NLP preprocessing pipeline for customer reviews.
    Handles tokenization, stop-word removal, lemmatization, and cleaning.
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        min_token_length: int = 2,
        custom_stopwords: Optional[List[str]] = None,
    ):
        """
        Initialize the NLP preprocessor.

        Args:
            remove_stopwords: Remove common English stop words
            lemmatize: Apply lemmatization to tokens
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_numbers: Remove numeric tokens
            min_token_length: Minimum length for tokens to keep
            custom_stopwords: Additional stop words to remove
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.min_token_length = min_token_length

        # Initialize stop words
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)

        # Add banking-specific stop words that don't add value
        banking_stopwords = {
            'app', 'bank', 'banking', 'mobile', 'application',
            'please', 'thank', 'thanks', 'would', 'could', 'also'
        }
        self.stop_words.update(banking_stopwords)

        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None

    def clean_text(self, text: str) -> str:
        """
        Clean raw text by removing URLs, emails, and extra whitespace.

        Args:
            text: Raw text string

        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text string

        Returns:
            List of tokens
        """
        return word_tokenize(text)

    def preprocess(self, text: str) -> List[str]:
        """
        Apply full preprocessing pipeline to text.

        Args:
            text: Raw text string

        Returns:
            List of preprocessed tokens
        """
        # Clean text
        text = self.clean_text(text)

        if not text:
            return []

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Tokenize
        tokens = self.tokenize(text)

        # Process tokens
        processed_tokens = []
        for token in tokens:
            # Remove punctuation
            if self.remove_punctuation:
                token = token.translate(str.maketrans('', '', string.punctuation))

            # Skip empty tokens
            if not token:
                continue

            # Remove numbers
            if self.remove_numbers and token.isdigit():
                continue

            # Check minimum length
            if len(token) < self.min_token_length:
                continue

            # Remove stop words
            if self.remove_stopwords and token in self.stop_words:
                continue

            # Lemmatize
            if self.lemmatize and self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)

            processed_tokens.append(token)

        return processed_tokens

    def preprocess_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            List of preprocessed token lists
        """
        return [self.preprocess(text) for text in texts]

    def extract_noun_phrases_spacy(self, text: str) -> List[str]:
        """
        Extract noun phrases using spaCy.

        Args:
            text: Input text string

        Returns:
            List of noun phrases
        """
        text = self.clean_text(text)
        if not text:
            return []

        doc = nlp(text)
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]

        # Filter out very short phrases and those with only stop words
        filtered_phrases = []
        for phrase in noun_phrases:
            words = phrase.split()
            if len(words) >= 2 or (len(words) == 1 and len(words[0]) > 3):
                if not all(word in self.stop_words for word in words):
                    filtered_phrases.append(phrase)

        return filtered_phrases

    def extract_entities_spacy(self, text: str) -> List[tuple]:
        """
        Extract named entities using spaCy.

        Args:
            text: Input text string

        Returns:
            List of (entity_text, entity_label) tuples
        """
        text = self.clean_text(text)
        if not text:
            return []

        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        return entities


def preprocess_reviews_dataframe(df, text_column='review', output_column='tokens'):
    """
    Preprocess reviews in a pandas DataFrame.

    Args:
        df: pandas DataFrame containing reviews
        text_column: Name of column containing text to preprocess
        output_column: Name of column to store preprocessed tokens

    Returns:
        DataFrame with added preprocessed tokens column
    """
    preprocessor = NLPPreprocessor()

    df[output_column] = df[text_column].fillna('').apply(preprocessor.preprocess)
    df['tokens_text'] = df[output_column].apply(lambda x: ' '.join(x))

    return df


if __name__ == "__main__":
    # Example usage
    sample_text = """
    This banking app is terrible! The login process crashes frequently and 
    customer service is very slow to respond. I can't access my account balance.
    Please fix these bugs ASAP!
    """

    preprocessor = NLPPreprocessor()

    print("Original text:")
    print(sample_text)

    print("\nCleaned text:")
    print(preprocessor.clean_text(sample_text))

    print("\nTokens:")
    tokens = preprocessor.preprocess(sample_text)
    print(tokens)

    print("\nNoun phrases:")
    phrases = preprocessor.extract_noun_phrases_spacy(sample_text)
    print(phrases)

    print("\nNamed entities:")
    entities = preprocessor.extract_entities_spacy(sample_text)
    print(entities)
