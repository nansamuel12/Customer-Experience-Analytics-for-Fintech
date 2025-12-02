"""
Insights and Visualization Script
Generates concrete plots for customer experience analytics:
1. Rating distributions per bank
2. Sentiment over time per bank
3. Top pain-point themes
4. Word clouds for each bank

Saves all plots to the 'analysis/visuals' directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from wordcloud import WordCloud

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Define paths
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "bank_reviews_clean.csv"
VISUALS_DIR = BASE_DIR / "analysis" / "visuals"

# Ensure visuals directory exists
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load processed review data."""
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {PROCESSED_DATA_PATH}")
    
    df = pd.read_csv(PROCESSED_DATA_PATH)
    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])
    return df

def plot_rating_distribution(df):
    """Plot rating distribution per bank."""
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x="rating", hue="bank", palette="viridis")
    plt.title("Rating Distribution per Bank", fontsize=16)
    plt.xlabel("Rating (1-5)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(title="Bank")
    
    output_path = VISUALS_DIR / "rating_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved rating distribution plot to {output_path}")

def plot_sentiment_over_time(df):
    """Plot average rating over time (monthly) per bank."""
    # Resample to monthly average
    df_monthly = df.groupby(["bank", pd.Grouper(key="date", freq="M")])["rating"].mean().reset_index()
    
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_monthly, x="date", y="rating", hue="bank", marker="o", palette="viridis")
    
    plt.title("Average Sentiment (Rating) Over Time", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Average Rating", fontsize=12)
    plt.legend(title="Bank")
    plt.ylim(1, 5)  # Ratings are 1-5
    
    output_path = VISUALS_DIR / "sentiment_over_time.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved sentiment over time plot to {output_path}")

def plot_pain_points(df):
    """
    Plot top pain points based on negative reviews (rating <= 2).
    This is a simplified keyword frequency analysis for demonstration.
    For more advanced NLP, use the keyword_extraction module.
    """
    from collections import Counter
    import re

    # Filter negative reviews
    neg_df = df[df["rating"] <= 2].copy()
    
    # Simple stop words
    stop_words = set(['the', 'and', 'to', 'is', 'a', 'it', 'of', 'in', 'for', 'i', 'this', 'app', 'bank', 'my', 'not', 'on', 'with', 'are', 'be', 'have', 'but', 'that', 'so', 'me', 'you', 'can', 'was', 'as', 'if', 'at', 'just', 'or', 'an', 'very', 'no', 'up', 'do', 'from', 'what', 'out', 'when', 'all', 'get', 'we', 'has', 'time', 'will', 'by', 'about', 'would', 'please', 'fix', 'update', 'even', 'now', 'there', 'am', 'they', 'one', 'use', 'back', 'make', 'service', 'customer', 'account', 'money', 'transfer', 'login', 'cant', 'cannot', 'working', 'open', 'transaction', 'mobile', 'banking'])

    all_pain_points = []
    
    for review in neg_df["review"].dropna():
        # Simple cleaning
        words = re.findall(r'\b\w+\b', review.lower())
        # Filter stop words and short words
        words = [w for w in words if w not in stop_words and len(w) > 3]
        all_pain_points.extend(words)
        
    # Count frequencies
    common_words = Counter(all_pain_points).most_common(15)
    
    if not common_words:
        print("No pain points found (not enough negative reviews?)")
        return

    pain_df = pd.DataFrame(common_words, columns=["Word", "Count"])
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=pain_df, x="Count", y="Word", palette="Reds_r")
    plt.title("Top Pain Point Themes (Negative Reviews)", fontsize=16)
    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("Theme / Keyword", fontsize=12)
    
    output_path = VISUALS_DIR / "top_pain_points.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved pain points plot to {output_path}")

def generate_wordclouds(df):
    """Generate word clouds for each bank."""
    banks = df["bank"].unique()
    
    for bank in banks:
        bank_reviews = " ".join(df[df["bank"] == bank]["review"].dropna().astype(str))
        
        wordcloud = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(bank_reviews)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud - {bank}", fontsize=16)
        
        output_path = VISUALS_DIR / f"wordcloud_{bank.lower()}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved word cloud for {bank} to {output_path}")

def main():
    print("Starting visualization generation...")
    try:
        df = load_data()
        print(f"Loaded {len(df)} reviews.")
        
        plot_rating_distribution(df)
        plot_sentiment_over_time(df)
        plot_pain_points(df)
        generate_wordclouds(df)
        
        print("\nAll visualizations generated successfully in 'analysis/visuals'.")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")

if __name__ == "__main__":
    main()
