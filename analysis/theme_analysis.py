"""
Comprehensive Theme Analysis Script
Analyzes customer reviews to identify key themes, pain points, and positive drivers
for each bank using NLP techniques.
"""

import sys
from pathlib import Path

import pandas as pd

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from keyword_extraction import analyze_bank_themes, KeywordExtractor
from nlp_preprocessing import NLPPreprocessor


def generate_bank_report(results: dict, output_dir: Path) -> None:
    """
    Generate a detailed text report for a bank's analysis.

    Args:
        results: Analysis results dictionary from analyze_bank_themes
        output_dir: Directory to save the report
    """
    bank = results['bank']
    report_path = output_dir / f"{bank.lower()}_theme_analysis.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"THEME ANALYSIS REPORT: {bank}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Reviews Analyzed: {results['total_reviews']}\n")
        if results['avg_rating']:
            f.write(f"Average Rating: {results['avg_rating']:.2f}/5.0\n")
        f.write("\n")

        # Top Keywords
        f.write("-" * 80 + "\n")
        f.write("TOP KEYWORDS (TF-IDF)\n")
        f.write("-" * 80 + "\n")
        if not results['top_keywords'].empty:
            for idx, row in results['top_keywords'].head(15).iterrows():
                f.write(f"{idx+1:2d}. {row['keyword']:30s} (score: {row['tfidf_score']:.4f})\n")
        else:
            f.write("No keywords found.\n")
        f.write("\n")

        # Top Bigrams
        f.write("-" * 80 + "\n")
        f.write("TOP BIGRAMS (2-word phrases)\n")
        f.write("-" * 80 + "\n")
        if not results['top_bigrams'].empty:
            for idx, row in results['top_bigrams'].head(15).iterrows():
                f.write(f"{idx+1:2d}. {row['ngram']:40s} (count: {int(row['frequency'])})\n")
        else:
            f.write("No bigrams found.\n")
        f.write("\n")

        # Top Trigrams
        f.write("-" * 80 + "\n")
        f.write("TOP TRIGRAMS (3-word phrases)\n")
        f.write("-" * 80 + "\n")
        if not results['top_trigrams'].empty:
            for idx, row in results['top_trigrams'].head(10).iterrows():
                f.write(f"{idx+1:2d}. {row['ngram']:50s} (count: {int(row['frequency'])})\n")
        else:
            f.write("No trigrams found.\n")
        f.write("\n")

        # Noun Phrases
        f.write("-" * 80 + "\n")
        f.write("COMMON NOUN PHRASES\n")
        f.write("-" * 80 + "\n")
        if not results['noun_phrases'].empty:
            for idx, row in results['noun_phrases'].head(15).iterrows():
                f.write(f"{idx+1:2d}. {row['noun_phrase']:40s} (count: {int(row['frequency'])})\n")
        else:
            f.write("No noun phrases found.\n")
        f.write("\n")

        # Pain Points
        f.write("=" * 80 + "\n")
        f.write("PAIN POINTS (from negative reviews, rating <= 3)\n")
        f.write("=" * 80 + "\n")
        if not results['pain_points'].empty:
            for idx, row in results['pain_points'].head(15).iterrows():
                f.write(f"{idx+1:2d}. {row['pain_point']:40s} (score: {row['score']:.2f}, type: {row['type']})\n")
        else:
            f.write("No pain points identified.\n")
        f.write("\n")

        # Positive Drivers
        f.write("=" * 80 + "\n")
        f.write("POSITIVE DRIVERS (from positive reviews, rating >= 4)\n")
        f.write("=" * 80 + "\n")
        if not results['positive_drivers'].empty:
            for idx, row in results['positive_drivers'].head(15).iterrows():
                f.write(f"{idx+1:2d}. {row['driver']:40s} (score: {row['score']:.2f}, type: {row['type']})\n")
        else:
            f.write("No positive drivers identified.\n")
        f.write("\n")

        # Summary insights
        f.write("=" * 80 + "\n")
        f.write("KEY INSIGHTS & RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n")

        # Generate automated insights based on pain points
        if not results['pain_points'].empty:
            top_pain = results['pain_points'].head(3)
            f.write("\nTop 3 Areas for Improvement:\n")
            for idx, row in top_pain.iterrows():
                f.write(f"  • {row['pain_point']}\n")

        if not results['positive_drivers'].empty:
            top_drivers = results['positive_drivers'].head(3)
            f.write("\nTop 3 Strengths to Maintain:\n")
            for idx, row in top_drivers.iterrows():
                f.write(f"  • {row['driver']}\n")

        f.write("\n")

    print(f"Report saved to: {report_path}")


def generate_comparison_report(all_results: list, output_dir: Path) -> None:
    """
    Generate a comparison report across all banks.

    Args:
        all_results: List of analysis results for all banks
        output_dir: Directory to save the report
    """
    report_path = output_dir / "bank_comparison_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CROSS-BANK COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Summary table
        f.write("OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Bank':<15} {'Total Reviews':<15} {'Avg Rating':<15}\n")
        f.write("-" * 80 + "\n")

        for result in all_results:
            bank = result['bank']
            total = result['total_reviews']
            avg_rating = result['avg_rating'] if result['avg_rating'] else 0.0
            f.write(f"{bank:<15} {total:<15} {avg_rating:<15.2f}\n")

        f.write("\n\n")

        # Compare pain points
        f.write("=" * 80 + "\n")
        f.write("PAIN POINTS COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        for result in all_results:
            bank = result['bank']
            f.write(f"{bank}:\n")
            if not result['pain_points'].empty:
                for idx, row in result['pain_points'].head(5).iterrows():
                    f.write(f"  {idx+1}. {row['pain_point']}\n")
            else:
                f.write("  No pain points identified.\n")
            f.write("\n")

        # Compare positive drivers
        f.write("=" * 80 + "\n")
        f.write("POSITIVE DRIVERS COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        for result in all_results:
            bank = result['bank']
            f.write(f"{bank}:\n")
            if not result['positive_drivers'].empty:
                for idx, row in result['positive_drivers'].head(5).iterrows():
                    f.write(f"  {idx+1}. {row['driver']}\n")
            else:
                f.write("  No positive drivers identified.\n")
            f.write("\n")

    print(f"Comparison report saved to: {report_path}")


def main():
    """Main analysis workflow."""
    # Setup paths
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "processed" / "bank_reviews_clean.csv"
    output_dir = base_dir / "analysis" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if data exists
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please run the following commands first:")
        print("  1. python scripts/scrape_reviews.py")
        print("  2. python scripts/preprocess_reviews.py")
        return

    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} reviews")
    print(f"Banks: {df['bank'].unique().tolist()}")

    # Analyze each bank
    all_results = []

    for bank in sorted(df['bank'].unique()):
        print(f"\n{'='*80}")
        print(f"Analyzing {bank}...")
        print(f"{'='*80}")

        df_bank = df[df['bank'] == bank]
        results = analyze_bank_themes(
            df_bank,
            bank,
            text_column='review',
            rating_column='rating'
        )

        all_results.append(results)

        # Generate individual bank report
        generate_bank_report(results, output_dir)

        # Print summary to console
        print(f"\n{bank} Summary:")
        print(f"  Total reviews: {results['total_reviews']}")
        print(f"  Average rating: {results['avg_rating']:.2f}")
        print(f"\n  Top 5 Pain Points:")
        if not results['pain_points'].empty:
            for idx, row in results['pain_points'].head(5).iterrows():
                print(f"    {idx+1}. {row['pain_point']}")
        print(f"\n  Top 5 Positive Drivers:")
        if not results['positive_drivers'].empty:
            for idx, row in results['positive_drivers'].head(5).iterrows():
                print(f"    {idx+1}. {row['driver']}")

    # Generate comparison report
    print(f"\n{'='*80}")
    print("Generating cross-bank comparison report...")
    print(f"{'='*80}")
    generate_comparison_report(all_results, output_dir)

    # Save detailed results to CSV
    for result in all_results:
        bank = result['bank']

        # Save keywords
        keywords_path = output_dir / f"{bank.lower()}_keywords.csv"
        result['top_keywords'].to_csv(keywords_path, index=False)

        # Save pain points
        pain_path = output_dir / f"{bank.lower()}_pain_points.csv"
        result['pain_points'].to_csv(pain_path, index=False)

        # Save positive drivers
        drivers_path = output_dir / f"{bank.lower()}_positive_drivers.csv"
        result['positive_drivers'].to_csv(drivers_path, index=False)

        print(f"Saved detailed CSV reports for {bank}")

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Reports saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
