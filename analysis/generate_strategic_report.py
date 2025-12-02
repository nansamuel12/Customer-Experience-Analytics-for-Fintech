"""
Strategic Insights Report Generator
Generates a high-level strategic report with actionable recommendations based on NLP analysis.
"""

import sys
from pathlib import Path
import pandas as pd

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from keyword_extraction import analyze_bank_themes

def get_recommendations(pain_points_df):
    """
    Generate actionable recommendations based on top pain points.
    
    Args:
        pain_points_df: DataFrame containing pain points
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    if pain_points_df.empty:
        return ["Conduct broader user research to identify hidden friction points."]
        
    # Get top 5 pain points as a list of strings
    top_pain_points = pain_points_df.head(5)['pain_point'].tolist()
    
    # Define rules for recommendations
    rules = {
        ('login', 'password', 'signin', 'access', 'log in'): 
            "Streamline the authentication process. Consider implementing biometric login or extending session timeouts to reduce login friction.",
        ('crash', 'close', 'force', 'bug', 'error', 'working'): 
            "Prioritize app stability. Establish a dedicated task force to address crash reports and implement automated UI testing.",
        ('slow', 'lag', 'loading', 'wait', 'time', 'fast'): 
            "Optimize performance. Audit network requests and reduce app startup time to improve responsiveness.",
        ('update', 'version', 'install'): 
            "Improve the update experience. Ensure backward compatibility and provide clear release notes. Investigate issues with the latest release.",
        ('transfer', 'send', 'transaction', 'payment'): 
            "Simplify the transaction flow. Reduce the number of steps required to send money and provide clearer success/failure feedback.",
        ('otp', 'code', 'sms'): 
            "Enhance OTP delivery reliability. Consider alternative verification methods like email or push notifications.",
        ('network', 'connection', 'internet'): 
            "Optimize for low-bandwidth conditions. Implement offline mode capabilities and better error handling for network timeouts.",
        ('service', 'support', 'customer'): 
            "Revamp customer support integration. Add in-app chat support or a clearer help center to resolve issues faster."
    }
    
    used_recommendations = set()
    
    for point in top_pain_points:
        point_lower = point.lower()
        matched = False
        for keywords, recommendation in rules.items():
            if any(k in point_lower for k in keywords):
                if recommendation not in used_recommendations:
                    recommendations.append(f"**Address '{point}'**: {recommendation}")
                    used_recommendations.add(recommendation)
                    matched = True
                    break
        
        if not matched and len(recommendations) < 3:
             recommendations.append(f"**Investigate '{point}'**: Conduct specific user testing to understand the root cause of dissatisfaction related to '{point}'.")

    return recommendations[:3] # Return top 3 recommendations

def generate_strategic_report():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "processed" / "bank_reviews_clean.csv"
    output_dir = base_dir / "analysis" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "strategic_insights_report.md"

    if not data_path.exists():
        print(f"Data not found at {data_path}")
        return

    print("Loading data...")
    df = pd.read_csv(data_path)
    
    print("Generating strategic report...")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Strategic Customer Experience Report\n\n")
        f.write("This report outlines key satisfaction drivers, pain points, and actionable recommendations for each bank based on customer review analysis.\n\n")
        
        for bank in sorted(df['bank'].unique()):
            print(f"Analyzing {bank}...")
            df_bank = df[df['bank'] == bank]
            
            # Run analysis
            results = analyze_bank_themes(
                df_bank, 
                bank, 
                text_column='review', 
                rating_column='rating'
            )
            
            f.write(f"## {bank}\n\n")
            
            # 1. Satisfaction Drivers
            f.write("### 🌟 Top Satisfaction Drivers\n")
            f.write("What customers love about the app:\n\n")
            if not results['positive_drivers'].empty:
                for idx, row in results['positive_drivers'].head(3).iterrows():
                    f.write(f"1. **{row['driver'].title()}** (Score: {row['score']:.1f})\n")
            else:
                f.write("No clear satisfaction drivers identified.\n")
            f.write("\n")
            
            # 2. Pain Points
            f.write("### ⚠️ Top Pain Points\n")
            f.write("Key areas causing friction:\n\n")
            if not results['pain_points'].empty:
                for idx, row in results['pain_points'].head(3).iterrows():
                    f.write(f"1. **{row['pain_point'].title()}** (Score: {row['score']:.1f})\n")
            else:
                f.write("No clear pain points identified.\n")
            f.write("\n")
            
            # 3. Actionable Recommendations
            f.write("### 🚀 Actionable Recommendations\n")
            f.write("Steps to improve customer experience:\n\n")
            
            recommendations = get_recommendations(results['pain_points'])
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            else:
                f.write("1. Continue monitoring feedback for emerging issues.\n")
            
            f.write("\n---\n\n")
            
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    generate_strategic_report()
