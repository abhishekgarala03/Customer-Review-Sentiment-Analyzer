import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import os
import re

def build_swiggy_sentiment_analyzer():
    """
    NLP model for Swiggy customer review analysis
    WHY:
    - Directly addresses "Experience in Generative AI" and "NLP" requirements
    - Uses industry-standard Hugging Face
    - Focuses on business impact: customer retention and satisfaction
    """
    # Load Swiggy-prepared data
    df = pd.read_csv('swiggy_review_data.csv')
    
    # Initialize sentiment pipeline
    print("⏳ Loading pre-trained model (takes < 30s)...")
    
    # Use distilled model for speed
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model=model_name,
        framework="pt",
        device=0 if torch.cuda.is_available() else -1
    )
    
    print(f"✅ Loaded {model_name} for Swiggy CX analysis")
    
    # Analyze sample reviews
    sample_reviews = [
        "The biryani was cold and arrived 45 minutes late",
        "Amazing delivery! Food arrived hot and fresh",
        "Packaging was damaged, but food was good"
    ]
    
    print("\n🔍 Sample Review Analysis:")
    for review in sample_reviews:
        result = sentiment_pipeline(review)[0]
        label = "Positive" if result['label'] == 'POSITIVE' else "Negative"
        print(f"• '{review[:50]}...' → {label} ({result['score']:.2%} confidence)")
    
    # Business impact evaluation
    # Calculate current sentiment baseline
    positive_count = 0
    total = min(500, len(df))  # Analyze subset for speed
    
    print(f"\n📊 Analyzing {total} reviews for business impact...")
    for review in df['full_review'].head(total):
        result = sentiment_pipeline(review)[0]
        if result['label'] == 'POSITIVE':
            positive_count += 1
    
    current_positive_rate = positive_count / total
    print(f"✅ Current positive sentiment rate: {current_positive_rate:.1%}")
    
    # Swiggy business impact calculation
    baseline_positive_rate = 0.65  # Industry standard for food delivery
    improvement = ((current_positive_rate - baseline_positive_rate) / baseline_positive_rate) * 100
    
    print(f"📈 Swiggy Impact: {improvement:.1f}% {'better' if improvement > 0 else 'worse'} than baseline → "
          f"potential for {'higher' if improvement > 0 else 'lower'} customer retention")
    
    # Save pipeline components
    # Note: Don't save the full pipeline (too large), but show how to recreate
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    joblib.dump({
        'model_name': model_name,
        'positive_threshold': 0.6
    }, 'swiggy_sentiment_config.pkl')
    
    print("💾 Configuration saved for production deployment")
    
    return sentiment_pipeline, current_positive_rate, improvement

def analyze_swiggy_review(review_text, sentiment_pipeline=None):
    """
    Swiggy-style sentiment analysis with business context
    WHY
    - Shows "end-to-end inference solutions at Swiggy scale"
    - Includes business context for CX improvement decisions
    - Ready for integration with Swiggy's review system
    """
    # Initialize pipeline if not provided
    if sentiment_pipeline is None:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=model_name,
            framework="pt",
            device=0 if torch.cuda.is_available() else -1
        )
    
    # Clean review text
    clean_text = re.sub(r'[^\w\s.,!?]', '', review_text).strip()
    
    # Get sentiment
    result = sentiment_pipeline(clean_text)[0]
    sentiment_label = "Positive" if result['label'] == 'POSITIVE' else "Negative"
    confidence = result['score']
    
    # Extract Swiggy-specific issues
    issue_keywords = {
        'delivery_time': ['late', 'delay', 'waiting', 'delayed', 'slow', 'minutes', 'hour', 'overtime'],
        'food_quality': ['cold', 'hot', 'taste', 'flavor', 'tasty', 'delicious', 'bad', 'spoiled', 'fresh'],
        'packaging': ['package', 'packaging', 'container', 'leak', 'spill', 'broken', 'damage', 'messy'],
        'order_accuracy': ['wrong', 'missing', 'incorrect', 'extra', 'substitute', 'mistake', 'error'],
        'delivery_partner': ['rude', 'friendly', 'courteous', 'behavior', 'attitude', 'delivery boy', 'partner']
    }
    
    detected_issues = []
    for issue, keywords in issue_keywords.items():
        if any(re.search(r'\b' + keyword + r'\b', clean_text, re.IGNORECASE) for keyword in keywords):
            detected_issues.append(issue.replace('_', ' ').title())
    
    # Convert to Swiggy business terms
    # Calculate business impact
    churn_risk = "High" if sentiment_label == "Negative" and confidence > 0.8 else "Medium" if sentiment_label == "Negative" else "Low"
    business_impact = "Critical" if 'Delivery Time' in detected_issues and sentiment_label == "Negative" else "High" if detected_issues else "Medium"
    
    return {
        'original_text': review_text,
        'clean_text': clean_text,
        'sentiment': sentiment_label,
        'confidence': f"{confidence:.1%}",
        'detected_issues': detected_issues,
        'churn_risk': churn_risk,
        'business_impact': business_impact,
        'swiggy_action_items': generate_swiggy_action_items(sentiment_label, detected_issues)
    }

def generate_swiggy_action_items(sentiment, issues):
    """
    Creates Swiggy-specific action items based on sentiment
    WHY:
    - Directly connects NLP to "business metrics"
    - Shows understanding of Swiggy's CX operations
    - Proves "ownership from inception to delivery" mindset
    """
    action_items = []
    
    if sentiment == "Negative":
        if 'Delivery Time' in issues:
            action_items.append("CALLTYPE: Notify delivery partner manager about routing issues")
            action_items.append("OFFER: 20% off next order for affected customer")
        if 'Food Quality' in issues:
            action_items.append("CALLTYPE: Contact restaurant about quality control")
            action_items.append("OFFER: Refund 50% of order value")
        if 'Packaging' in issues:
            action_items.append("CALLTYPE: Review packaging standards with restaurant")
        if 'Order Accuracy' in issues:
            action_items.append("CALLTYPE: Investigate POS system at restaurant")
        if 'Delivery Partner' in issues:
            action_items.append("CALLTYPE: Retrain delivery partner on customer service")
            action_items.append("OFFER: Free delivery credit")
    
    # Default actions for negative sentiment
    if sentiment == "Negative" and not action_items:
        action_items.append("CALLTYPE: Customer care follow-up within 1 hour")
        action_items.append("OFFER: 15% discount on next order")
    
    # Positive sentiment actions
    if sentiment == "Positive":
        if 'Delivery Time' in issues:
            action_items.append("RECOGNIZE: Commend delivery partner for speed")
        if 'Food Quality' in issues:
            action_items.append("RECOGNIZE: Feature restaurant in 'Top Quality' section")
        if not issues:
            action_items.append("REWARD: Offer referral bonus to customer")
    
    return action_items[:3]  # Limit to top 3 actions (Swiggy needs concise insights)

def calculate_cx_impact(analysis_results):
    """
    Translates sentiment to Swiggy business metrics
    WHY:
    - Directly connects NLP to "business metrics"
    - Shows understanding of Swiggy's customer retention
    - Proves "ownership from inception to delivery" mindset
    """
    # Calculate metrics
    total_reviews = len(analysis_results)
    positive_count = sum(1 for r in analysis_results if r['sentiment'] == "Positive")
    negative_count = total_reviews - positive_count
    
    # Swiggy-specific CX metrics
    delivery_issues = sum(1 for r in analysis_results if 'Delivery Time' in r['detected_issues'])
    food_issues = sum(1 for r in analysis_results if 'Food Quality' in r['detected_issues'])
    
    # Business impact calculation
    churn_risk_high = sum(1 for r in analysis_results if r['churn_risk'] == "High")
    potential_churn = (churn_risk_high / total_reviews) * 100
    
    # Calculate revenue impact (Swiggy cares about this)
    avg_order_value = 350  # Swiggy's average order value (INR)
    customers_at_risk = churn_risk_high
    revenue_at_risk = customers_at_risk * avg_order_value * 5  # 5 future orders
    
    return {
        'total_reviews': total_reviews,
        'positive_rate': f"{positive_count/total_reviews:.1%}",
        'negative_rate': f"{negative_count/total_reviews:.1%}",
        'delivery_issues_count': delivery_issues,
        'food_issues_count': food_issues,
        'customers_at_risk': customers_at_risk,
        'revenue_at_risk': f"₹{revenue_at_risk:,.0f}",
        'key_issues': determine_key_issues(analysis_results)
    }

def determine_key_issues(analysis_results):
    """Identify top issues for Swiggy's CX team"""
    issue_counts = {
        'Delivery Time': 0,
        'Food Quality': 0,
        'Packaging': 0,
        'Order Accuracy': 0,
        'Delivery Partner': 0
    }
    
    for result in analysis_results:
        for issue in result['detected_issues']:
            issue_counts[issue] += 1
    
    # Sort and return top 2 issues
    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
    return [issue for issue, count in sorted_issues[:2] if count > 0]

if __name__ == "__main__":
    # Build and evaluate model
    sentiment_pipeline, positive_rate, improvement = build_swiggy_sentiment_analyzer()
    
    # Analyze sample review
    sample_review = "The biryani was cold and arrived 45 minutes late. Packaging was also damaged."
    analysis = analyze_swiggy_review(sample_review, sentiment_pipeline)
    print("\n📝 Sample Review Analysis:")
    print(f"• Original: '{sample_review}'")
    print(f"• Sentiment: {analysis['sentiment']} ({analysis['confidence']})")
    print(f"• Issues Detected: {', '.join(analysis['detected_issues'])}")
    print(f"• Churn Risk: {analysis['churn_risk']}")
    print("• Action Items:")
    for item in analysis['swiggy_action_items']:
        print(f"  - {item}")
    
    # Calculate CX impact
    cx_impact = calculate_cx_impact([analysis])
    print("\n📊 Swiggy CX Impact:")
    print(f"• Negative Review Impact: {cx_impact['revenue_at_risk']} revenue at risk")
    print(f"• Top Issues: {', '.join(cx_impact['key_issues'])}")
