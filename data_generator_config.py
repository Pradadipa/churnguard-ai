"""
Configuration for Synthetic Training Data Generation
ChurnGuard AI - Week 1, Days 1-2
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class FeatureRanges:
    """Define realistic ranges for all features"""
    
    # Business metrics
    monthly_value: Tuple[float, float] = (100.0, 15000.0)  # Log-normal distribution
    contract_length_months: Tuple[int, ...] = (1, 3, 6, 12, 18, 24, 36)
    days_to_renewal: Tuple[int, int] = (-60, 365)  # Negative = expired!
    
    # Sentiment metrics
    avg_sentiment: Tuple[float, float] = (-1.0, 1.0)
    sentiment_std: Tuple[float, float] = (0.0, 0.5)
    negative_ratio: Tuple[float, float] = (0.0, 1.0)
    urgent_count: Tuple[int, int] = (0, 20)
    
    # Behavioral metrics
    response_time_hours: Tuple[float, float] = (0.25, 120.0)
    meeting_attendance_rate: Tuple[float, float] = (0.0, 1.0)
    feature_adoption_score: Tuple[float, float] = (0.0, 1.0)
    
    # Risk flags (binary)
    technical_issue_flag: Tuple[int, int] = (0, 1)
    pricing_concern_flag: Tuple[int, int] = (0, 1)
    competitor_mention_flag: Tuple[int, int] = (0, 1)


@dataclass
class CustomerTypeConfig:
    """Configuration for each customer segment"""
    name: str
    proportion: float
    churn_rate: float
    
    # Mean values for each feature (will add noise)
    avg_sentiment_mean: float
    sentiment_std_mean: float
    negative_ratio_mean: float
    urgent_count_mean: float
    
    monthly_value_mean: float
    days_to_renewal_mean: float
    response_time_mean: float
    meeting_attendance_mean: float
    feature_adoption_mean: float
    
    technical_issue_prob: float
    pricing_concern_prob: float
    competitor_mention_prob: float
    
    # Noise level (std deviation as % of mean)
    noise_level: float = 0.25  # 25% noise


# Define 4 customer types with realistic profiles
CUSTOMER_TYPES = [
    CustomerTypeConfig(
        name="happy_loyal",
        proportion=0.40,
        churn_rate=0.05,  # 5% unexpected churn
        
        # Very positive sentiment
        avg_sentiment_mean=0.65,
        sentiment_std_mean=0.15,
        negative_ratio_mean=0.10,
        urgent_count_mean=1.0,
        
        # Strong business metrics
        monthly_value_mean=5000.0,
        days_to_renewal_mean=180.0,
        response_time_mean=4.0,
        meeting_attendance_mean=0.85,
        feature_adoption_mean=0.75,
        
        # Low risk flags
        technical_issue_prob=0.10,
        pricing_concern_prob=0.05,
        competitor_mention_prob=0.05,
        
        noise_level=0.20
    ),
    
    CustomerTypeConfig(
        name="satisfied",
        proportion=0.25,
        churn_rate=0.15,  # 15% churn
        
        # Neutral to positive sentiment
        avg_sentiment_mean=0.25,
        sentiment_std_mean=0.25,
        negative_ratio_mean=0.30,
        urgent_count_mean=3.0,
        
        # Average business metrics
        monthly_value_mean=2000.0,
        days_to_renewal_mean=120.0,
        response_time_mean=12.0,
        meeting_attendance_mean=0.65,
        feature_adoption_mean=0.50,
        
        # Moderate risk flags
        technical_issue_prob=0.25,
        pricing_concern_prob=0.20,
        competitor_mention_prob=0.15,
        
        noise_level=0.25
    ),
    
    CustomerTypeConfig(
        name="at_risk",
        proportion=0.20,
        churn_rate=0.50,  # 50% churn (half can be saved!)
        
        # Negative sentiment
        avg_sentiment_mean=-0.20,
        sentiment_std_mean=0.35,
        negative_ratio_mean=0.55,
        urgent_count_mean=7.0,
        
        # Declining business metrics
        monthly_value_mean=1200.0,
        days_to_renewal_mean=45.0,  # Near renewal!
        response_time_mean=24.0,
        meeting_attendance_mean=0.40,
        feature_adoption_mean=0.30,
        
        # High risk flags
        technical_issue_prob=0.55,
        pricing_concern_prob=0.50,
        competitor_mention_prob=0.40,
        
        noise_level=0.30
    ),
    
    CustomerTypeConfig(
        name="churning",
        proportion=0.15,
        churn_rate=0.85,  # 85% churn (some false alarms)
        
        # Very negative sentiment
        avg_sentiment_mean=-0.60,
        sentiment_std_mean=0.40,
        negative_ratio_mean=0.75,
        urgent_count_mean=12.0,
        
        # Poor business metrics
        monthly_value_mean=800.0,
        days_to_renewal_mean=15.0,  # Critical renewal window!
        response_time_mean=48.0,
        meeting_attendance_mean=0.20,
        feature_adoption_mean=0.15,
        
        # Very high risk flags
        technical_issue_prob=0.75,
        pricing_concern_prob=0.70,
        competitor_mention_prob=0.65,
        
        noise_level=0.25
    )
]


# Correlation targets for validation
TARGET_CORRELATIONS = {
    'avg_sentiment_vs_churn': (-0.60, -0.70),  # Strong negative
    'days_to_renewal_vs_churn': (-0.45, -0.55),  # Moderate negative (closer = more risk)
    'negative_ratio_vs_churn': (0.55, 0.65),  # Strong positive
    'technical_issue_vs_churn': (0.40, 0.50),  # Moderate positive
    'meeting_attendance_vs_churn': (-0.35, -0.45),  # Moderate negative
    'feature_adoption_vs_churn': (-0.30, -0.40),  # Moderate negative
}


# Temporal urgency boost
def apply_renewal_urgency(days_to_renewal: np.ndarray) -> np.ndarray:
    """
    Apply urgency multiplier based on days to renewal
    < 30 days: 2x urgency
    < 0 days (expired): 3x urgency
    """
    urgency_multiplier = np.ones_like(days_to_renewal, dtype=float)
    urgency_multiplier[days_to_renewal < 30] = 2.0
    urgency_multiplier[days_to_renewal < 0] = 3.0
    return urgency_multiplier


# Output configuration
OUTPUT_CONFIG = {
    'csv_path': 'data/synthetic_training_data.csv',
    'pickle_path': 'data/synthetic_training_data.pkl',
    'validation_report_path': 'data/validation_report.txt',
    'random_seed': 42,
    'n_samples': 1000
}
