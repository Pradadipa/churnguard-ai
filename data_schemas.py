"""
Data Schemas for ChurnGuard AI System
Defines the structure of data that flows between agents
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Communication:
    """Single customer communication"""
    id: str
    type: str  # 'slack', 'email', 'support_ticket'
    timestamp: str
    content: str
    source: str
    metadata: Dict[str, Any]

@dataclass 
class CustomerProfile:
    """Customer profile information"""
    customer_id: str
    company_name: str
    industry: str
    subscription_tier: str
    monthly_value: float
    contract_start_date: str
    contract_end_date: str
    primary_contacts: List[Dict[str, str]]

@dataclass
class SentimentAnalysis:
    """Results from sentiment analysis"""
    communication_id: str
    overall_sentiment: str  # 'positive', 'negative', 'neutral'
    sentiment_score: float  # -1.0 to 1.0
    urgency_level: str     # 'low', 'medium', 'high', 'critical'
    key_topics: List[str]
    risk_indicators: List[str]
    individual_risk_score: int  # 0-100

@dataclass
class ChurnPrediction:
    """Churn risk prediction results"""
    overall_risk_score: float  # 0-100
    risk_factors: List[str]
    confidence_level: float    # 0.0-1.0
    urgency_level: str
    explanation: str

@dataclass
class RetentionStrategy:
    """Generated retention strategies"""
    immediate_actions: List[str]
    short_term_strategies: List[str]
    long_term_initiatives: List[str]
    success_probability: float
    estimated_timeline: str

@dataclass
class ExecutiveReport:
    """Final executive report"""
    customer_id: str
    summary: str
    churn_risk_percentage: float
    key_findings: List[str]
    recommended_actions: List[str]
    success_metrics: List[str]
    generated_at: str

# State Schema - What flows between agents
class AgentState:
    """
    The complete state object that flows between agents
    Each agent adds their results to this state
    """
    
    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        self.processing_stage = "initialized"
        self.timestamp = datetime.now().isoformat()
        
        # Agent 1 - Data Collector results
        self.raw_communications: List[Communication] = []
        self.customer_profile: Optional[CustomerProfile] = None
        
        # Agent 2 - Sentiment Analyzer results  
        self.sentiment_analyses: List[SentimentAnalysis] = []
        self.overall_sentiment_summary: Optional[Dict[str, Any]] = None
        
        # Agent 3 - Churn Predictor results
        self.churn_prediction: Optional[ChurnPrediction] = None
        
        # Agent 4 - Strategy Generator results
        self.retention_strategies: Optional[RetentionStrategy] = None
        
        # Agent 5 - Reporter results
        self.executive_report: Optional[ExecutiveReport] = None
        
        # System metadata
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.processing_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LangGraph compatibility"""
        return {
            "customer_id": self.customer_id,
            "processing_stage": self.processing_stage,
            "timestamp": self.timestamp,
            "raw_communications": [vars(comm) for comm in self.raw_communications],
            "customer_profile": vars(self.customer_profile) if self.customer_profile else None,
            "sentiment_analyses": [vars(analysis) for analysis in self.sentiment_analyses],
            "overall_sentiment_summary": self.overall_sentiment_summary,
            "churn_prediction": vars(self.churn_prediction) if self.churn_prediction else None,
            "retention_strategies": vars(self.retention_strategies) if self.retention_strategies else None,
            "executive_report": vars(self.executive_report) if self.executive_report else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "processing_time": self.processing_time
        }

# Demo Data Templates
DEMO_CUSTOMER_PROFILES = {
    "CUST001": CustomerProfile(
        customer_id="CUST001",
        company_name="TechCorp Solutions",
        industry="Software Development",
        subscription_tier="Premium",
        monthly_value=2500.0,
        contract_start_date="2024-01-15",
        contract_end_date="2024-12-15",
        primary_contacts=[
            {"name": "Sarah Johnson", "role": "Product Manager", "email": "sarah@techcorp.com"},
            {"name": "Mike Chen", "role": "Technical Lead", "email": "mike@techcorp.com"}
        ]
    ),
    "CUST002": CustomerProfile(
        customer_id="CUST002", 
        company_name="DataFlow Analytics",
        industry="Data Analytics",
        subscription_tier="Enterprise",
        monthly_value=5000.0,
        contract_start_date="2023-06-01",
        contract_end_date="2024-06-01",
        primary_contacts=[
            {"name": "Lisa Wang", "role": "CTO", "email": "lisa@dataflow.com"}
        ]
    )
}

DEMO_COMMUNICATIONS = {
    "CUST001": [
        Communication(
            id="comm_001",
            type="slack",
            timestamp="2024-09-10T09:30:00Z",
            content="The new dashboard feature is amazing! Our reporting time has decreased by 50%.",
            source="customer-success-channel",
            metadata={"channel": "customer-success", "user": "sarah.johnson", "reactions": 5}
        ),
        Communication(
            id="comm_002",
            type="email", 
            timestamp="2024-09-08T14:15:00Z",
            content="We've been experiencing slow response times during peak hours. This is affecting our daily operations and team productivity.",
            source="support@techcorp.com",
            metadata={"subject": "Performance Issues", "priority": "high", "ticket_id": "SUP-1234"}
        ),
        Communication(
            id="comm_003",
            type="support_ticket",
            timestamp="2024-09-07T11:45:00Z", 
            content="I noticed our usage limits seem to be hit more frequently. Are there any pricing changes coming that we should be aware of?",
            source="billing_inquiry",
            metadata={"category": "billing", "status": "open", "assigned_to": "billing_team"}
        )
    ]
}

def get_demo_customer_data(customer_id: str) -> Dict[str, Any]:
    """
    Get demo customer data for testing
    """
    return {
        "profile": DEMO_CUSTOMER_PROFILES.get(customer_id, DEMO_CUSTOMER_PROFILES["CUST001"]),
        "communications": DEMO_COMMUNICATIONS.get(customer_id, DEMO_COMMUNICATIONS["CUST001"])
    }
