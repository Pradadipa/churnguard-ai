import os
import json
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv

from langgraph.graph import Graph, START, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# load environment variables
load_dotenv() # Ensure this is called to load .env file

class ChurnGuardAgent:
    """
    Multi-Agent Customer Success System
    Agents: Data Collector → Sentiment Analyzer → Churn Predictor → Strategy Generator → Reporter
    """
    
    # Initialize the LLM
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), # Model name from env or default
            temperature=0.1  # Low temperature for consistent results
        )
    
    # Agent 1: Data Collector
    def data_collector_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collects and preprocesses customer communication data
        In production: This would pull from Slack, Gmail, support systems
        For now: Uses demo data generation
        """
        
        print("Data Collector Agent: Collecting data...")
        customer_data = self._generate_demo_customer_data(state.get("customer_id", "CUST001"))
        
        # Update state with collected data
        state['raw_communication'] = customer_data['communications']
        state['customer_profile'] = customer_data['profile']
        state['collection_time'] = datetime.now().isoformat()
        state['processing_stage'] = 'data_collected'
        
        print(f"Collected {len(customer_data['communications'])} communications.")
        return state
    
    # Agent 2: Sentiment Analyzer
    def sentiment_analysis_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes sentiment and communication patterns
        """
        
        print("Sentiment Analysis Agent: Analyzing sentiment...")
        
        communications = state.get("raw_communication", [])
        
        # Handle case with no communications
        if not communications:
            state['sentiment_analysis'] = {"error" : "No communications to analyze."}
            return state
        
        # Analyze each communication
        sentiment_results = []
        
        for comm in communications:
            analysis = self._analyze_single_communication(comm)
            sentiment_results.append(analysis)
            
        # Create overall sentiment summary
        overall_analysis = self._create_sentiment_summary(sentiment_results)
        
        state['sentiment_analysis'] = {
            "individual_analyses": sentiment_results,
            "overall_summary": overall_analysis,
            "analysis_time": datetime.now().isoformat()
        }
        state['processing_stage'] = 'sentiment_analyzed' 
        
        print("Sentiment analysis complete.")
        return state 
            
    # Agent 3: Churn Predictor
    def churn_predictor_agent(self, state: Dict[str, Any]) -> Dict[str,Any]:
        """
        Predicts churn risk using ML model + LLM insights
        """
        
        print("Churn Predictor Agent: Predicting churn risk...")
        
        # Get sentiment analysis results
        sentiment_data = state.get("sentiment_analysis", {})
        customer_profile = state.get("customer_profile", {})
        
        # Calculate churn risk score ( Simple heuristic for demo purposes )
        churn_analysis = self._calculate_churn_risk(sentiment_data, customer_profile)
        
        # return updated state
        state['churn_prediction'] = churn_analysis
        state['processing_stage'] = 'churn_predicted'
        
        print(f"Churn Risk Score: {churn_analysis['overall_risk_score']}")
        return state
    
    
    # Agent 4: Strategy Generator
    def strategy_generator_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate personalized retention strategies
        """
        
        print("Strategy Generator Agent: Generating retention strategies...")
        
        churn_data = state.get("churn_prediction", {})
        customer_profile = state.get("customer_profile", {})
        sentiment_data =state.get("sentiment_analysis", {})
        
        # Generate strategies using LLM
        strategies = self._generate_retention_strategies(churn_data, customer_profile, sentiment_data)
        
        # Update state
        state['retention_strategies'] = strategies
        state['processing_stage'] = 'strategies_generated'
        
        print(f"✅ Generated {len(strategies['immediate_actions'])} action items")
        return state

    # Agent 5: Reporter
    def reporter_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates executive reports and dashboards
        """
        print("📈 Reporter Agent: Generating executive summary...")
        
        # Compile all analyses into executive summary
        executive_report = self._create_executive_report(state)
        
        state["executive_report"] = executive_report
        state["processing_stage"] = "report_complete"
        state["final_timestamp"] = datetime.now().isoformat()
        
        print("Executive report generated.")
        return state
    
    # Create the workflow graph
    def create_workflow(self):
        """Create complate multi agent workflow
        """
        
        workflow = Graph()
        
        # add all agents as nodes
        workflow.add_node("data_collector", self.data_collector_agent)
        workflow.add_node("sentiment_analyzer", self.sentiment_analysis_agent)
        workflow.add_node("churn_predictor", self.churn_predictor_agent)
        workflow.add_node("strategy_generator", self.strategy_generator_agent)
        workflow.add_node("reporter", self.reporter_agent)
        
        # Define edges between nodes
        workflow.add_edge(START, "data_collector")
        workflow.add_edge("data_collector", "sentiment_analyzer")
        workflow.add_edge("sentiment_analyzer", "churn_predictor")
        workflow.add_edge("churn_predictor", "strategy_generator")
        workflow.add_edge("strategy_generator", "reporter")
        workflow.add_edge("reporter", END)
        
        return workflow.compile()
    
    # ===== Helper Method =====
    # Helper Method: Generate Demo Customer Data
    def _generate_demo_customer_data(self, customer_id: str) -> Dict[str, Any]:
        """
        Generates demo customer communication data
        """
        return {
            "profile": {
                "customer_id": customer_id,
                "company_name": "TechCorp Solutions",
                "subscription_tier": "Premium",
                "months_as_customer": 8,
                "monthly_value": 2500
            },
            "communications": [ 
                {
                    "id": "comm_001",
                    "type": "slack_message",
                    "date": "2024-09-01",
                    "content": "Hey team, the new dashboard feature is amazing! Really helps with our reporting.",
                    "channel": "customer-success"
                },
                {
                    "id": "comm_002", 
                    "type": "support_ticket",
                    "date": "2024-09-03",
                    "content": "We're experiencing slow load times during peak hours. This is affecting our daily operations.",
                    "priority": "high"
                },
                {
                    "id": "comm_003",
                    "type": "email",
                    "date": "2024-09-05", 
                    "content": "I noticed our usage has been limited lately. Are there any upcoming price changes we should know about?",
                    "sender": "cto@techcorp.com"
                }
            ]
        }
        
    # Helper Method: Analyze Single Communication
    def _analyze_single_communication(self, communication: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes a single communication for sentiment and key themes
        """
        prompt = f"""
        Analyze this customer communication:
        Type: {communication['type']}
        Date: {communication['date']}
        Content: "{communication['content']}"
        
        Provide:
        1. Sentiment: (Positive/Negative/Neutral)
        2. Urgency: (Low/Medium/High/Critical)
        3. Key Topics: (list main subjects)
        4. Risk Indicators: (any concerning patterns)
        5. Individual Risk Score: (0-100)
        
        Respond in JSON format.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
    # Helper Method: Calculate Churn Risk
    def _calculate_churn_risk(self, sentiment_data: Dict, customer_profile: Dict) -> Dict[str, Any]:
        """
        Calculates overall churn risk score based on sentiment and profile (placeholder for ML model)
        """
        return {
            "overall_risk_score": 65,  # Placeholder score
            "risk_factors": [
                "Multiple high-urgency support tickets",
                "Negative sentiment in recent communications"
            ],
            "confidence": 0.75,
            "recommendations": [
                "Schedule a proactive check-in call",
                "Offer a loyalty discount"
            ]}
        
    # Helper Method: Generate Retention Strategies    
    def _generate_retention_strategies(self, churn_data: Dict, customer_profile: Dict, sentiment_data: Dict) -> Dict[str, Any]:
        """Generate retention strategies using LLM"""
        return {
            "immediate_actions": [
                "Schedule performance review call within 24 hours",
                "Provide technical optimization consultation",
                "Offer pricing discussion meeting"
            ],
            "medium_term_strategies": [
                "Implement proactive monitoring",
                "Assign dedicated success manager",
                "Provide advanced training sessions"
            ],
            "success_metrics": [
                "Response time improvement",
                "Customer satisfaction score increase",
                "Retention confirmation within 30 days"
            ]
        }    
        
    # Helper Method: Create Sentiment Summary
    def _create_sentiment_summary(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Create overall sentiment summary"""
        return {
            "total_communications": len(analyses),
            "overall_sentiment": "Mixed",  # Will calculate properly later
            "trend": "Declining",  # Will implement trend analysis
            "key_concerns": ["Performance issues", "Pricing questions"]
        }    
        
    # Helper Method: Create Executive Report
    def _create_executive_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary"""
        return {
            "summary": "Customer TechCorp Solutions shows moderate churn risk requiring immediate attention",
            "key_metrics": {
                "churn_risk": state.get("churn_prediction", {}).get("overall_risk_score", 0),
                "sentiment_trend": "Declining",
                "action_items": 3
            },
            "recommendations": [
                "Immediate technical performance review",
                "Pricing discussion within 1 week", 
                "Enhanced support tier consideration"
            ]
        }
        
        
        

# ===== TESTING =====
def test_multi_agent_system():
    """Tests the multi-agent customer success system
    """
    
    print("=== Starting Multi-Agent Customer Success System Test ===")
    print("Initializing ChurnGuardAgent...")
    
    churnguard = ChurnGuardAgent()
    app = churnguard.create_workflow()
    
    # Run complete workflow
    result = app.invoke({
        "customer_id": "CUST001",
        "analysis_request": "full_customer_health_check"
    })
    
    # Display results
    print("\n📋 FINAL EXECUTIVE REPORT:")
    print("=" * 40)
    exec_report = result.get("executive_report", {})
    print(f"Summary: {exec_report.get('summary', 'No summary available')}")
    print(f"Churn Risk: {exec_report.get('key_metrics', {}).get('churn_risk', 'N/A')}%")
    print("\nRecommendations:")
    for i, rec in enumerate(exec_report.get('recommendations', []), 1):
        print(f"{i}. {rec}")
    
    return result

if __name__ == "__main__":
    test_multi_agent_system()      
            