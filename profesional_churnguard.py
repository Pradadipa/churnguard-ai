"""
Professional ChurnGuard AI System
Production-ready multi-agent system with proper data structures and error handling
"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import data structures
from data_schemas import (Communication, CustomerProfile, SentimentAnalysis, 
                          ChurnPrediction, RetentionStrategy, ExecutiveReport, get_demo_customer_data)

from utils.helpers import (
    setup_logging, safe_json_parse, calculate_time_until_renewal,
    categorize_urgency, ErrorHandler, performance_timer
)

from langgraph.graph import Graph, START, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage


# Load environment variables
load_dotenv()

class ProfessionalChurnGuardAI:
    """
    Professional Multi-Agent ChurnGuard AI System

    Features:
    - Structured data models
    - Comprehensive error handling
    - Professional logging
    - Performance monitoring
    - Robust API integration patterns
    """

    def __init__(self):
        """Initialize the Professional ChurnGuard AI System"""
        # Setup logging
        self.logger = setup_logging()
        self.logger.info("Initializing Profesional ChurnGuard AI System")

        # Setup error handling
        self.error_handler = ErrorHandler(self.logger)

        # Initialize AI model
        try:
            self.llm = ChatOpenAI(
                model = os.getenv("OPENAI_MODEL", "gpt-40-mini"),
                temperature = 0.1
            )
            self.logger.info("✅ AI model initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize AI model: {e}")
            raise

        # System configurations
        self.config = {
            "data_mode" : os.getenv("DATA_MODE", "demo"),  # 'demo' or 'live'
            "debug_mode" : os.getenv("DEBUG_MODE", "false").lower() == "true",
            "environtment" : os.getenv("ENVIRONMENT", "development")  # 'development' or 'production'    
        }

        self.logger.info(f"System configuration: {self.config}")

    # AGENT 1: Data Collector
    @performance_timer
    def data_collector_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Data Collector Agent

        Responsibilities:
        - Colect customer communications from multiple sources
        - Build comprehensive customer profile
        - Handle API failures
        - Structure data using defined schemas
        """
        self.logger.info("🔍 Data Collector Agent: Starting data collection...")

        try:
            customer_id = state.get("customer_id", "UNKNOWN")

            # Collect customer data (using demo data for now)
            demo_data = get_demo_customer_data(customer_id)

            # Convert to professional data structures
            customer_profile = demo_data["profile"]
            communications = demo_data["communications"]

            # Add collection metadata
            collection_metadata ={
                "sources_attempted": ["slack", "email", "support_tickets"],
                "sources_succesful": ["demo_data"],
                "collection_timestamp": datetime.now().isoformat(),
                "total_communications": len(communications)
            }

            # Update stat with structured data
            state.update({
                "customer_profile": customer_profile.__dict__, # Convert to dict
                "raw_communications": [comm.__dict__ for comm in communications],
                "collection_metadata": collection_metadata,
                "processing_stage": "data_collected"
            })

            self.logger.info(f"✅ Data collection successful for customer {customer_profile.company_name}")
            self.logger.info(f"   Collected {len(communications)} communications")

            return state
        
        except Exception as e:
            self.logger.error(f"❌ Data Collector Agent failed: {e}")
            return self.error_handler.handle_agent_error("Data Collector Agent", e, state)

    # AGENT 2: Sentiment Analyzer
    @performance_timer
    def sentiment_analyzer_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sentiment Analyzer Agent

        Responsibilities:
        - Analyze sentiment of each communication using structured prompts
        - Generate professional sentiment analysis objects
        - Calculate overall sentiment trends
        - Handle edge cases and API failures
        """
        self.logger.info("🧠 Sentiment Analyzer Agent: Starting sentiment analysis...")

        try:
            # Extract communications
            raw_communications = state.get("raw_communications", [])

            if not raw_communications:
                self.logger.warning("⚠️ No communications data to analyze")
                state['sentiment_analyses'] = ["error: no communications data"]
                return state
            
            # Analyze each communication
            sentiment_analyses = []

            for comm_dict in raw_communications:
                analysis = self._analyze_communications_sentiment(comm_dict)
                sentiment_analyses.append(analysis)

            # Calculate overall sentiment summary
            # For simplicity, we just count positive/negative/neutral

            overall_summary = self._calculate_sentiment_summary(sentiment_analyses)

            # Update satate with professional data structures
            state.update({
                "sentiment_analyses": [analysis.__dict__ for analysis in sentiment_analyses],
                "overall_sentiment_summary": overall_summary,
                "processing_stage": "sentiment_analyzed"
            })

            self.logger.info(f"✅ Sentiment analysis complete")
            self.logger.info(f"   Analyzed {len(sentiment_analyses)} communications")
            self.logger.info(f"   Overall sentiment: {overall_summary['overall_sentiment']}")

            return state

        except Exception as e:
            self.logger.error(f"❌ Sentiment Analyzer Agent failed: {e}")
            return self.error_handler.handle_agent_error("Sentiment Analyzer Agent", e, state)    

    def _analyze_communications_sentiment(self, comm_dict: Dict[str, Any]) -> SentimentAnalysis:
        """Analyze sentiment of a single communication and return structured result"""
        
        # Create professional promt
        prompt = f"""
        Your are a professional customer succes analyst, Analyze this customer communication:

        Communication Details:
        - Type: {comm_dict.get('type')}
        - Timestamp: {comm_dict.get('timestamp')}
        - Source: {comm_dict.get('source')}
        - Content: {comm_dict.get('content')}

        Provide analysis in the following JSON format:
        {{
            "overall_sentiment": "positive|negative|neutral",
            "sentiment_score": float (-1.0 to 1.0),
            "urgency_level": "low|medium|high|critical",
            "key_topics": ["topic1", "topic2", "topic3"],
            "risk_indicators": ["indicator1", "indicator2"],
            "individual_risk_score": <integer 0-100>
        }}

        Be specific and analytical. Focus on business implications.
        Respond only with the JSON object.
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            analysis_data = safe_json_parse(response.content)

            # Create professional SentimentAnalysis object
            return SentimentAnalysis(
                communication_id=comm_dict['id'],
                overall_sentiment=analysis_data.get("overall_sentiment", "neutral"),
                sentiment_score=float(analysis_data.get("sentiment_score", 0.0)),
                urgency_level=analysis_data.get("urgency_level", "medium"),
                key_topics=analysis_data.get("key_topics", []),
                risk_indicators=analysis_data.get("risk_indicators", []),
                individual_risk_score=int(analysis_data.get("individual_risk_score", 50)) 
            )
        
        except Exception as e:
            self.logger.error(f"Failed to analyze communication {comm_dict['id']}: {e}")
            return SentimentAnalysis(
                communication_id=comm_dict['id'],
                overall_sentiment="neutral",
                sentiment_score=0.0,
                urgency_level="medium",
                key_topics=['communication_analysis'],
                risk_indicators=["analysis_failed"],
                individual_risk_score=50
            )
    def _calculate_sentiment_summary(self, analyses: List[SentimentAnalysis]) -> Dict[str, Any]:
        """Calculate overall sentiment summary from individual analyses"""
        if not analyses:
            return {"error": "No analyses to summarize"}
        
        # Calculate metrics
        total_comms = len(analyses)
        avg_sentiment_score = sum(a.sentiment_score for a in analyses) / total_comms
        avg_risk_score = sum(a.individual_risk_score for a in analyses) / total_comms

        # Count sentiment distribution
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        urgency_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        for analysis in analyses:
            sentiment_counts[analysis.overall_sentiment] = sentiment_counts.get(analysis.overall_sentiment, 0) + 1
            urgency_counts[analysis.urgency_level] = urgency_counts.get(analysis.urgency_level, 0) + 1

        # Determine overall sentiment
        if avg_sentiment_score > 0.3:
            overall_sentiment = "positive"
        elif avg_sentiment_score < -0.3:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "mixed"

        # Collect all risk indicators
        all_risk_indicators = []
        for analysis in analyses:
            all_risk_indicators.extend(analysis.risk_indicators)

        return {
            "total_communications": total_comms,
            "average_sentiment_score": round(avg_sentiment_score, 3),
            "average_risk_score": round(avg_risk_score, 1),
            "overall_sentiment": overall_sentiment,
            "sentiment_distribution": sentiment_counts,
            "urgency_distribution": urgency_counts,
            "high_risk_communications": len([a for a in analyses if a.individual_risk_score > 70]),
            "urgent_communications": len([a for a in analyses if a.urgency_level in ["high", "critical"]]),
            "common_risk_indicators": list(set(all_risk_indicators)),
            "analysis_timestamp": datetime.now().isoformat()
        }            

    # AGENT 3: Churn Predictor
    @performance_timer
    def churn_predictor_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Churn Predictor Agent

        Responsibilities:
        - Predict churn risk using sentiment analysis and customer profile
        - Generate structured churn prediction objects
        - Provide confidence levels and explanations
        - Handle edge cases and API failures
        """
        self.logger.info("📉 Churn Predictor Agent: Calculating churn risk...")

        try:
            # Extract necessary data
            sentiment_summary = state.get("overall_sentiment_summary", {})
            customer_profile = state.get("customer_profile", {})

            # Calculate base risk from sentiment
            base_risk = sentiment_summary.get("average_risk_score", 50) 

            # Calculate additional risk factors
            risk_factors = self._calculate_risk_factors(customer_profile, sentiment_summary)

            # Calculate overall risk score
            additional_risk = sum(risk_factors.values())
            overall_risk = min(base_risk + additional_risk, 100)

            # Determine confidence and urgency
            confidence = self._calculate_confidence(sentiment_summary, customer_profile)
            urgency = categorize_urgency(overall_risk, risk_factors.get("contract_risk", 0))

            # Generate professional explanation
            explanation = self._generate_risk_explanation(overall_risk, risk_factors, customer_profile)

            # Create professional ChurnPrediction object
            churn_prediction = ChurnPrediction(
                overall_risk_score=round(overall_risk, 1),
                risk_factors=list(risk_factors.keys()),
                confidence_level=confidence,
                urgency_level=urgency,
                explanation=explanation
            )

            # Update state with structured data
            state.update({
                "churn_prediction": churn_prediction.__dict__,
                "risk_factor_details": risk_factors,
                "processing_stage": "churn_predicted"
            })

            self.logger.info(f"✅ Churn prediction complete")
            self.logger.info(f"   Overall risk score: {overall_risk:.1f}")
            self.logger.info(f"   Urgency level: {urgency}")
            self.logger.info(f"   Confidence level: {confidence}")

            return state
        
        except Exception as e:
            self.logger.error(f"❌ Churn Predictor Agent failed: {e}")
            return self.error_handler.handle_agent_error("Churn Predictor Agent", e, state)
        
    def _calculate_risk_factors(self, customer_profile: Dict[str, Any], sentiment_summary: Dict[str, Any]) -> Dict[str, float]:
        """ Calculate specific risk factors with professional scoring """

        risk_factors = {}

        # Contract timing risk
        try:
            days_to_renewal = calculate_time_until_renewal(customer_profile.get("contract_end_date", ""))
            if days_to_renewal <= 30:
                risk_factors["contract_renewal_imminent"] = 20
            elif days_to_renewal <= 90:
                risk_factors["contract_renewal_approaching"] = 10

        except:
            risk_factors["contract_date_unknown"] = 5


        # Communication volume risk
        total_comms = sentiment_summary.get("total_communications", 0)
        if total_comms < 2:
            risk_factors["low_engagement"] = 15
        elif total_comms < 5:
            risk_factors["moderate_engagement"] = 5

        # Urgency risk
        urgent_comms = sentiment_summary.get("urgent_communications", 0)
        if urgent_comms >= 2:
            risk_factors["multiple_urgent_issues"] = 15
        elif urgent_comms >= 1:
            risk_factors["urgent_issues_present"] = 8

        # Sentiment trend risk
        avg_sentiment = sentiment_summary.get("average_sentiment_score", 0)
        if avg_sentiment < -0.5:
            risk_factors["highly_negative_sentiment"] = 20
        elif avg_sentiment < -0.2:
            risk_factors["negative_sentiment_trend"] = 10

        # Hisgh-value customer risk
        monthly_value = customer_profile.get("monthly_value", 0)
        if monthly_value > 5000:
            # High-value customer get attention faster
            for key in list(risk_factors.keys()):
                risk_factors[key] * 1.5

        return risk_factors

    def _calculate_confidence(self, sentiment_summary: Dict[str, Any], customer_profile: Dict[str, Any]) -> str:
        """Calculate confidence in the churn prediction"""

        confidence_factors = []

        # More communications = higher confidence
        total_comms = sentiment_summary.get("total_communications", 0)
        confidence_factors.append(min(total_comms / 10, 1.0))  # Max weight 1.0

        # Recent communications = higher confidence
        confidence_factors.append(0.9)

        # Complete customer profile = higher confidence
        profile_completeness = len([v for v in customer_profile.values() if v]) / len(customer_profile)
        confidence_factors.append(profile_completeness)

        return round(sum(confidence_factors) / len(confidence_factors), 2)

    def _generate_risk_explanation(self, risk_score: float, risk_factors: Dict[str, float], customer_profile: Dict[str, Any]) -> str:
        """Generate a professional explanation for the churn risk prediction"""

        company_name = customer_profile.get("company_name", "Customer")
        monthly_value = customer_profile.get("monthly_value", 0)

        if risk_score >= 80:
            severity = "CRITICAL"
            action = "immediate executive intervention"
        elif risk_score >= 60:
            severity = "HIGH"  
            action = "urgent customer success outreach"
        elif risk_score >= 40:
            severity = "MODERATE"
            action = "proactive engagement"
        else:
            severity = "LOW"
            action = "standard monitoring"

        top_risks = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)[:3]

        explanation = f"{severity} CHURN RISK: {company_name} shows {risk_score:.1f}% churn probability requiring {action}." 

        if top_risks:
            risk_list = ", ".join([risk.replace("-", " ") for risk, _ in top_risks])
            explanation += f"Primary risk factors: {risk_list}."

        explanation += f"Customer value: ${monthly_value:,.0f}/month makes retention high priority"

        return explanation
    
    # AGENT 4: Retention Strategist
    @performance_timer
    def strategy_generator_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Professional Strategy Generator Agent
        
        Responsibilities:
        - Generate evidence-based retention strategies
        - Create actionable implementation plans
        - Estimate success probabilities
        - Provide timeline and resource requirements
        """
        self.logger.info("🎯 Strategy Generator Agent: Generating retention strategies ...")

        try:
            # Extract context
            churn_prediction = state.get("churn_prediction", {})
            customer_profile = state.get("customer_profile", {})
            sentiment_summary = state.get("overall_sentiment_summary", {})

            # Generate strategies using professional LLM prompts
            strategies = self._generate_professional_strategies(churn_prediction, customer_profile, sentiment_summary)

            # Update state
            state.update({
                "retention_strategies": strategies.__dict__,
                "processing_stage": "strategies_generated"
            })

            self.logger.info(f"✅ Retention strategies complete")
            self.logger.info(f"   Success probability: {strategies.success_probability:.0f}%")
            self.logger.info(f"   Recommended actions: {len(strategies.immediate_actions)}")
            
            return state
        
        except Exception as e:
            self.logger.error(f"❌ Strategy Generator Agent failed: {e}")
            return self.error_handler.handle_agent_error("StrategyGenerator Agent", e, state)
        
    def _generate_professional_strategies(self, churn_prediction: Dict[str, Any],
                                          customer_profile: Dict[str, Any],
                                          sentiment_summary: Dict[str, Any]) -> RetentionStrategy:
        
        """Generate professional retention strategies using LLM"""   
        
        # Create professional prompt
        prompt = f"""
            You are a senior customer success strategist. Generate a professional retention strategy.

            CUSTOMER CONTEXT:
            - Company: {customer_profile.get('company_name', 'Unknown')}
            - Monthly Value: ${customer_profile.get('monthly_value', 0):,.0f}
            - Industry: {customer_profile.get('industry', 'Unknown')}
            - Contract End: {customer_profile.get('contract_end_date', 'Unknown')}

            RISK ANALYSIS:
            - Churn Risk: {churn_prediction.get('overall_risk_score', 0):.1f}%
            - Urgency: {churn_prediction.get('urgency_level', 'medium')}
            - Key Risks: {', '.join(churn_prediction.get('risk_factors', []))}

            SENTIMENT ANALYSIS:
            - Overall Sentiment: {sentiment_summary.get('overall_sentiment', 'mixed')}
            - Urgent Issues: {sentiment_summary.get('urgent_communications', 0)}
            - Risk Indicators: {', '.join(sentiment_summary.get('common_risk_indicators', []))}

            Generate a JSON response with:
            {{
                "immediate_actions": ["action1", "action2", "action3"],
                "short_term_strategies": ["strategy1", "strategy2", "strategy3"], 
                "long_term_initiatives": ["initiative1", "initiative2"],
                "success_probability": <integer 0-100>,
                "estimated_timeline": "description of timeline"
            }}

            Focus on specific, actionable strategies. Consider the customer's industry and value.
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            strategy_data = safe_json_parse(response.content)

            return RetentionStrategy(
                immediate_actions = strategy_data.get("immediate_actions", [
                    "Schedule custome health check call",
                    "Review outstanding support issues",
                    "Prepare retention discussion"
                ]),
                short_term_strategies = strategy_data.get("short_term_strategies", [
                    "Implement weekly check-ins",
                    "Provide additional product training",
                    "Address technical concerns"
                ]),
                long_term_initiatives= strategy_data.get("long_term_initiatives", [
                    "Develop strategic partnership",
                    "Create custom success metrics"
                ]),
                success_probability=float(strategy_data.get('success_probability', 70)),
                estimated_timeline=strategy_data.get('estimated_timeline', "2-4 weeks for immediate actions")
            )
                
        
        except Exception as e:
            self.logger.warning(f"LLM strategy generation failed: {e}, using fallback")

            # Professional fallback strategies
            risk_score = churn_prediction.get("overall_risk_score", 50)

            if risk_score >= 70:
                return RetentionStrategy(
                    immediate_actions = [
                        "Schedule emergency customer success call within 24 hours",
                        "Escalate to account management and senior leadership",
                        "Prepare comprehensive retention offer with contract flexibility"
                    ],
                    short_term_strategies=[
                        "Implement daily check-in cadence until resolution",
                        "Fast-track resolution of all outstanding technical issues",
                        "Provide executive sponsor and dedicated support channel"
                    ],
                    long_term_initiatives=[
                        "Develop comprehensive customer success plan",
                        "Create custom integration and training program"
                    ],
                    success_probability=60.0,
                    estimated_timeline="Immediate action required - 24-48 hours"
                )
            else:
                return RetentionStrategy(
                    immediate_actions=[
                        "Schedule customer health check within 48 hours",
                        "Review and prioritize outstanding support requests",
                        "Proactively communicate recent product improvements"
                    ],
                    short_term_strategies=[
                        "Establish regular check-in cadence",
                        "Provide advanced feature training and best practices",
                        "Connect with customer success manager for ongoing support"
                    ],
                    long_term_initiatives=[
                        "Develop strategic partnership opportunities",
                        "Create quarterly business review process"
                    ],
                    success_probability=75.0,
                    estimated_timeline="1-2 weeks for immediate actions"
                )
            
    # AGENT 5: Executive Reporter
    @performance_timer
    def reporter_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Professional Reporter Agent
        
        Responsibilities:
        - Generate executive-quality reports
        - Create actionable dashboards
        - Provide clear recommendations
        - Include success tracking metrics
        """
        self.logger.info("📝 Reporter Agent: Generating executive report...")

        try:
            # Generate comprehensive executive report
            executive_report = self._create_professional_executive_report(state)

            # Create action alerts
            alerts = self._generate_action_alerts(state)

            # Update state
            state.update({
                "executive_report": executive_report.__dict__,
                "action_alerts": alerts,
                "processing_stage": "report_generated",
                "final_timestamp": datetime.now().isoformat()
            }) 

            self.logger.info(f"✅ Executive report generated successfully")

            return state
        
        except Exception as e:
            self.logger.error(f"❌ Reporter Agent failed: {e}")
            return self.error_handler.handle_agent_error("Reporter", e, state)
        
    def _create_professional_executive_report(self, state: Dict[str, Any]) -> ExecutiveReport:
        """Create comprehensive executive report"""

        # Extract data
        customer_profile = state.get("customer_profile", {})
        churn_prediction = state.get("churn_prediction", {})
        sentiment_summary = state.get("overall_sentiment_summary", {})
        retention_strategies = state.get("retention_strategies", {})

        # Generate professional summary
        risk_score = churn_prediction.get("overall_risk_score", 0)
        company_name = customer_profile.get("company_name", "Customer")
        monthly_value = customer_profile.get("monthly_value", 0)
        urgency = churn_prediction.get("urgency_level", "medium")

        summary = f"Customer {company_name} (${monthly_value:,.0f}/month) requires {urgency.upper()} priority attention with {risk_score:.1f}% churn risk."

        # Key findings
        key_findings = [
            f"Churn Risk Level: {urgency.upper()} ({risk_score:.1f}%)",
            f"Monthly Revenue at Risk: ${monthly_value:,.0f}",
            f"Communications Analyzed: {sentiment_summary.get('total_communications', 0)}",
            f"Urgent Issues: {sentiment_summary.get('urgent_communications', 0)}",
            f"Overall Sentiment: {sentiment_summary.get('overall_sentiment', 'mixed').title()}"
        ]
        
        # Contract timing
        try:
            days_to_renewal = calculate_time_until_renewal(customer_profile.get("contract_end_date", ""))
            key_findings.append(f"Days Until Contract Renewal: {days_to_renewal} days")
        except:
            key_findings.append("Days Until Contract Renewal: Date Unknown")

        return ExecutiveReport(
            customer_id=customer_profile.get("customer_id", "UNKNOWN"),
            summary=summary,
            churn_risk_percentage=risk_score,
            key_findings=key_findings,
            recommended_actions=retention_strategies.get("immediate_actions", []),
            success_metrics=[
                "Customer satisfaction score improvement",
                "Reduced support ticket volume",
                "Contract renewal confirmation",
                "Increased product engagement"
            ],
            generated_at=datetime.now().isoformat()
        )

    def _generate_action_alerts(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable alerts for customer success team"""
        
        alerts = []
        churn_prediction = state.get("churn_prediction", {})
        customer_profile = state.get("customer_profile", {})
        
        risk_score = churn_prediction.get("overall_risk_score", 0)
        urgency = churn_prediction.get("urgency_level", "medium")
        
        if risk_score >= 80 or urgency == "critical":
            alerts.append({
                "type": "critical",
                "priority": "immediate",
                "message": f"CRITICAL: {customer_profile.get('company_name', 'Customer')} requires immediate intervention",
                "action": "Schedule emergency customer success call within 24 hours",
                "owner": "Customer Success Manager + Account Executive",
                "deadline": (datetime.now() + timedelta(hours=24)).isoformat()
            })
        
        if risk_score >= 60 or urgency in ["high", "critical"]:
            alerts.append({
                "type": "warning",
                "priority": "high",
                "message": "High churn risk detected - proactive intervention needed",
                "action": "Schedule customer health check and retention discussion",
                "owner": "Customer Success Manager",
                "deadline": (datetime.now() + timedelta(hours=48)).isoformat()
            })
        
        return alerts

    # Create workflow graph
    def create_professional_workflow(self) -> Any:
        """Create the professional multi-agent workflow graph"""

        self.logger.info("🔗 Creating professional workflow graph...")

        # Create workflow graph
        workflow = Graph()

        # Add All agents as nodes
        workflow.add_node("data_collector", self.data_collector_agent)
        workflow.add_node("sentiment_analyzer", self.sentiment_analyzer_agent)
        workflow.add_node("churn_predictor", self.churn_predictor_agent)
        workflow.add_node("strategy_generator", self.strategy_generator_agent)
        workflow.add_node("reporter", self.reporter_agent)

        # Define workflow edges
        workflow.add_edge(START, "data_collector")
        workflow.add_edge("data_collector", "sentiment_analyzer")
        workflow.add_edge("sentiment_analyzer", "churn_predictor")
        workflow.add_edge("churn_predictor", "strategy_generator")
        workflow.add_edge("strategy_generator", "reporter")
        workflow.add_edge("reporter", END)

        self.logger.info("✅ Workflow graph created successfully")

        return workflow.compile()

# Profesional test run
def test_professional_system():
    """Test the complete professional system"""
    
    print("🚀 PROFESSIONAL CHURNGUARD AI SYSTEM")
    print("=" * 80)
    print("Production-ready multi-agent customer success system")
    print("=" * 80)
    
    try:
        # Initialize professional system
        print("\n⚙️  Initializing professional system...")
        churnguard = ProfessionalChurnGuardAI()
        
        # Create workflow
        print("🔗 Creating professional workflow...")
        app = churnguard.create_professional_workflow()
        
        # Execute analysis
        print("🔄 Executing professional customer analysis...\n")
        
        # Professional input
        professional_input = {
            "customer_id": "CUST001",
            "analysis_type": "comprehensive_health_assessment",
            "requested_by": "customer_success_team",
            "priority": "high"
        }
        
        # Execute workflow
        result = app.invoke(professional_input)
        
        # Display professional results
        print("\n" + "=" * 80)
        print("📊 EXECUTIVE DASHBOARD")
        print("=" * 80)
        
        # Executive Report
        exec_report_dict = result.get("executive_report", {})
        if exec_report_dict:
            print(f"\n📋 EXECUTIVE SUMMARY")
            print("-" * 40)
            print(f"{exec_report_dict.get('summary', 'No summary available')}")
            
            print(f"\n📈 KEY FINDINGS")
            print("-" * 40)
            for i, finding in enumerate(exec_report_dict.get('key_findings', []), 1):
                print(f"   {i}. {finding}")
            
            print(f"\n⚡ IMMEDIATE ACTIONS REQUIRED")
            print("-" * 40)
            for i, action in enumerate(exec_report_dict.get('recommended_actions', []), 1):
                print(f"   {i}. {action}")
            
            print(f"\n📊 SUCCESS METRICS TO TRACK")
            print("-" * 40)
            for i, metric in enumerate(exec_report_dict.get('success_metrics', []), 1):
                print(f"   {i}. {metric}")
        
        # Action Alerts
        alerts = result.get("action_alerts", [])
        if alerts:
            print(f"\n🚨 ACTION ALERTS")
            print("-" * 40)
            for alert in alerts:
                priority = alert.get('priority', 'medium').upper()
                message = alert.get('message', 'No message')
                action = alert.get('action', 'No action specified')
                owner = alert.get('owner', 'Unassigned')
                
                print(f"   🔔 {priority}: {message}")
                print(f"      Action: {action}")
                print(f"      Owner: {owner}")
                print()
        
        # System Performance
        print(f"\n⚡ SYSTEM PERFORMANCE")
        print("-" * 40)
        print(f"   Processing Stage: {result.get('processing_stage', 'Unknown')}")
        print(f"   Total Execution Time: {result.get('execution_time', 'N/A')} seconds")

        # Fixed data sources counting
        collection_metadata = result.get('collection_metadata', {})
        sources_successful = collection_metadata.get('sources_successful', ['demo_data'])
        print(f"   Data Sources: {len(sources_successful)}")

        print(f"   Communications Analyzed: {result.get('overall_sentiment_summary', {}).get('total_communications', 0)}")
        
        # Error Handling
        errors = result.get('errors', [])
        if errors:
            print(f"\n⚠️  SYSTEM WARNINGS")
            print("-" * 40)
            for error in errors:
                print(f"   • {error}")
        else:
            print(f"\n✅ SYSTEM STATUS: All agents executed successfully")
        
        print("\n" + "=" * 80)
        print("🎯 PROFESSIONAL ANALYSIS COMPLETE")
        print("=" * 80)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Professional system test failed: {e}")
        return None

if __name__ == "__main__":
    # Run the professional system test
    result = test_professional_system()
    
    if result:
        print(f"\n💡 NEXT STEPS:")
        print("   1. Review the executive report with customer success team")
        print("   2. Execute immediate actions within specified timeframes")
        print("   3. Monitor success metrics for improvement")
        print("   4. Schedule follow-up analysis in 2 weeks")
        print("\n🚀 Ready for production deployment!")      