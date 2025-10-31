import os
from typing import Dict, Any
from dotenv import load_dotenv

from langgraph.graph import Graph, START, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

class CustomerSentimentAgent:
    def __init__(self):
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1  # Low temperature for consistent results
        )
    
    def analyze_sentiment(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the sentiment of customer communication
        """
        customer_message = state.get("customer_message", "")
        
        # Create the prompt for sentiment analysis
        prompt = f"""
        Analyze the sentiment and urgency of this customer message.
        
        Customer Message: "{customer_message}"
        
        Please provide:
        1. Overall Sentiment: (Positive/Negative/Neutral)
        2. Urgency Level: (Low/Medium/High/Critical)
        3. Key Concerns: (list any issues mentioned)
        4. Churn Risk Score: (0-100, where 100 is highest risk)
        5. Recommended Action: (what should customer success do?)
        
        Format your response clearly with each point on a new line.
        """
        
        # Get response from LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Update state with analysis
        state["sentiment_analysis"] = response.content
        state["status"] = "analyzed"
        
        return state
    
    def create_workflow(self):
        """
        Create the LangGraph workflow
        """
        # Create a new graph
        workflow = Graph()
        
        # Add our sentiment analysis node
        workflow.add_node("analyze_sentiment", self.analyze_sentiment)
        
        # Define the flow: START -> analyze_sentiment -> END
        workflow.add_edge(START, "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", END)
        
        # Compile the graph
        return workflow.compile()

def test_agent():
    """
    Test our agent with sample customer messages
    """
    # Initialize agent
    agent = CustomerSentimentAgent()
    app = agent.create_workflow()
    
    # Test messages (realistic customer communications)
    test_messages = [
        "Hi team, I've been having issues with the dashboard loading slowly. It's been happening for 3 days now and it's affecting our daily reports. Can someone please help?",
        
        "I'm really frustrated! This is the third time this month that our data export has failed. We're paying premium for this service and it's not working. I'm considering switching to a competitor if this doesn't get fixed immediately.",
        
        "Just wanted to say thanks for the quick response yesterday! The new feature you added is exactly what we needed. Our team loves it!"
    ]
    
    print("🤖 ChurnGuard AI - Customer Sentiment Analyzer")
    print("=" * 50)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n📧 CUSTOMER MESSAGE {i}:")
        print(f"'{message[:60]}...'")
        print("\n🔍 ANALYSIS:")
        print("-" * 30)
        
        # Run the agent
        result = app.invoke({
            "customer_message": message,
            "status": "pending"
        })
        
        print(result["sentiment_analysis"])
        print("\n" + "="*50)

if __name__ == "__main__":
    test_agent()