"""
Helper functions for ChurnGuard AI System
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os

def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/churnguard.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('ChurnGuard')

def safe_json_parse(text: str, fallback: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Safely parse JSON from LLM responses
    Sometimes LLMs return invalid JSON, this handles it gracefully
    """
    if fallback is None:
        fallback = {"error": "Failed to parse JSON", "raw_text": text}
        
    try:
        # Try to parse as-is first
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            # Attempt to fix common issues
            if "```json" in text:
                json_start = text.find("```json") + 7 # Skip past ```json
                json_end = text.find("```", json_start) # Find closing ```  
                json_text = text[json_start:json_end].strip()
                return json.loads(json_text)
            
            text = text.strip()
            if text.startswith('{') and text.endswith('}'):
                return json.loads(text)
            
        except json.JSONDecodeError:
            pass
        
    return fallback

def calculate_time_until_renewal(contract_end_date: str) -> int:
    """
    Calculate days until contract renewal
    Expects contract_end_date in 'YYYY-MM-DD' format
    """
    try:
        end_date = datetime.fromisoformat(contract_end_date.replace("Z", "+00:00"))
        now = datetime.now()
        delta = end_date - now
        return max(0, delta.days)
    except:
        return 365  # Default to 1 year if parsing fails
    
def categorize_urgency(risk_score: float, days_to_renewal: int) -> str:
    """
    Categorize urgency based on risk score and days until renewal
    """
    if risk_score >= 80 or days_to_renewal <= 30:
        return "Critical"
    elif risk_score >= 60 or days_to_renewal <= 90:
        return "High"
    elif risk_score >= 40 or days_to_renewal <= 180:
        return "Medium"
    else:
        return "Low"
    
def format_currency(amout: float) -> str:
    """
    Format a number as currency
    """
    return f"${amout:,.2f}"

def calculate_customer_value(monthly_value: float, month_as_customer: int) -> Dict[str, float]:
    """Calculate various customer metrics"""
    return {
        "monthly_value" : monthly_value,
        "annual_value" : monthly_value * 12,
        "lifetime_value" : monthly_value * month_as_customer,
        "average_monthly_value": monthly_value
    }  

def extract_key_topics(communications: List[Dict[str,Any]]) -> List[str]:
    """Extract key topics from communications for analysis"""

    common_business_topics = [
        "performance", "pricing", "support", "features", "integration", 
        "renewal", "contract", "billing", "technical", "training"
    ]

    found_topics = []
    for comm in communications:
        content = comm.get("content", "").lower()
        for topic in common_business_topics:
            if topic in content and topic not in found_topics:
                found_topics.append(topic)

    return found_topics[:5]  # return top 5 topics

def validate_state(state: Dict[str, Any]) -> List[str]:
    """Validate state object for required fields"""
    errors = []

    if not state.get("customer_id"):
        errors.append("Missing customer_id")

    if state.get("processing_stage") == "data_collected":
        if not state.get("raw_communications"):
            errors.append("No communications data collected")
        if not state.get("customer_profile"):
            errors.append("No customer profil data")

    return errors

def create_progress_indicator(current_step: int, total_steps: int, step_name: str) -> str:
    """Create a visual progress indicator"""

    progress = "█" * current_step + "░" * (total_steps - current_step)
    percentage = (current_step / total_steps) * 100
    return f"[{progress}] {percentage:5.1f}% - {step_name}"

def sanitize_customer_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove or mask sensitive information from customer data"""
    sanitized = data.copy()

    # Mask email addresses
    if "email" in sanitized:
        email = sanitized["email"]
        if "@" in email:
            local, domain = email.split("@", 1)
            sanitized["email"] = f"{local[:2]}***@{domain}"

    # Remove sensitive fields
    sensitive_fields = ["password", "api_key", "token", "ssn", "credit_card"]
    for field in sensitive_fields:
        if field in sanitized:
            sanitized[field] = "[REDACTED]"

    return sanitized

def generate_report_id() -> str:
    """Generate unique report ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"CHG_RPT_{timestamp}"

class ErrorHandler:
    """Centralized error handling"""

    def __init__(self, logger):
        self.logger = logger

    def handle_agent_error(self, agent_name: str, error: Exception, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors in agent processing"""
        error_msg = f"Error in {agent_name}: {str(error)}"
        self.logger.error(error_msg)

        # Add error to state
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error_msg)

        # Set processing state to indicate error
        state["processing_stage"] = f"{agent_name}_error"

        return state
    
    def handle_api_error(self, api_name: str, error: Exception) -> Dict[str, Any]:
        """Handle API errors gracefully"""
        error_msg = f"API Error ({error_msg}): {str(error)}"
        self.logger.warning(error_msg)

        return {
            "success": False,
            "error": error_msg,
            "fallback_used": True
        }

def performance_timer(func):
    """Decorator to measure function execution time"""

    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        if isinstance(result, dict):
            result["execution_time"] = execution_time

        return result
    
    return wrapper