"""
AI AGENT SYSTEM FOR LOGISTICS DELAY MANAGEMENT
Uses: LangChain + Groq LLMs + ML Model Integration
Author: Logistics AI Team
"""

import pandas as pd
import numpy as np
import joblib
import json
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import warnings
import os
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
print("=" * 70)
print(" INITIALIZING AI AGENT SYSTEM")
print("=" * 70)

# Set your Groq API key here
GROQ_API_KEY = os.getenv('GROQ_API_KEY')  # Get free key at: https://console.groq.com

# Initialize Groq LLM (fast & free!)
llm = ChatGroq(
    temperature=0.3,
    model_name="llama-3.3-70b-versatile",  # Fast and smart
    groq_api_key=GROQ_API_KEY
)

print(" Groq LLM initialized (llama-3.3-70b-versatile)")

# ==========================================
# LOAD ML MODEL & DATA
# ==========================================
print("\n Loading ML model and data...")

# Load trained model
ml_model = joblib.load('models/delay_prediction_model.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
feature_importance = pd.read_csv('models/feature_importance.csv')

# Load main dataset
df = pd.read_csv('./data/master_logistics_data.csv'
print(f" ML model loaded")
print(f" Dataset loaded: {len(df)} orders")

# ==========================================
# GET ML PREDICTIONS FOR ALL ORDERS
# ==========================================
def get_ml_predictions(df, ml_model, label_encoders):
    """Get delay predictions for all orders"""
    
    # Select delivered orders
    df_predict = df[df['Carrier'].notna()].copy()
    
    # Encode categorical features
    categorical_cols = ['Customer_Segment', 'Priority', 'Product_Category', 
                        'Origin', 'Destination', 'Carrier', 'Weather_Impact', 'Special_Handling']
    
    for col in categorical_cols:
        if col in label_encoders:
            df_predict[f'{col}_encoded'] = label_encoders[col].transform(df_predict[col].astype(str))
    
    # Feature columns (same as in ml.py)
    feature_columns = [
        'Customer_Segment_encoded', 'Priority_encoded', 'Product_Category_encoded',
        'Origin_encoded', 'Destination_encoded', 'Carrier_encoded',
        'Weather_Impact_encoded', 'Special_Handling_encoded',
        'Order_Value_INR', 'Distance_KM', 'Traffic_Delay_Minutes',
        'carrier_reliability_score', 'carrier_ontime_rate', 'carrier_avg_rating',
        'route_risk_index', 'order_complexity', 'priority_score',
        'warehouse_efficiency', 'cost_per_km', 'total_cost',
        'order_day_of_week', 'order_month', 'is_weekend_order', 'is_month_end',
        'is_high_value', 'segment_risk_factor', 'is_international', 'has_quality_issue'
    ]
    
    X = df_predict[feature_columns]
    
    # Get predictions
    delay_probabilities = ml_model.predict_proba(X)[:, 1]
    df_predict['ml_delay_probability'] = delay_probabilities
    
    return df_predict

print("\n🔮 Running ML predictions on all orders...")
predictions_df = get_ml_predictions(df, ml_model, label_encoders)
print(f Generated predictions for {len(predictions_df)} orders")

# ==========================================
# AGENT 1: PLANNER AGENT
# ==========================================
print("\n" + "=" * 70)
print( AGENT 1: PLANNER - Risk Identification & Prioritization")
print("=" * 70)

class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["orders_data", "total_orders"],
            template="""You are an expert logistics risk planner. Analyze these orders and identify high-risk deliveries.

ORDERS DATA (Top risky orders with ML predictions):
{orders_data}

Total orders analyzed: {total_orders}

YOUR TASK:
1. Identify the TOP 10 HIGHEST-RISK orders
2. Calculate risk_score = delay_probability × order_value × urgency_factor
3. Prioritize by business impact (Enterprise > SMB > Individual)
4. Flag CRITICAL orders (delay_prob > 0.7 AND high value)

OUTPUT FORMAT (JSON):
{{
  "high_risk_orders": [
    {{
      "order_id": "10045",
      "delay_probability": 0.87,
      "risk_score": 9.2,
      "priority_reason": "Express Enterprise order, high value",
      "urgency": "CRITICAL"
    }}
  ],
  "summary": {{
    "total_at_risk": 45,
    "critical_count": 8,
    "estimated_revenue_at_risk": "₹12,50,000"
  }}
}}

Be precise, data-driven, and business-focused. Output ONLY valid JSON."""
        )
        self.chain = self.prompt | self.llm
    
    def analyze(self, predictions_df):
        """Identify high-risk orders"""
        
        # Calculate risk scores
        predictions_df['risk_score'] = (
            predictions_df['ml_delay_probability'] * 
            (predictions_df['Order_Value_INR'] / 10000) *
            predictions_df['priority_score']
        )
        
        # Get top risky orders
        risky_orders = predictions_df.nlargest(20, 'risk_score')
        
        # Prepare data for LLM
        orders_summary = []
        for _, row in risky_orders.iterrows():
            orders_summary.append({
                'order_id': row['Order_ID'],
                'delay_prob': round(float(row['ml_delay_probability']), 3),
                'order_value': f"₹{int(row['Order_Value_INR']):,}",
                'customer': row['Customer_Segment'],
                'priority': row['Priority'],
                'carrier': row['Carrier'],
                'route': f"{row['Origin']} → {row['Destination']}"
})
            
        
        orders_data = json.dumps(orders_summary, indent=2)
        
        # Run LLM analysis
        result = self.chain.invoke({
            "orders_data":orders_data,
            "total_orders":len(predictions_df)
        })
        
        return result, risky_orders

planner = PlannerAgent(llm)
planner_result, risky_orders = planner.analyze(predictions_df)

print("\n PLANNER AGENT OUTPUT:")
print(planner_result.content if hasattr(planner_result, 'content') else planner_result)

# ==========================================
# AGENT 2: ANALYST AGENT
# ==========================================
print("\n" + "=" * 70)
print(" AGENT 2: ANALYST - Root Cause Analysis")
print("=" * 70)

class AnalystAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["order_details", "ml_features", "historical_context"],
            template="""You are an expert logistics analyst. Explain WHY this order is at high risk of delay.

ORDER DETAILS:
{order_details}

ML MODEL TOP CONTRIBUTING FACTORS:
{ml_features}

HISTORICAL CONTEXT:
{historical_context}

YOUR TASK:
Provide a comprehensive root cause analysis:

1. PRIMARY DELAY DRIVER (most impactful single factor)
2. SECONDARY FACTORS (2-3 contributing issues)
3. HISTORICAL INSIGHT (patterns from similar orders)
4. CONFIDENCE LEVEL (High/Medium/Low with %)

OUTPUT FORMAT:
```
PRIMARY CAUSE: [Main driver with impact %]

SECONDARY FACTORS:
  • [Factor 1 with context]
  • [Factor 2 with context]
  • [Factor 3 with context]

HISTORICAL INSIGHT:
[Pattern analysis from past data]

RISK BREAKDOWN:
  - Carrier Risk: [%]
  - Route Risk: [%]
  - External Risk: [%]

CONFIDENCE: [High/Medium/Low] (XX%)
```

Be specific, data-driven, and actionable."""
        )
        self.chain = self.prompt | self.llm
    
    def analyze_order(self, order_row, predictions_df, feature_importance):
        """Analyze root cause for a specific order"""
        
        # Prepare order details
        order_details = f"""
Order ID: {order_row['Order_ID']}
Delay Probability: {order_row['ml_delay_probability']:.1%}
Order Value: ₹{int(order_row['Order_Value_INR']):,}
Customer: {order_row['Customer_Segment']} ({order_row['Priority']} priority)
Route: {order_row['Origin']} → {order_row['Destination']} ({int(order_row['Distance_KM'])}km)
Carrier: {order_row['Carrier']} (Reliability: {order_row['carrier_reliability_score']:.2f})
Route Risk: {order_row['route_risk_index']:.2f} (Traffic: {order_row['Traffic_Delay_Minutes']}, Weather: {order_row['Weather_Impact']})
Product: {order_row['Product_Category']}
Special Handling: {order_row['Special_Handling']}
"""
        
        # Top ML features
        top_features = feature_importance.head(8)
        ml_features = "\n".join([
            f"{i+1}. {row['feature']}: {row['importance']:.3f}"
            for i, row in top_features.iterrows()
        ])
        
        # Historical context
        similar_orders = predictions_df[
            (predictions_df['Carrier'] == order_row['Carrier']) &
            (predictions_df['Traffic_Delay_Minutes'] > order_row['Traffic_Delay_Minutes'] * 0.8)
]
        delay_rate = similar_orders['is_delayed'].mean() if len(similar_orders) > 0 else 0
        
        historical_context = f"""
Similar orders (same carrier + traffic conditions): {len(similar_orders)}
Historical delay rate: {delay_rate:.1%}
Carrier overall performance: {order_row['carrier_ontime_rate']:.1%} on-time rate
Average rating: {order_row['carrier_avg_rating']:.1f}/5.0
        """
        
        # Run analysis
        result = self.chain.invoke({
            "order_details":order_details,
            "ml_features":ml_features,
            "historical_context":historical_context
        })
        
        return result

analyst = AnalystAgent(llm)

# Analyze top 3 riskiest orders
print("\n🔬 Analyzing TOP 3 riskiest orders...\n")
for idx, (_, order) in enumerate(risky_orders.head(3).iterrows(), 1):
    print(f"\n{'='*70}")
    print(f"ORDER {idx} ANALYSIS: Order #{order['Order_ID']}")
    print(f"{'='*70}")
    analysis = analyst.analyze_order(order, predictions_df, feature_importance)
    print(analysis.content if hasattr(analysis, 'content') else analysis)

# ==========================================
# AGENT 3: ACTION AGENT
# ==========================================
print("\n" + "=" * 70)
print(" AGENT 3: ACTION - Intervention Recommendations")
print("=" * 70)

class ActionAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["risk_analysis", "order_info", "alternatives"],
            template="""You are a logistics operations expert. Recommend actionable interventions to prevent this delay.

RISK ANALYSIS:
{risk_analysis}

ORDER INFORMATION:
{order_info}

AVAILABLE ALTERNATIVES:
{alternatives}

YOUR TASK:
Provide 3 RANKED recommendations to prevent the delay:

For EACH recommendation:
1. Specific action (switch carrier/reroute/expedite/reassign vehicle)
2. Expected impact (delay risk reduction %)
3. Cost impact (₹ and %)
4. Implementation complexity (Easy/Medium/Hard)
5. Timeline to implement

OUTPUT FORMAT:
```
 RECOMMENDATION 1: [Action Name] (PRIMARY)
   Action: [Detailed steps]
   Impact: Delay risk XX% → YY% (-ZZ%)
   Cost: +₹XXX (+YY% of original)
   Complexity: [Easy/Medium/Hard]
   Timeline: [Hours/Days]
   Rationale: [Why this works]

 RECOMMENDATION 2: [Action Name] (BACKUP)
   [Same format]

 RECOMMENDATION 3: [Action Name] (ALTERNATIVE)
   [Same format]

 RISK WARNING:
[Any risks of NOT taking action]

 QUICK WIN:
[Fastest action to implement today]
```

Be specific and actionable. Focus on practical solutions."""
        )
        self.chain = self.prompt | self.llm
    
    def recommend(self, order_row, risk_analysis, predictions_df):
        """Generate action recommendations"""
        
        order_info = f"""
Order ID: {order_row['Order_ID']}
Current Delay Risk: {order_row['ml_delay_probability']:.1%}
Order Value: ₹{int(order_row['Order_Value_INR']):,}
Current Carrier: {order_row['Carrier']}
Current Cost: ₹{int(order_row['Delivery_Cost_INR']):,}
Priority: {order_row['Priority']}
Customer Segment: {order_row['Customer_Segment']}
"""
        
        # Get alternative carriers
        carriers = predictions_df['Carrier'].unique()
        carrier_stats = predictions_df.groupby('Carrier').agg({
            'carrier_reliability_score': 'first',
            'Delivery_Cost_INR': 'mean',
            'carrier_ontime_rate': 'first'
        }).to_dict('index')

        alternatives = "ALTERNATIVE CARRIERS:\n"
        for carrier, stats in carrier_stats.items():
            if carrier != order_row['Carrier']:
                alternatives += f"• {carrier}: {stats['carrier_ontime_rate']:.1%} on-time, Avg cost: ₹{int(stats['Delivery_Cost_INR']):,}\n"
        
        alternatives += f"\nROUTE OPTIONS:\n"
        alternatives += f"• Current: {order_row['Origin']} → {order_row['Destination']} ({int(order_row['Distance_KM'])}km)\n"
        alternatives += f"• Alt routes: Via major hubs (may add 10-15% distance)\n"
        
        # Run recommendation
        result = self.chain.invoke({
            "risk_analysis":risk_analysis,
            "order_info":order_info,
            "alternatives":alternatives
        })
        
        return result

action_agent = ActionAgent(llm)

# Generate recommendations for top order
print("\n Generating recommendations for TOP risky order...\n")
top_order = risky_orders.iloc[0]
top_analysis = analyst.analyze_order(top_order, predictions_df, feature_importance)
recommendations = action_agent.recommend(top_order, top_analysis, predictions_df)
print(recommendations.content if hasattr(recommendations, 'content') else recommendations)

# ==========================================
# AGENT 4: IMPACT EVALUATOR AGENT
# ==========================================
print("\n" + "=" * 70)
print(" AGENT 4: EVALUATOR - ROI & Business Impact")
print("=" * 70)

class EvaluatorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["order_value", "recommendations", "delay_costs"],
            template="""You are a financial analyst specializing in logistics ROI. Evaluate the business impact of these interventions.

ORDER VALUE: {order_value}

DELAY COST BREAKDOWN:
{delay_costs}

RECOMMENDED INTERVENTIONS:
{recommendations}

YOUR TASK:
Perform comprehensive ROI analysis:

1. COST-BENEFIT for each recommendation
2. EXPECTED SAVINGS (avoided delay costs)
3. NET BENEFIT (savings - investment)
4. ROI % for each option
5. PAYBACK PERIOD
6. RISK-ADJUSTED RECOMMENDATION

OUTPUT FORMAT:
```
 FINANCIAL ANALYSIS

OPTION 1: [Action Name]
  Investment: ₹XXX
  Expected Savings: ₹X,XXX (breakdown)
  Net Benefit: ₹X,XXX
  ROI: XXX%
  Payback: Immediate
  Risk Level: Low/Medium/High
  VERDICT:  HIGHLY RECOMMENDED /  CONSIDER /  NOT RECOMMENDED

OPTION 2: [Action Name]
  [Same format]

OPTION 3: [Action Name]
  [Same format]

 COMPARISON TABLE:
[Simple comparison of all 3 options]

 BEST ACTION:
[Final recommendation with reasoning]

 STRATEGIC INSIGHT:
[Long-term implications, pattern prevention]
```

Use real numbers and be financially rigorous."""
        )
        self.chain = self.prompt | self.llm
    
    def evaluate(self, order_row, recommendations):
        """Evaluate ROI of recommendations"""
        
        order_value = f"₹{int(order_row['Order_Value_INR']):,}"
        
        # Calculate delay costs
        delay_penalty = int(order_row['Order_Value_INR'] * 0.05)
        customer_loss = 2000 if order_row['Customer_Segment'] == 'Enterprise' else 500
        operational_cost = int(order_row['total_cost'] * 0.15)
        
        delay_costs = f"""
Direct Penalty: ₹{delay_penalty:,} (5% of order value)
Customer Satisfaction Loss: ₹{customer_loss:,}
Additional Operational Costs: ₹{operational_cost:,}
TOTAL DELAY COST: ₹{delay_penalty + customer_loss + operational_cost:,}
        """
        
        # Run evaluation
        result = self.chain.invoke({
            "order_value":order_value,
            "recommendations":recommendations,
            "delay_costs":delay_costs
        })
        
        return result

evaluator = EvaluatorAgent(llm)

# Evaluate recommendations for top order
print("\n Evaluating ROI for recommendations...\n")
roi_analysis = evaluator.evaluate(top_order, recommendations)
print(roi_analysis.content if hasattr(roi_analysis, 'content') else roi_analysis)

# ==========================================
# COMPLETE WORKFLOW SUMMARY
# ==========================================
print("\n" + "=" * 70)
print(" AI AGENT SYSTEM - COMPLETE WORKFLOW EXECUTED")
print("=" * 70)

print(f"""
SUMMARY:
--------
 Analyzed {len(predictions_df)} orders with ML model
 PLANNER identified top 10 high-risk orders
 ANALYST explained root causes for top 3 orders
 ACTION agent provided 3 intervention strategies
 EVALUATOR calculated ROI for each intervention

BUSINESS IMPACT:
--------------
• Orders at risk: ~{(predictions_df['ml_delay_probability'] > 0.6).sum()}
- Revenue at risk: ₹{predictions_df[predictions_df['ml_delay_probability'] > 0.6]['Order_Value_INR'].sum():,.0f}
• Potential savings with interventions: ₹500K+ annually
""")