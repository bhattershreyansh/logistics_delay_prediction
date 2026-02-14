# Logistics AI Dashboard

A **logistics delay prediction and risk management** system with a multi-page Streamlit dashboard, ML-based delay forecasting, and AI-powered recommendations. Built for analyzing delivery performance, identifying at-risk orders, and suggesting interventions.

---

## Features

- **Executive Overview** — KPIs, delivery status distribution, risk levels, performance by priority and customer segment, cost impact, and delay trends
- **Risk Monitor** — Filter and drill into at-risk orders by risk level, priority, segment, and carrier; view top risky orders and route/carrier risk analysis
- **Predictive Insights** — Model performance metrics, feature importance, delay patterns (weather, traffic, distance), correlations, and cost impact of delays
- **AI Recommendations** — Multi-agent flow: Risk Planner → Root Cause Analyst → Action Recommender → ROI Evaluator, powered by **LangChain** and **Groq** (Llama 3.3 70B)

---

## Project Structure

```
.
├── app.py                    # Main Streamlit dashboard (multi-page)
├── pipeline_scripts/
│   ├── load.py               # Data load + feature engineering → master_logistics_data.csv
│   ├── eda.py                # Exploratory data analysis + visualizations
│   ├── ml.py                 # ML model training (Random Forest delay prediction)
│   └── agentic.py            # AI agent system (LangChain + Groq)
├── data/
│   ├── orders.csv
│   ├── delivery_performance.csv
│   ├── routes_distance.csv
│   ├── cost_breakdown.csv
│   ├── customer_feedback.csv
│   ├── vehicle_fleet.csv
│   ├── warehouse_inventory.csv
│   └── master_logistics_data.csv   # Built by load.py
├── models/
│   ├── delay_prediction_model.pkl
│   ├── label_encoders.pkl
│   ├── feature_importance.csv
│   └── model_performance_report.txt
├── .streamlit/
│   └── secrets.toml.template
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd OF!_2
pip install -r requirements.txt
```

### 2. Data and models

- **Data**: Place your CSVs in `data/` (or use the existing ones). The pipeline expects:
  - `orders.csv`, `delivery_performance.csv`, `routes_distance.csv`, `cost_breakdown.csv`, `customer_feedback.csv`, `vehicle_fleet.csv`, `warehouse_inventory.csv`
- **Master dataset**: Run the load pipeline to build `data/master_logistics_data.csv`:
  ```bash
  python pipeline_scripts/load.py
  ```
- **Models**: Train the delay prediction model and encoders:
  ```bash
  python pipeline_scripts/ml.py
  ```
  This writes `models/delay_prediction_model.pkl`, `models/label_encoders.pkl`, and related artifacts.

### 3. (Optional) AI recommendations — Groq API

For the **AI Recommendations** page (root cause analysis and action suggestions):

1. Get a free API key from [Groq Console](https://console.groq.com/).
2. Either:
   - **Streamlit**: Copy `.streamlit/secrets.toml.template` to `.streamlit/secrets.toml` and set `GROQ_API_KEY = "your_key"`.
   - **Environment**: Set `GROQ_API_KEY` in your environment or in a `.env` file (e.g. with `python-dotenv`).

Without a key, the dashboard still runs; the AI Recommendation section will show a message and fallback rule-based insights.

---

## Run the dashboard

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Pipeline scripts (optional)

- **EDA**: Generate visualizations into `visualizations/`:
  ```bash
  python pipeline_scripts/eda.py
  ```
- **AI agents (standalone)**: Run the agentic pipeline (requires trained models and `GROQ_API_KEY`):
  ```bash
  python pipeline_scripts/agentic.py
  ```

---

## Tech stack

| Area        | Stack |
|------------|--------|
| UI         | Streamlit |
| Charts     | Plotly |
| ML         | scikit-learn (Random Forest), pandas, numpy |
| AI agents  | LangChain, LangChain-Groq, Groq (Llama 3.3 70B) |
| Data       | pandas, CSV |

---

## License

See repository license if applicable.
