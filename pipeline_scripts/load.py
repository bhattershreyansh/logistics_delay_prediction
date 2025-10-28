import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 60)
print("LOGISTICS AI - FEATURE ENGINEERING PIPELINE")
print("=" * 60)

# ==========================================
# LOAD ALL DATASETS
# ==========================================
print("\n Loading all datasets...")
orders_df = pd.read_csv('./data/orders.csv', parse_dates=['Order_Date'])
delivery_df = pd.read_csv('./data/delivery_performance.csv')
routes_df = pd.read_csv('./data/routes_distance.csv')
cost_breakdown_df = pd.read_csv('./data/cost_breakdown.csv')
feedback_df = pd.read_csv('./data/customer_feedback.csv', parse_dates=['Feedback_Date'])
vehicles_df = pd.read_csv('./data/vehicle_fleet.csv')
warehouse_df = pd.read_csv('./data/warehouse_inventory.csv', parse_dates=['Last_Restocked_Date'])

print(f" Orders: {len(orders_df)} | Delivery: {len(delivery_df)} | Routes: {len(routes_df)}")
print(f" Costs: {len(cost_breakdown_df)} | Feedback: {len(feedback_df)}")
print(f" Vehicles: {len(vehicles_df)} | Warehouses: {len(warehouse_df)}")

# ==========================================
# STEP 1: MERGE ORDER-LEVEL TABLES
# ==========================================
print("\n🔗 Merging order-level tables...")

# Start with orders, left join delivery info
main_df = orders_df.merge(delivery_df, on='Order_ID', how='left')
print(f"  After delivery merge: {len(main_df)} rows")

# Merge routes info
main_df = main_df.merge(routes_df, on='Order_ID', how='left')
print(f"  After routes merge: {len(main_df)} rows")

# Merge cost breakdown
main_df = main_df.merge(cost_breakdown_df, on='Order_ID', how='left')
print(f"  After costs merge: {len(main_df)} rows")

# Merge customer feedback (optional - not all orders have feedback)
main_df = main_df.merge(
    feedback_df[['Order_ID', 'Feedback_Date', 'Rating', 'Would_Recommend']], 
    on='Order_ID', 
    how='left'
)
print(f"  After feedback merge: {len(main_df)} rows")

# ==========================================
# STEP 2: CREATE TARGET VARIABLE
# ==========================================
print("\n Creating target variable...")

# Binary target: is_delayed (1 = Delayed, 0 = On-Time)
main_df['is_delayed'] = main_df['Delivery_Status'].isin(['Slightly-Delayed', 'Severely-Delayed']).astype(int)

# Calculate actual delay in days
main_df['delay_days'] = main_df['Actual_Delivery_Days'] - main_df['Promised_Delivery_Days']

print(f"  Delay Rate: {main_df['is_delayed'].mean():.2%}")
print(f"  Delayed Orders: {main_df['is_delayed'].sum()} / {len(main_df)}")

# ==========================================
# STEP 3: ADVANCED FEATURE ENGINEERING
# ==========================================
print("\n Engineering advanced features...")

# --- 1. CARRIER RELIABILITY SCORE ---
carrier_performance = delivery_df.groupby('Carrier').agg({
    'Delivery_Status': lambda x: (x == 'On-Time').mean(),
    'Customer_Rating': 'mean'
}).reset_index()
carrier_performance.columns = ['Carrier', 'carrier_ontime_rate', 'carrier_avg_rating']
carrier_performance['carrier_reliability_score'] = (
    carrier_performance['carrier_ontime_rate'] * 0.7 + 
    carrier_performance['carrier_avg_rating'] / 5 * 0.3
)

main_df = main_df.merge(carrier_performance, on='Carrier', how='left')
print(f"   Carrier reliability score added")

# --- 2. ROUTE RISK INDEX ---
# Combine traffic delay and weather impact into a risk score
main_df['traffic_risk_score'] = (main_df['Traffic_Delay_Minutes'] / main_df['Traffic_Delay_Minutes'].max()).fillna(0)

weather_weight = {'None': 0, 'Light_Rain': 0.3, 'Heavy_Rain': 0.7, 'Fog': 0.5}
main_df['weather_risk_score'] = main_df['Weather_Impact'].map(weather_weight).fillna(0)

main_df['route_risk_index'] = (main_df['traffic_risk_score'] * 0.6 + main_df['weather_risk_score'] * 0.4)
print(f"   Route risk index created")

# --- 3. ORDER COMPLEXITY SCORE ---
# Based on value, special handling, distance
special_handling_score = main_df['Special_Handling'].apply(lambda x: 0 if x == 'None' else 1)

main_df['order_complexity'] = (
    (main_df['Order_Value_INR'] / main_df['Order_Value_INR'].max()) * 0.3 +
    special_handling_score * 0.4 +
    (main_df['Distance_KM'] / main_df['Distance_KM'].max()) * 0.3
)
print(f"   Order complexity score calculated")

# --- 4. TOTAL DELIVERY COST ---
cost_columns = ['Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance', 'Insurance', 
                'Packaging_Cost', 'Technology_Platform_Fee', 'Other_Overhead']
main_df['total_cost'] = main_df[cost_columns].sum(axis=1)
print(f"   Total delivery cost computed")

# --- 5. COST PER KM ---
main_df['cost_per_km'] = main_df['total_cost'] / main_df['Distance_KM']
main_df['cost_per_km'] = main_df['cost_per_km'].replace([np.inf, -np.inf], np.nan)
print(f"   Cost efficiency metric added")

# --- 6. PRIORITY URGENCY SCORE ---
priority_scores = {'Express': 3, 'Standard': 2, 'Economy': 1}
main_df['priority_score'] = main_df['Priority'].map(priority_scores)
print(f"   Priority urgency encoded")

# --- 7. SEASONALITY FEATURES ---
main_df['order_day_of_week'] = main_df['Order_Date'].dt.dayofweek  # 0=Monday
main_df['order_month'] = main_df['Order_Date'].dt.month
main_df['is_weekend_order'] = main_df['order_day_of_week'].isin([5, 6]).astype(int)
main_df['is_month_end'] = (main_df['Order_Date'].dt.day >= 25).astype(int)
print(f"   Temporal features extracted")

# --- 8. WAREHOUSE EFFICIENCY ---
# Calculate stock availability for order's product category at origin warehouse
warehouse_df['stock_ratio'] = warehouse_df['Current_Stock_Units'] / warehouse_df['Reorder_Level']
warehouse_df['warehouse_efficiency'] = warehouse_df['stock_ratio'].clip(0, 2) / 2  # Normalize to 0-1

# Create warehouse lookup with location and product category
warehouse_lookup = warehouse_df.groupby(['Location', 'Product_Category'])['warehouse_efficiency'].mean().reset_index()

# Merge warehouse efficiency
main_df = main_df.merge(
    warehouse_lookup,
    left_on=['Origin', 'Product_Category'],
    right_on=['Location', 'Product_Category'],
    how='left'
)
main_df['warehouse_efficiency'].fillna(main_df['warehouse_efficiency'].median(), inplace=True)
print(f"   Warehouse efficiency matched")

# --- 9. HIGH VALUE FLAG ---
main_df['is_high_value'] = (main_df['Order_Value_INR'] > main_df['Order_Value_INR'].quantile(0.75)).astype(int)
print(f"   High-value order flag created")

# --- 10. CUSTOMER SEGMENT ENCODING ---
segment_risk = {'Enterprise': 0.1, 'SMB': 0.15, 'Individual': 0.25}
main_df['segment_risk_factor'] = main_df['Customer_Segment'].map(segment_risk)
print(f"   Customer segment risk added")

# --- 11. QUALITY ISSUE FLAG ---
main_df['has_quality_issue'] = (main_df['Quality_Issue'] != 'Perfect').astype(int)
print(f"   Quality issue flag created")

# --- 12. INTERNATIONAL ROUTE FLAG ---
international_destinations = ['Dubai', 'Singapore', 'Hong Kong', 'Bangkok']
main_df['is_international'] = main_df['Destination'].isin(international_destinations).astype(int)
print(f"   International route flag added")

# --- 13. FUEL EFFICIENCY RATIO ---
main_df['fuel_efficiency'] = main_df['Distance_KM'] / main_df['Fuel_Consumption_L']
main_df['fuel_efficiency'] = main_df['fuel_efficiency'].replace([np.inf, -np.inf], np.nan)
print(f"   Fuel efficiency calculated")

# --- 14. PROFITABILITY SCORE ---
main_df['profit_margin'] = (main_df['Delivery_Cost_INR'] - main_df['total_cost']) / main_df['Delivery_Cost_INR']
main_df['profit_margin'] = main_df['profit_margin'].replace([np.inf, -np.inf], np.nan)
print(f"   Profitability metrics added")

# ==========================================
# STEP 4: HANDLE MISSING VALUES
# ==========================================
print("\n🔧 Handling missing values...")

missing_before = main_df.isnull().sum().sum()

# Fill missing values with appropriate defaults
numeric_columns = main_df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if main_df[col].isnull().any():
        main_df[col].fillna(main_df[col].median(), inplace=True)

missing_after = main_df.isnull().sum().sum()
print(f"  Missing values: {missing_before} → {missing_after}")

# ==========================================
# STEP 5: SAVE MASTER DATASET
# ==========================================
print("\n Saving master_logistics_data.csv...")

main_df.to_csv('./data/master_logistics_data.csv', index=False)

print(f" Saved: master_logistics_data.csv ({len(main_df)} rows × {len(main_df.columns)} columns)")

# ==========================================
# STEP 6: DATA QUALITY REPORT
# ==========================================
print("\n" + "=" * 60)
print(" DATA QUALITY REPORT")
print("=" * 60)

print(f"\n Total Orders: {len(main_df)}")
print(f" Delivered Orders: {len(main_df)}")
print(f"⏱ Delayed Orders: {main_df['is_delayed'].sum()} ({main_df['is_delayed'].mean():.1%})")

print(f"\n Financial Summary:")
print(f"  Average Order Value: ₹{main_df['Order_Value_INR'].mean():,.2f}")
print(f"  Average Delivery Cost: ₹{main_df['Delivery_Cost_INR'].mean():,.2f}")
print(f"  Average Total Cost: ₹{main_df['total_cost'].mean():,.2f}")
print(f"  Total Revenue: ₹{main_df['Order_Value_INR'].sum():,.2f}")

print(f"\n🚦 Top Delay Drivers:")
delayed_df = main_df[main_df['is_delayed'] == 1]
if len(delayed_df) > 0:
    print(f"  High Route Risk: {(delayed_df['route_risk_index'] > 0.5).mean():.1%}")
    print(f"  Low Carrier Reliability: {(delayed_df['carrier_reliability_score'] < 0.7).mean():.1%}")
    print(f"  Special Handling: {(delayed_df['Special_Handling'] != 'None').mean():.1%}")
    print(f"  International Routes: {delayed_df['is_international'].mean():.1%}")

print(f"\n Feature Engineering Summary:")
print(f"   Carrier reliability scores")
print(f"   Route risk indices")
print(f"   Order complexity scores")
print(f"   Warehouse efficiency metrics")
print(f"   Temporal seasonality features")
print(f"   Cost efficiency metrics")
print(f"   Quality issue flags")
print(f"   Profitability indicators")

print(f"\n🎯 Target Variable Distribution:")
print(main_df['Delivery_Status'].value_counts())

print(f"\n📊 Carrier Performance:")
print(carrier_performance.sort_values('carrier_reliability_score', ascending=False).to_string(index=False))

print("\n" + "=" * 60)
print("🎉 FEATURE ENGINEERING COMPLETE!")
print("=" * 60)

# Show sample of engineered features
print("\n Sample of Engineered Features:")
feature_cols = ['Order_ID', 'is_delayed', 'delay_days', 'carrier_reliability_score', 
                'route_risk_index', 'order_complexity', 'priority_score',
                'warehouse_efficiency', 'cost_per_km', 'is_international']
print(main_df[feature_cols].head(10).to_string(index=False))

print("\n Output file: ./datamaster_logistics_data.csv")
print(f"Total features created: {len(main_df.columns)}")