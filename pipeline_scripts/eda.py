import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs('visualizations', exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("=" * 70)
print(" LOGISTICS DELAY ANALYSIS - EDA")
print("=" * 70)

df = pd.read_csv('./data/master_logistics_data.csv', parse_dates=['Order_Date'])

print(f"\n Loaded {len(df)} orders")
print(f" Delay Rate: {df['is_delayed'].mean():.1%}")
print(f" Total Revenue: ₹{df['Order_Value_INR'].sum():,.0f}")

# CHART 1: DELAY OVERVIEW
print("\n Creating Chart 1: Delay Overview...")

fig = plt.figure(figsize=(16, 6))
gs = GridSpec(1, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
delay_counts = df['Delivery_Status'].value_counts()
colors = ['#2ecc71', '#f39c12', '#e74c3c']
ax1.pie(delay_counts.values, labels=delay_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
ax1.set_title('Overall Delivery Performance', fontsize=14, weight='bold', pad=20)

ax2 = fig.add_subplot(gs[0, 1])
priority_delays = df.groupby('Priority')['is_delayed'].agg(['sum', 'count', 'mean']).reset_index()
priority_delays = priority_delays.sort_values('mean', ascending=False)
bars = ax2.bar(priority_delays['Priority'], priority_delays['mean'] * 100, 
               color=['#e74c3c', '#f39c12', '#3498db'])
ax2.set_ylabel('Delay Rate (%)', fontsize=11, weight='bold')
ax2.set_title('Delay Rate by Priority Level', fontsize=14, weight='bold', pad=20)
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10, weight='bold')

ax3 = fig.add_subplot(gs[0, 2])
delayed_orders = df[df['is_delayed'] == 1]['delay_days']
ax3.hist(delayed_orders, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Delay Days', fontsize=11, weight='bold')
ax3.set_ylabel('Frequency', fontsize=11, weight='bold')
ax3.set_title('Distribution of Delay Duration', fontsize=14, weight='bold', pad=20)
ax3.axvline(delayed_orders.mean(), color='navy', linestyle='--', linewidth=2, 
            label=f'Avg: {delayed_orders.mean():.1f} days')
ax3.legend()

plt.tight_layout()
plt.savefig('visualizations/chart1_delay_overview.png', dpi=300, bbox_inches='tight')
print("   Saved: visualizations/chart1_delay_overview.png")
plt.close()

# CHART 2: CARRIER PERFORMANCE
print(" Creating Chart 2: Carrier Performance...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

carrier_perf = df.groupby('Carrier').agg({
    'is_delayed': ['sum', 'count', 'mean'],
    'Customer_Rating': 'mean',
    'Delivery_Cost_INR': 'mean'
}).reset_index()
carrier_perf.columns = ['carrier', 'delayed_count', 'total_orders', 'delay_rate', 'avg_rating', 'avg_cost']
carrier_perf = carrier_perf.sort_values('delay_rate', ascending=True)

bars = axes[0, 0].barh(carrier_perf['carrier'], carrier_perf['delay_rate'] * 100, 
                        color=plt.cm.RdYlGn_r(carrier_perf['delay_rate']))
axes[0, 0].set_xlabel('Delay Rate (%)', fontsize=11, weight='bold')
axes[0, 0].set_title('Carrier Delay Rate Comparison', fontsize=13, weight='bold', pad=15)
for i, (bar, val) in enumerate(zip(bars, carrier_perf['delay_rate'] * 100)):
    axes[0, 0].text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                    va='center', fontsize=10, weight='bold')

axes[0, 1].scatter(carrier_perf['delay_rate'] * 100, carrier_perf['avg_rating'], 
                   s=carrier_perf['total_orders'] * 5, alpha=0.6, c=carrier_perf['delay_rate'],
                   cmap='RdYlGn_r', edgecolors='black', linewidth=1.5)
for idx, row in carrier_perf.iterrows():
    axes[0, 1].annotate(row['carrier'], (row['delay_rate'] * 100, row['avg_rating']),
                       fontsize=9, weight='bold')
axes[0, 1].set_xlabel('Delay Rate (%)', fontsize=11, weight='bold')
axes[0, 1].set_ylabel('Average Rating', fontsize=11, weight='bold')
axes[0, 1].set_title('Carrier Rating vs Delay Rate', fontsize=13, weight='bold', pad=15)
axes[0, 1].grid(alpha=0.3)

axes[1, 0].bar(carrier_perf['carrier'], carrier_perf['total_orders'], 
               color='#3498db', alpha=0.8, edgecolor='black')
axes[1, 0].set_ylabel('Order Volume', fontsize=11, weight='bold')
axes[1, 0].set_title('Order Volume by Carrier', fontsize=13, weight='bold', pad=15)
axes[1, 0].tick_params(axis='x', rotation=45)

axes[1, 1].bar(carrier_perf['carrier'], carrier_perf['avg_cost'], 
               color='#f39c12', alpha=0.8, edgecolor='black')
axes[1, 1].set_ylabel('Average Cost (₹)', fontsize=11, weight='bold')
axes[1, 1].set_title('Average Delivery Cost by Carrier', fontsize=13, weight='bold', pad=15)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('visualizations/chart2_carrier_performance.png', dpi=300, bbox_inches='tight')
print("   Saved: visualizations/chart2_carrier_performance.png")
plt.close()

# CHART 3: ROUTE & RISK ANALYSIS
print(" Creating Chart 3: Route Risk Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

weather_delays = df.groupby('Weather_Impact')['is_delayed'].mean().sort_values(ascending=False)
bars = axes[0, 0].bar(weather_delays.index, weather_delays.values * 100,
                      color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'], alpha=0.8, edgecolor='black')
axes[0, 0].set_ylabel('Delay Rate (%)', fontsize=11, weight='bold')
axes[0, 0].set_title('Impact of Weather on Delays', fontsize=13, weight='bold', pad=15)
axes[0, 0].tick_params(axis='x', rotation=45)
for bar in bars:
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, weight='bold')

axes[0, 1].scatter(df['route_risk_index'], df['is_delayed'], alpha=0.3, s=50, c='#3498db')
axes[0, 1].set_xlabel('Route Risk Index', fontsize=11, weight='bold')
axes[0, 1].set_ylabel('Delayed (1) or On-Time (0)', fontsize=11, weight='bold')
axes[0, 1].set_title('Route Risk Index vs Delivery Outcome', fontsize=13, weight='bold', pad=15)
z = np.polyfit(df['route_risk_index'].dropna(), df['is_delayed'].dropna(), 1)
p = np.poly1d(z)
axes[0, 1].plot(df['route_risk_index'].sort_values(), 
                p(df['route_risk_index'].sort_values()), 
                "r-", linewidth=2, label='Trend')
axes[0, 1].legend()

distance_bins = pd.cut(df['Distance_KM'], bins=5)
distance_delays = df.groupby(distance_bins)['is_delayed'].mean()
axes[1, 0].bar(range(len(distance_delays)), distance_delays.values * 100,
               color='#9b59b6', alpha=0.8, edgecolor='black')
axes[1, 0].set_xticks(range(len(distance_delays)))
axes[1, 0].set_xticklabels([f'{int(interval.left)}-{int(interval.right)}km' 
                            for interval in distance_delays.index], rotation=45)
axes[1, 0].set_ylabel('Delay Rate (%)', fontsize=11, weight='bold')
axes[1, 0].set_title('Delay Rate by Distance Range', fontsize=13, weight='bold', pad=15)

special_delays = df.groupby(df['Special_Handling'] != 'None')['is_delayed'].mean()
bars = axes[1, 1].bar(['Normal', 'Special Handling'], special_delays.values * 100,
                      color=['#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black')
axes[1, 1].set_ylabel('Delay Rate (%)', fontsize=11, weight='bold')
axes[1, 1].set_title('Impact of Special Handling on Delays', fontsize=13, weight='bold', pad=15)
for bar in bars:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig('visualizations/chart3_route_risk.png', dpi=300, bbox_inches='tight')
print("   Saved: visualizations/chart3_route_risk.png")
plt.close()

# CHART 4: COST ANALYSIS
print(" Creating Chart 4: Cost Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

cost_cols = ['Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance', 'Insurance',
             'Packaging_Cost', 'Technology_Platform_Fee', 'Other_Overhead']
avg_costs = df[cost_cols].mean()
colors_cost = plt.cm.Set3(range(len(cost_cols)))
axes[0, 0].pie(avg_costs.values, labels=[c.replace('_', ' ').title() for c in cost_cols],
               autopct='%1.1f%%', colors=colors_cost, startangle=90)
axes[0, 0].set_title('Average Cost Breakdown', fontsize=13, weight='bold', pad=15)

cost_comparison = df.groupby('Delivery_Status')[cost_cols].mean()
cost_comparison.T.plot(kind='bar', ax=axes[0, 1], alpha=0.8, edgecolor='black')
axes[0, 1].set_ylabel('Average Cost (₹)', fontsize=11, weight='bold')
axes[0, 1].set_title('Cost by Delivery Status', fontsize=13, weight='bold', pad=15)
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].legend(title='Status')

axes[1, 0].hist([df[df['is_delayed']==0]['total_cost'], 
                 df[df['is_delayed']==1]['total_cost']],
                bins=30, label=['On-Time', 'Delayed'], color=['#2ecc71', '#e74c3c'],
                alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Total Delivery Cost (₹)', fontsize=11, weight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=11, weight='bold')
axes[1, 0].set_title('Total Cost Distribution', fontsize=13, weight='bold', pad=15)
axes[1, 0].legend()

axes[1, 1].boxplot([df[df['is_delayed']==0]['cost_per_km'],
                     df[df['is_delayed']==1]['cost_per_km']],
                    labels=['On-Time', 'Delayed'], patch_artist=True,
                    boxprops=dict(facecolor='#3498db', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
axes[1, 1].set_ylabel('Cost per KM (₹)', fontsize=11, weight='bold')
axes[1, 1].set_title('Cost Efficiency by Status', fontsize=13, weight='bold', pad=15)

plt.tight_layout()
plt.savefig('visualizations/chart4_cost_analysis.png', dpi=300, bbox_inches='tight')
print("   Saved: visualizations/chart4_cost_analysis.png")
plt.close()

# CHART 5: FEATURE CORRELATION
print(" Creating Chart 5: Feature Correlation...")

fig, ax = plt.subplots(figsize=(14, 10))

corr_features = ['is_delayed', 'Order_Value_INR', 'Distance_KM', 'route_risk_index',
                 'carrier_reliability_score', 'order_complexity', 'priority_score',
                 'warehouse_efficiency', 'cost_per_km', 'total_cost', 'delay_days']

corr_matrix = df[corr_features].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax)

ax.set_title('Feature Correlation Matrix', fontsize=16, weight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('visualizations/chart5_correlation.png', dpi=300, bbox_inches='tight')
print("   Saved: visualizations/chart5_correlation.png")
plt.close()