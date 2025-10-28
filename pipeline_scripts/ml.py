import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, recall_score, 
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 70)
print(" ML MODEL TRAINING - DELAY PREDICTION")
print("=" * 70)

print("\n Loading data...")
df = pd.read_csv('./data/master_logistics_data.csv')
print(f" Loaded {len(df)} orders")

df_train = df[df['Carrier'].notna()].copy()
print(f"📦 Training on {len(df_train)} delivered orders")
print(f"🎯 Target distribution: {df_train['is_delayed'].value_counts().to_dict()}")

print("\n Preparing features...")

label_encoders = {}
categorical_cols = ['Customer_Segment', 'Priority', 'Product_Category', 
                    'Origin', 'Destination', 'Carrier', 'Weather_Impact', 'Special_Handling']

for col in categorical_cols:
    le = LabelEncoder()
    df_train[f'{col}_encoded'] = le.fit_transform(df_train[col].astype(str))
    label_encoders[col] = le

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

X = df_train[feature_columns]
y = df_train['is_delayed']

print(f" Features prepared: {X.shape[1]} features")

print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f" Training set: {len(X_train)} samples")
print(f" Test set: {len(X_test)} samples")
print(f"   Train delay rate: {y_train.mean():.1%}")
print(f"   Test delay rate: {y_test.mean():.1%}")

print("\n Training baseline Random Forest...")

baseline_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

baseline_rf.fit(X_train, y_train)
y_pred_baseline = baseline_rf.predict(X_test)
y_pred_proba_baseline = baseline_rf.predict_proba(X_test)[:, 1]

print(" Baseline Model Performance:")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_baseline):.3f}")
print(f"   F1 Score: {f1_score(y_test, y_pred_baseline):.3f}")
print(f"   Recall: {recall_score(y_test, y_pred_baseline):.3f}")
print(f"   ROC-AUC: {roc_auc_score(y_test, y_pred_proba_baseline):.3f}")

print("\n🔧 Hyperparameter tuning...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

print(f" Best parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")

print("\n🚀 Training final model...")

final_model = grid_search.best_estimator_
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 70)
print(" FINAL MODEL PERFORMANCE")
print("=" * 70)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n🎯 Classification Metrics:")
print(f"   Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"   F1 Score:  {f1:.3f}")
print(f"   Recall:    {recall:.3f} ({recall*100:.1f}% of delays caught)")
print(f"   ROC-AUC:   {roc_auc:.3f}")

print(f"\n Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['On-Time', 'Delayed']))

cv_scores = cross_val_score(final_model, X_train, y_train, cv=5, scoring='f1')
print(f"\n Cross-Validation F1 Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

print("\n🔍 Feature Importance Analysis...")

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🏆 Top 15 Most Important Features:")
for idx, row in feature_importance.head(15).iterrows():
    print(f"   {row['feature']:.<40} {row['importance']:.4f}")

print("\n Creating visualizations...")

fig = plt.figure(figsize=(18, 12))

ax1 = plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['On-Time', 'Delayed'],
            yticklabels=['On-Time', 'Delayed'])
ax1.set_title('Confusion Matrix', fontsize=14, weight='bold', pad=15)
ax1.set_ylabel('Actual', fontsize=12, weight='bold')
ax1.set_xlabel('Predicted', fontsize=12, weight='bold')

for i in range(2):
    for j in range(2):
        percentage = cm[i, j] / cm[i].sum() * 100
        ax1.text(j+0.5, i+0.7, f'({percentage:.1f}%)', 
                ha='center', va='center', fontsize=10, color='gray')

ax2 = plt.subplot(2, 3, 2)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
ax2.plot(fpr, tpr, color='#e74c3c', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
ax2.set_xlabel('False Positive Rate', fontsize=12, weight='bold')
ax2.set_ylabel('True Positive Rate', fontsize=12, weight='bold')
ax2.set_title('ROC Curve', fontsize=14, weight='bold', pad=15)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(alpha=0.3)

ax3 = plt.subplot(2, 3, 3)
top_features = feature_importance.head(12)
bars = ax3.barh(range(len(top_features)), top_features['importance'].values,
                color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features))))
ax3.set_yticks(range(len(top_features)))
ax3.set_yticklabels(top_features['feature'].values, fontsize=9)
ax3.set_xlabel('Importance', fontsize=12, weight='bold')
ax3.set_title('Top 12 Feature Importance', fontsize=14, weight='bold', pad=15)
ax3.invert_yaxis()

ax4 = plt.subplot(2, 3, 4)
ax4.hist([y_pred_proba[y_test==0], y_pred_proba[y_test==1]], 
         bins=30, label=['Actual On-Time', 'Actual Delayed'],
         color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
ax4.set_xlabel('Predicted Probability of Delay', fontsize=12, weight='bold')
ax4.set_ylabel('Frequency', fontsize=12, weight='bold')
ax4.set_title('Prediction Distribution', fontsize=14, weight='bold', pad=15)
ax4.legend()
ax4.axvline(0.5, color='black', linestyle='--', linewidth=1, label='Threshold')

ax5 = plt.subplot(2, 3, 5)
metrics = ['Accuracy', 'F1 Score', 'Recall', 'ROC-AUC']
values = [accuracy, f1, recall, roc_auc]
bars = ax5.bar(metrics, values, color=['#3498db', '#9b59b6', '#f39c12', '#2ecc71'],
               alpha=0.8, edgecolor='black')
ax5.set_ylim(0, 1.1)
ax5.set_ylabel('Score', fontsize=12, weight='bold')
ax5.set_title('Model Performance Metrics', fontsize=14, weight='bold', pad=15)
for bar, val in zip(bars, values):
    ax5.text(bar.get_x() + bar.get_width()/2, val + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, weight='bold')

ax6 = plt.subplot(2, 3, 6)
categorical_importance = feature_importance[feature_importance['feature'].str.contains('_encoded')]['importance'].sum()
numerical_importance = feature_importance[~feature_importance['feature'].str.contains('_encoded')]['importance'].sum()
categories = ['Categorical\nFeatures', 'Numerical\nFeatures']
cat_values = [categorical_importance, numerical_importance]
colors = ['#e67e22', '#16a085']
bars = ax6.bar(categories, cat_values, color=colors, alpha=0.8, edgecolor='black')
ax6.set_ylabel('Total Importance', fontsize=12, weight='bold')
ax6.set_title('Feature Type Contribution', fontsize=14, weight='bold', pad=15)
for bar, val in zip(bars, cat_values):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig('visualizations/ml_model_evaluation.png', dpi=300, bbox_inches='tight')
print("   Saved: visualizations/ml_model_evaluation.png")
plt.close()

print("\n Saving model and artifacts...")

joblib.dump(final_model, 'models/delay_prediction_model.pkl')
print("   Saved: models/delay_prediction_model.pkl")

joblib.dump(label_encoders, 'models/label_encoders.pkl')
print("   Saved: models/label_encoders.pkl")

with open('models/feature_columns.txt', 'w') as f:
    f.write('\n'.join(feature_columns))
print("   Saved: models/feature_columns.txt")

feature_importance.to_csv('models/feature_importance.csv', index=False)
print("   Saved: models/feature_importance.csv")

with open('models/model_performance_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("DELAY PREDICTION MODEL - PERFORMANCE REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Model Type: Random Forest Classifier\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Test Samples: {len(X_test)}\n")
    f.write(f"Features: {len(feature_columns)}\n\n")
    f.write(f"Best Parameters:\n")
    for param, value in grid_search.best_params_.items():
        f.write(f"  {param}: {value}\n")
    f.write(f"\nPerformance Metrics:\n")
    f.write(f"  Accuracy:  {accuracy:.3f}\n")
    f.write(f"  F1 Score:  {f1:.3f}\n")
    f.write(f"  Recall:    {recall:.3f}\n")
    f.write(f"  ROC-AUC:   {roc_auc:.3f}\n\n")
    f.write(f"Cross-Validation F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})\n\n")
    f.write("Top 10 Features:\n")
    for idx, row in feature_importance.head(10).iterrows():
        f.write(f"  {row['feature']}: {row['importance']:.4f}\n")

print("   Saved: models/model_performance_report.txt")

print("\n" + "=" * 70)
print("💼 BUSINESS IMPACT ANALYSIS")
print("=" * 70)

avg_delay_cost = 500
total_delays_prevented = int(recall * y_test.sum())
monthly_savings = total_delays_prevented * avg_delay_cost * 3

print(f"\n Financial Impact (Monthly Projection):")
print(f"   Delays Caught: {total_delays_prevented} / {y_test.sum()}")
print(f"   Potential Savings: ₹{monthly_savings:,}")
print(f"   Annual Impact: ₹{monthly_savings * 12:,}")

print(f"\n Model Deployment Value:")
print(f" Can predict {recall*100:.1f}% of delays before they occur")
