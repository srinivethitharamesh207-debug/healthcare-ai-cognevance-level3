"""
preprocessing.py
----------------
Advanced Data Preprocessing and Feature Engineering Pipeline
for the Healthcare Disease Prediction project.

Steps:
    1.  Load raw dataset
    2.  Exploratory Data Analysis (EDA) summary
    3.  Handle missing values
    4.  Encode categorical variables
    5.  Feature engineering (new derived features)
    6.  Outlier detection and capping (IQR method)
    7.  Feature scaling (StandardScaler)
    8.  Feature selection (correlation + importance)
    9.  Train/test split
    10. Save processed artifacts

Output:
    data/X_train.npy, data/X_test.npy
    data/y_train.npy, data/y_test.npy
    models/scaler.pkl
    models/feature_names.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# -------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------
print("=" * 60)
print("PREPROCESSING PIPELINE")
print("=" * 60)

df = pd.read_csv('data/healthcare_dataset.csv')
print(f"\n[1] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# -------------------------------------------------------
# 2. EDA Summary
# -------------------------------------------------------
print("\n[2] EDA Summary:")
print(f"    Missing values : {df.isnull().sum().sum()}")
print(f"    Duplicates     : {df.duplicated().sum()}")
print(f"    Target balance : {df['disease_positive'].value_counts().to_dict()}")
print(f"    Numeric cols   : {df.select_dtypes(include=np.number).columns.tolist()}")
print(f"    Categorical    : {df.select_dtypes(include='object').columns.tolist()}")

# -------------------------------------------------------
# 3. Handle Missing Values
# -------------------------------------------------------
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
print(f"\n[3] Missing values after imputation: {df.isnull().sum().sum()}")

# -------------------------------------------------------
# 4. Encode Categorical Variables
# -------------------------------------------------------
le = LabelEncoder()
df['gender_enc']            = le.fit_transform(df['gender'])

smoking_map  = {'Never': 0, 'Former': 1, 'Current': 2}
activity_map = {'Low': 0, 'Moderate': 1, 'High': 2}
diet_map     = {'Poor': 0, 'Average': 1, 'Good': 2}

df['smoking_enc']   = df['smoking_status'].map(smoking_map)
df['activity_enc']  = df['physical_activity'].map(activity_map)
df['diet_enc']      = df['diet_quality'].map(diet_map)

print("\n[4] Categorical encoding complete.")

# -------------------------------------------------------
# 5. Feature Engineering
# -------------------------------------------------------
df['bmi_glucose_ratio']     = df['bmi'] / (df['glucose_level'] + 1)
df['age_bmi_interaction']   = df['age'] * df['bmi']
df['glucose_insulin_ratio'] = df['glucose_level'] / (df['insulin'] + 1)
df['risk_score']            = (
    (df['glucose_level'] > 140).astype(int) * 3 +
    (df['bmi'] > 30).astype(int) * 2 +
    (df['hba1c'] > 6.5).astype(int) * 3 +
    df['family_history'] * 2 +
    (df['age'] > 45).astype(int)
)
df['is_obese']              = (df['bmi'] >= 30).astype(int)
df['high_glucose']          = (df['glucose_level'] >= 140).astype(int)
df['diabetic_hba1c']        = (df['hba1c'] >= 6.5).astype(int)

print("\n[5] Feature engineering complete. New features added:")
print("    bmi_glucose_ratio, age_bmi_interaction, glucose_insulin_ratio,")
print("    risk_score, is_obese, high_glucose, diabetic_hba1c")

# -------------------------------------------------------
# 6. Outlier Capping (IQR Method)
# -------------------------------------------------------
numeric_cols = ['bmi', 'glucose_level', 'insulin', 'cholesterol',
                'blood_pressure', 'skin_thickness']

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)

print("\n[6] Outlier capping applied (IQR method).")

# -------------------------------------------------------
# 7. Select Features
# -------------------------------------------------------
feature_cols = [
    'age', 'gender_enc', 'bmi', 'glucose_level', 'blood_pressure',
    'insulin', 'skin_thickness', 'cholesterol', 'hba1c',
    'family_history', 'smoking_enc', 'activity_enc', 'diet_enc',
    'stress_level', 'pregnancies',
    'bmi_glucose_ratio', 'age_bmi_interaction', 'glucose_insulin_ratio',
    'risk_score', 'is_obese', 'high_glucose', 'diabetic_hba1c'
]

X = df[feature_cols].values
y = df['disease_positive'].values

# -------------------------------------------------------
# 8. Train/Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[7] Train/Test split:")
print(f"    Train : {X_train.shape}")
print(f"    Test  : {X_test.shape}")

# -------------------------------------------------------
# 9. Feature Scaling
# -------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print("\n[8] StandardScaler applied.")

# -------------------------------------------------------
# 10. Save Artifacts
# -------------------------------------------------------
np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy',  X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy',  y_test)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print("\n[9] Artifacts saved:")
print("    data/X_train.npy, data/X_test.npy")
print("    data/y_train.npy, data/y_test.npy")
print("    models/scaler.pkl, models/feature_names.pkl")

# -------------------------------------------------------
# 11. Correlation Heatmap
# -------------------------------------------------------
plt.figure(figsize=(14, 10))
corr_cols = ['age', 'bmi', 'glucose_level', 'hba1c', 'insulin',
             'cholesterol', 'blood_pressure', 'risk_score', 'disease_positive']
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# -------------------------------------------------------
# 12. Target Distribution Plot
# -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df['disease_positive'].value_counts().plot(kind='bar', ax=axes[0],
    color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[0].set_title('Disease Distribution', fontweight='bold')
axes[0].set_xticklabels(['Negative', 'Positive'], rotation=0)
axes[0].set_ylabel('Count')

for feat, ax in zip(['glucose_level', 'bmi'], [axes[1]]):
    df.groupby('disease_positive')[feat].plot(kind='kde', ax=ax, legend=True)
    ax.set_title(f'{feat} Distribution by Disease Status', fontweight='bold')
    ax.legend(['Negative', 'Positive'])

plt.tight_layout()
plt.savefig('reports/eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n[10] EDA plots saved to reports/")
print("\n[DONE] Preprocessing pipeline complete.")
