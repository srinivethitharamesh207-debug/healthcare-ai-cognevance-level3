"""
generate_dataset.py
-------------------
Generates a synthetic healthcare dataset for Disease Prediction (Diabetes).

Features:
    - 1500 patient records
    - 15 clinically relevant features
    - Realistic value distributions based on medical literature
    - Target: disease_positive (0 = No, 1 = Yes)

Output: data/healthcare_dataset.csv
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
N = 1500

age         = np.random.randint(20, 80, N)
gender      = np.random.choice(['Male', 'Female'], N)
bmi         = np.round(np.random.normal(27.5, 5.5, N).clip(15, 50), 1)
glucose     = np.round(np.random.normal(110, 30, N).clip(60, 300), 1)
blood_pressure   = np.round(np.random.normal(80, 12, N).clip(50, 130), 1)
insulin     = np.round(np.random.normal(120, 80, N).clip(0, 500), 1)
skin_thickness   = np.round(np.random.normal(23, 8, N).clip(5, 60), 1)
cholesterol = np.round(np.random.normal(200, 40, N).clip(100, 400), 1)
hba1c       = np.round(np.random.normal(5.8, 1.2, N).clip(4.0, 12.0), 1)
family_history   = np.random.choice([0, 1], N, p=[0.6, 0.4])
smoking_status   = np.random.choice(['Never', 'Former', 'Current'], N, p=[0.5, 0.3, 0.2])
physical_activity= np.random.choice(['Low', 'Moderate', 'High'], N, p=[0.3, 0.4, 0.3])
diet_quality     = np.random.choice(['Poor', 'Average', 'Good'], N, p=[0.25, 0.45, 0.30])
stress_level     = np.random.randint(1, 11, N)
pregnancies      = np.where(gender == 'Female', np.random.randint(0, 10, N), 0)

# Risk score for realistic label generation
risk = (
    (glucose > 140).astype(int) * 3 +
    (bmi > 30).astype(int) * 2 +
    (hba1c > 6.5).astype(int) * 3 +
    family_history * 2 +
    (age > 45).astype(int) * 1 +
    (blood_pressure > 90).astype(int) * 1 +
    (smoking_status == 'Current').astype(int) * 1 +
    (physical_activity == 'Low').astype(int) * 1 +
    (diet_quality == 'Poor').astype(int) * 1 +
    (stress_level > 7).astype(int) * 1
)

prob = 1 / (1 + np.exp(-(risk - 6) * 0.5))
disease_positive = (np.random.random(N) < prob).astype(int)

df = pd.DataFrame({
    'age'              : age,
    'gender'           : gender,
    'bmi'              : bmi,
    'glucose_level'    : glucose,
    'blood_pressure'   : blood_pressure,
    'insulin'          : insulin,
    'skin_thickness'   : skin_thickness,
    'cholesterol'      : cholesterol,
    'hba1c'            : hba1c,
    'family_history'   : family_history,
    'smoking_status'   : smoking_status,
    'physical_activity': physical_activity,
    'diet_quality'     : diet_quality,
    'stress_level'     : stress_level,
    'pregnancies'      : pregnancies,
    'disease_positive' : disease_positive
})

df.to_csv('data/healthcare_dataset.csv', index=False)
print(f"Dataset generated: {df.shape}")
print(f"Disease positive: {df['disease_positive'].sum()} ({df['disease_positive'].mean()*100:.1f}%)")
print(df.head())
