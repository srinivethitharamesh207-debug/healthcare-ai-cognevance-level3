# HealthAI - Predictive Analytics and AI Model Deployment

**Level 3 Advanced Project | Cognevance Internship**

A complete end-to-end machine learning pipeline for Healthcare Disease Prediction (Diabetes). The project covers data generation, advanced preprocessing, feature engineering, training and comparison of 5 ML models, FastAPI deployment, and an interactive prediction dashboard.

---

## Project Overview

This project demonstrates a production-grade predictive analytics system built entirely in Python. A synthetic dataset of 1,500 patient records is used to train multiple machine learning classifiers. The best-performing model is deployed via a FastAPI REST API with full documentation. An interactive HTML dashboard provides real-time predictions and visual performance reports.

---

## Project Structure

```
healthcare_ai/
│
├── run_pipeline.py                 Master script to run full pipeline
├── requirements.txt                Python dependencies
├── README.md                       Project documentation
│
├── data/
│   ├── healthcare_dataset.csv      Raw synthetic dataset (1500 records, 16 features)
│   ├── X_train.npy                 Preprocessed training features
│   ├── X_test.npy                  Preprocessed test features
│   ├── y_train.npy                 Training labels
│   └── y_test.npy                  Test labels
│
├── scripts/
│   ├── generate_dataset.py         Synthetic dataset generation
│   ├── preprocessing.py            NLP preprocessing and feature engineering pipeline
│   └── train_models.py             Multi-model training, evaluation, and comparison
│
├── models/
│   ├── logistic_regression.pkl     Trained Logistic Regression model
│   ├── random_forest.pkl           Trained Random Forest model (best)
│   ├── svm_model.pkl               Trained SVM model
│   ├── xgboost_model.pkl           Trained Gradient Boosting / XGBoost model
│   ├── neural_network.keras        Trained Neural Network (TensorFlow)
│   ├── best_model.pkl              Best model wrapper
│   ├── scaler.pkl                  Fitted StandardScaler
│   └── feature_names.pkl           Feature name list
│
├── api/
│   └── main.py                     FastAPI deployment application
│
├── dashboard/
│   └── index.html                  Interactive prediction dashboard
│
└── reports/
    ├── model_comparison.csv        Model metrics comparison table
    ├── model_comparison_chart.png  Bar chart of model performance
    ├── roc_curves.png              ROC curves for all models
    ├── confusion_matrices.png      Confusion matrices for all models
    ├── correlation_heatmap.png     Feature correlation heatmap
    └── eda_plots.png               EDA distribution plots
```

---

## Technologies Used

| Category | Tool / Library |
|---|---|
| Language | Python 3.11 |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn, Chart.js |
| Machine Learning | scikit-learn |
| Deep Learning | TensorFlow / Keras |
| API Framework | FastAPI, Uvicorn |
| Data Validation | Pydantic |
| Frontend | HTML, CSS, JavaScript |

---

## Dataset

The synthetic healthcare dataset contains 1,500 patient records with 16 raw features and 7 engineered features (22 total after preprocessing).

**Raw Features:**

| Feature | Type | Description |
|---|---|---|
| age | Numeric | Patient age (20-80) |
| gender | Categorical | Male / Female |
| bmi | Numeric | Body Mass Index |
| glucose_level | Numeric | Fasting blood glucose (mg/dL) |
| blood_pressure | Numeric | Diastolic blood pressure (mmHg) |
| insulin | Numeric | Serum insulin (mu U/ml) |
| skin_thickness | Numeric | Triceps skin fold (mm) |
| cholesterol | Numeric | Total cholesterol (mg/dL) |
| hba1c | Numeric | HbA1c percentage |
| family_history | Binary | Family history of diabetes |
| smoking_status | Categorical | Never / Former / Current |
| physical_activity | Categorical | Low / Moderate / High |
| diet_quality | Categorical | Poor / Average / Good |
| stress_level | Numeric | Self-reported stress (1-10) |
| pregnancies | Numeric | Number of pregnancies |
| disease_positive | Binary | Target label (0 = No, 1 = Yes) |

**Engineered Features (7 additional):**

| Feature | Formula |
|---|---|
| bmi_glucose_ratio | BMI / (glucose + 1) |
| age_bmi_interaction | age * BMI |
| glucose_insulin_ratio | glucose / (insulin + 1) |
| risk_score | Weighted clinical risk sum |
| is_obese | BMI >= 30 |
| high_glucose | glucose >= 140 |
| diabetic_hba1c | HbA1c >= 6.5 |

---

## ML Pipeline

```
Raw Dataset (1500 records)
        |
        v
Data Preprocessing
  - Missing value imputation (median / mode)
  - Categorical encoding (label encoding)
  - Outlier capping (IQR method)
  - Feature engineering (7 new features)
  - StandardScaler normalization
        |
        v
Train / Test Split (80% / 20%, stratified)
        |
        v
Model Training (5 models)
  1. Logistic Regression
  2. Random Forest (200 estimators)
  3. SVM (RBF kernel)
  4. Gradient Boosting / XGBoost
  5. Neural Network (Dense 128 -> 64 -> 32 -> Sigmoid)
        |
        v
Model Evaluation
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC Score
  - 5-Fold Cross Validation
  - Confusion Matrix
        |
        v
Best Model Selection (Random Forest, ROC-AUC: 99.80%)
        |
        v
FastAPI Deployment (/predict endpoint)
        |
        v
Interactive Dashboard (dashboard/index.html)
```

---

## Model Performance Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 87.33% | 85.10% | 83.20% | 84.14% | 93.50% |
| **Random Forest** | **98.53%** | **98.20%** | **98.10%** | **98.15%** | **99.80%** |
| SVM | 89.67% | 88.40% | 87.60% | 88.00% | 95.10% |
| Gradient Boosting | 96.40% | 96.10% | 95.80% | 95.95% | 99.20% |
| Neural Network | 94.27% | 93.80% | 93.50% | 93.65% | 98.40% |

**Best Model: Random Forest (Accuracy: 98.53%, ROC-AUC: 99.80%)**

---

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/srinivethitharamesh207-debug/healthcare-ai-cognevance-level3.git
cd healthcare-ai-cognevance-level3
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Full Pipeline

```bash
python run_pipeline.py
```

This generates the dataset, runs preprocessing, trains all 5 models, and saves all reports and model files.

### Step 4: Launch the API

```bash
uvicorn api.main:app --reload --port 8000
```

Visit: `http://127.0.0.1:8000/docs` for interactive Swagger documentation.

### Step 5: Open the Dashboard

Open `dashboard/index.html` in any browser for the interactive prediction dashboard.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | API information |
| GET | /health | Health check |
| POST | /predict | Single patient prediction |
| POST | /predict/batch | Batch patient predictions |
| GET | /model/info | Model metadata and metrics |

**POST /predict - Request Example:**

```json
{
  "age": 52,
  "gender": "Female",
  "bmi": 31.5,
  "glucose_level": 148.0,
  "blood_pressure": 88.0,
  "insulin": 110.0,
  "skin_thickness": 26.0,
  "cholesterol": 210.0,
  "hba1c": 6.8,
  "family_history": 1,
  "smoking_status": "Never",
  "physical_activity": "Low",
  "diet_quality": "Average",
  "stress_level": 7,
  "pregnancies": 2
}
```

**Response:**

```json
{
  "prediction": 1,
  "prediction_label": "Disease Positive",
  "confidence": 91.40,
  "risk_level": "High Risk",
  "model_used": "Random Forest",
  "timestamp": "2026-06-07T14:32:05",
  "recommendations": [
    "Monitor blood glucose levels regularly.",
    "HbA1c is elevated. Consult a physician for diabetes screening.",
    "Increase physical activity to at least 150 minutes per week."
  ]
}
```

---

## Dashboard Features

The interactive HTML dashboard (`dashboard/index.html`) provides:

- Live patient prediction form with real-time results
- Risk level classification (High / Moderate / Low)
- Confidence score with animated progress bar
- Personalized health recommendations
- Radar chart showing patient risk profile
- Model accuracy comparison bar chart
- ROC-AUC score comparison chart
- Dataset distribution doughnut chart
- Full model performance metrics table

---

## Reports Generated

After running the pipeline, the following reports are saved in `reports/`:

| File | Description |
|---|---|
| model_comparison.csv | Tabular metrics for all 5 models |
| model_comparison_chart.png | Grouped bar chart of all metrics |
| roc_curves.png | ROC curves for all models |
| confusion_matrices.png | Confusion matrices side by side |
| correlation_heatmap.png | Feature correlation heatmap |
| eda_plots.png | Target distribution and feature density plots |

---

## Author

**Srinivethitha R**
B.E. - Artificial Intelligence and Data Science
Cognevance Internship - Level 3 Advanced Project

GitHub: [srinivethitharamesh207-debug](https://github.com/srinivethitharamesh207-debug)

---

## License

This project is submitted as part of the Cognevance Internship Program (Level 3 - Advanced).
