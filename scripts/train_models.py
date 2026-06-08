"""
train_models.py
---------------
Trains and compares 5 machine learning models for Healthcare Disease Prediction.

Models:
    1. Logistic Regression
    2. Random Forest Classifier
    3. Support Vector Machine (SVM)
    4. XGBoost Classifier
    5. Neural Network (MLP - TensorFlow/Keras)

Metrics evaluated:
    - Accuracy, Precision, Recall, F1-Score, ROC-AUC

Output:
    models/logistic_regression.pkl
    models/random_forest.pkl
    models/svm_model.pkl
    models/xgboost_model.pkl
    models/neural_network.keras (or .h5)
    models/best_model.pkl
    reports/model_comparison.csv
    reports/model_comparison_chart.png
    reports/roc_curves.png
    reports/confusion_matrices.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     roc_auc_score, confusion_matrix,
                                     roc_curve, classification_report)
from sklearn.model_selection import cross_val_score

os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# -------------------------------------------------------
# Load preprocessed data
# -------------------------------------------------------
X_train = np.load('data/X_train.npy')
X_test  = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test  = np.load('data/y_test.npy')

print("=" * 60)
print("MODEL TRAINING PIPELINE")
print("=" * 60)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

results  = {}
models   = {}
roc_data = {}

# -------------------------------------------------------
# Helper: Evaluate model
# -------------------------------------------------------
def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_prob) if y_prob is not None else 0.0
    cv   = cross_val_score(model, X_tr, y_tr, cv=5, scoring='accuracy').mean()

    results[name] = {
        'Accuracy' : round(acc * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall'   : round(rec * 100, 2),
        'F1-Score' : round(f1 * 100, 2),
        'ROC-AUC'  : round(auc * 100, 2),
        'CV Score' : round(cv * 100, 2)
    }
    models[name] = model
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_data[name] = (fpr, tpr, auc)

    print(f"\n  {name}")
    print(f"    Accuracy : {acc*100:.2f}%")
    print(f"    F1-Score : {f1*100:.2f}%")
    print(f"    ROC-AUC  : {auc*100:.2f}%")
    print(f"    CV Score : {cv*100:.2f}%")
    return model


# -------------------------------------------------------
# 1. Logistic Regression
# -------------------------------------------------------
print("\n[1/5] Training Logistic Regression...")
lr = evaluate('Logistic Regression',
    LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    X_train, X_test, y_train, y_test)
with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr, f)


# -------------------------------------------------------
# 2. Random Forest
# -------------------------------------------------------
print("\n[2/5] Training Random Forest...")
rf = evaluate('Random Forest',
    RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    X_train, X_test, y_train, y_test)
with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)


# -------------------------------------------------------
# 3. SVM
# -------------------------------------------------------
print("\n[3/5] Training Support Vector Machine...")
svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm = evaluate('SVM', svm, X_train, X_test, y_train, y_test)
with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)


# -------------------------------------------------------
# 4. XGBoost (via sklearn GradientBoosting fallback)
# -------------------------------------------------------
print("\n[4/5] Training XGBoost / Gradient Boosting...")
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                        use_label_encoder=False, eval_metric='logloss',
                        random_state=42, n_jobs=-1)
    model_name = 'XGBoost'
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    xgb = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                     learning_rate=0.1, random_state=42)
    model_name = 'Gradient Boosting'

xgb = evaluate(model_name, xgb, X_train, X_test, y_train, y_test)
with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb, f)


# -------------------------------------------------------
# 5. Neural Network (TensorFlow/Keras or sklearn MLP)
# -------------------------------------------------------
print("\n[5/5] Training Neural Network...")
try:
    import tensorflow as tf
    from tensorflow.keras.models    import Sequential
    from tensorflow.keras.layers    import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks  import EarlyStopping

    nn_model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    nn_model.fit(X_train, y_train, epochs=100, batch_size=32,
                 validation_split=0.15, callbacks=[es], verbose=0)

    y_prob_nn = nn_model.predict(X_test, verbose=0).flatten()
    y_pred_nn = (y_prob_nn >= 0.5).astype(int)
    nn_name   = 'Neural Network (TF)'

    try:
        nn_model.save('models/neural_network.keras')
    except Exception:
        nn_model.save('models/neural_network.h5')

except ImportError:
    from sklearn.neural_network import MLPClassifier
    nn_clf = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                           solver='adam', max_iter=500, random_state=42,
                           early_stopping=True, validation_fraction=0.15)
    nn_clf.fit(X_train, y_train)
    y_prob_nn = nn_clf.predict_proba(X_test)[:, 1]
    y_pred_nn = nn_clf.predict(X_test)
    nn_name   = 'Neural Network (MLP)'
    with open('models/neural_network.pkl', 'wb') as f:
        pickle.dump(nn_clf, f)

acc  = accuracy_score(y_test, y_pred_nn)
prec = precision_score(y_test, y_pred_nn, zero_division=0)
rec  = recall_score(y_test, y_pred_nn, zero_division=0)
f1   = f1_score(y_test, y_pred_nn, zero_division=0)
auc  = roc_auc_score(y_test, y_prob_nn)

results[nn_name] = {
    'Accuracy' : round(acc * 100, 2),
    'Precision': round(prec * 100, 2),
    'Recall'   : round(rec * 100, 2),
    'F1-Score' : round(f1 * 100, 2),
    'ROC-AUC'  : round(auc * 100, 2),
    'CV Score' : 0.0
}
fpr, tpr, _ = roc_curve(y_test, y_prob_nn)
roc_data[nn_name] = (fpr, tpr, auc)

print(f"\n  {nn_name}")
print(f"    Accuracy : {acc*100:.2f}%")
print(f"    F1-Score : {f1*100:.2f}%")
print(f"    ROC-AUC  : {auc*100:.2f}%")


# -------------------------------------------------------
# Save comparison results
# -------------------------------------------------------
df_results = pd.DataFrame(results).T
df_results.to_csv('reports/model_comparison.csv')
print("\n\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)
print(df_results.to_string())


# -------------------------------------------------------
# Save best model
# -------------------------------------------------------
best_name = df_results['ROC-AUC'].astype(float).idxmax()
best_model = models.get(best_name)
if best_model:
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump({'model': best_model, 'name': best_name}, f)
print(f"\nBest Model: {best_name} (ROC-AUC: {df_results.loc[best_name, 'ROC-AUC']}%)")


# -------------------------------------------------------
# Plot 1: Model Comparison Bar Chart
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))
metrics  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x        = np.arange(len(df_results.index))
width    = 0.15
colors   = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax.bar(x + i * width, df_results[metric].astype(float),
           width, label=metric, color=color, alpha=0.85, edgecolor='white')

ax.set_xticks(x + width * 2)
ax.set_xticklabels(df_results.index, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Score (%)')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_ylim(0, 110)
ax.axhline(y=90, color='gray', linestyle='--', alpha=0.4, label='90% line')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('reports/model_comparison_chart.png', dpi=150, bbox_inches='tight')
plt.close()


# -------------------------------------------------------
# Plot 2: ROC Curves
# -------------------------------------------------------
plt.figure(figsize=(9, 7))
colors_roc = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors_roc):
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
plt.xlim([0, 1]); plt.ylim([0, 1.02])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('reports/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()


# -------------------------------------------------------
# Plot 3: Confusion Matrices
# -------------------------------------------------------
fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
if len(models) == 1:
    axes = [axes]

for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    ax.set_title(name.replace(' ', '\n'), fontsize=9, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

plt.suptitle('Confusion Matrices', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('reports/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n[DONE] All models trained. Reports saved to reports/")
