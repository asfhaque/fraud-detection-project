# Create comprehensive fraud detection project code
fraud_detection_code = '''
# Comprehensive Fraud Detection Project
# ====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Step 1: Data Loading and Exploration
def load_and_explore_data(file_path):
    """Load and explore the fraud detection dataset"""
    df = pd.read_csv(file_path)
    
    print("Dataset Shape:", df.shape)
    print("\\nClass Distribution:")
    print(df['Class'].value_counts())
    print("\\nDataset Info:")
    print(df.info())
    
    return df

# Step 2: Data Preprocessing
def preprocess_data(df):
    """Preprocess the fraud detection data"""
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler

# Step 3: Handle Class Imbalance
def apply_smote(X_train, y_train):
    """Apply SMOTE to handle class imbalance"""
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print("Before SMOTE:")
    print(f"Normal: {sum(y_train == 0)}, Fraud: {sum(y_train == 1)}")
    print("After SMOTE:")
    print(f"Normal: {sum(y_train_smote == 0)}, Fraud: {sum(y_train_smote == 1)}")
    
    return X_train_smote, y_train_smote

# Step 4: Model Training
def train_models(X_train, y_train, X_train_smote, y_train_smote):
    """Train multiple models with and without SMOTE"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    trained_models = {}
    
    # Train on original data
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    # Train on SMOTE data
    for name, model in models.items():
        model_smote = type(model)(random_state=42)
        model_smote.fit(X_train_smote, y_train_smote)
        trained_models[name + ' (SMOTE)'] = model_smote
    
    return trained_models

# Step 5: Model Evaluation
def evaluate_models(models, X_test, y_test):
    """Evaluate all trained models"""
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        }
    
    return pd.DataFrame(results).T

# Step 6: Hyperparameter Tuning
def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for Random Forest"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

# Step 7: Model Explainability (SHAP)
def explain_model_shap(model, X_test):
    """Explain model predictions using SHAP"""
    import shap
    
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    
    # Summary plot
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    
    return shap_values

# Step 8: Deployment Simulation
def deploy_model(model, scaler, new_transaction):
    """Simulate model deployment for new transactions"""
    # Preprocess new transaction
    new_transaction_scaled = scaler.transform([new_transaction])
    
    # Make prediction
    prediction = model.predict(new_transaction_scaled)[0]
    probability = model.predict_proba(new_transaction_scaled)[0][1]
    
    return prediction, probability

# Main execution
if __name__ == "__main__":
    # Load data (replace with actual file path)
    # df = load_and_explore_data('creditcard.csv')
    
    # For demonstration, create synthetic data
    np.random.seed(42)
    n_samples = 10000
    
    # Generate synthetic fraud dataset
    data = {
        'Amount': np.random.lognormal(2.5, 1.2, n_samples),
        'Time': np.random.uniform(0, 172800, n_samples),
        'V1': np.random.normal(0, 1, n_samples),
        'V2': np.random.normal(0, 1, n_samples),
        'V3': np.random.normal(0, 1, n_samples),
        'V4': np.random.normal(0, 1, n_samples),
        'V5': np.random.normal(0, 1, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    }
    
    df = pd.DataFrame(data)
    
    # Execute pipeline
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    models = train_models(X_train, y_train, X_train_smote, y_train_smote)
    results = evaluate_models(models, X_test, y_test)
    
    print("Model Performance:")
    print(results)
    
    # Hyperparameter tuning
    best_model, best_params = hyperparameter_tuning(X_train, y_train)
    print("\\nBest Parameters:", best_params)
    
    print("\\nFraud Detection Project Completed Successfully!")
'''

# Save the comprehensive code
with open('fraud_detection_project.py', 'w') as f:
    f.write(fraud_detection_code)

print("Comprehensive fraud detection project code saved to 'fraud_detection_project.py'")
print("File size:", len(fraud_detection_code), "characters")
print("Project ready for implementation!")