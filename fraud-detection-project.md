# Comprehensive Fraud Detection Project

## Project Overview

This project demonstrates a complete end-to-end fraud detection system using machine learning techniques. The system is designed to detect fraudulent transactions in credit card data with high accuracy and reliability.

**Key Objectives:**
- Build a robust fraud detection model
- Handle class imbalance effectively
- Achieve high fraud detection rates (recall)
- Maintain low false positive rates
- Provide model explainability

## Dataset Information

**Dataset:** Credit Card Fraud Detection Dataset
- **Size:** 550,000+ transactions
- **Features:** 30 features (V1-V28 anonymized via PCA, Time, Amount)
- **Target:** Class (0=Normal, 1=Fraud)
- **Challenge:** Highly imbalanced dataset (~0.17% fraud)

## Required Libraries

```python
# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve

# Class imbalance handling
from imblearn.over_sampling import SMOTE

# Advanced models
import xgboost as xgb

# Model explainability
import shap
import lime
```

## Project Workflow

### Step 1: Data Loading & Exploration

```python
def load_and_explore_data(file_path):
    """Load and explore the fraud detection dataset"""
    df = pd.read_csv(file_path)
    
    print("Dataset Shape:", df.shape)
    print("\nClass Distribution:")
    print(df['Class'].value_counts())
    print(f"\nFraud Rate: {df['Class'].mean():.4f}")
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Basic statistics
    print("\nDataset Statistics:")
    print(df.describe())
    
    return df
```

### Step 2: Data Preprocessing & Feature Engineering

```python
def preprocess_data(df):
    """Preprocess the fraud detection data"""
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Feature scaling (important for fraud detection)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training fraud rate: {y_train.mean():.4f}")
    print(f"Test fraud rate: {y_test.mean():.4f}")
    
    return X_train, X_test, y_train, y_test, scaler
```

### Step 3: Handling Class Imbalance with SMOTE

```python
def apply_smote(X_train, y_train):
    """Apply SMOTE to handle class imbalance"""
    print("Before SMOTE:")
    print(f"Normal transactions: {sum(y_train == 0)}")
    print(f"Fraud transactions: {sum(y_train == 1)}")
    
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print("\nAfter SMOTE:")
    print(f"Normal transactions: {sum(y_train_smote == 0)}")
    print(f"Fraud transactions: {sum(y_train_smote == 1)}")
    
    return X_train_smote, y_train_smote
```

### Step 4: Model Training

```python
def train_models(X_train, y_train, X_train_smote, y_train_smote):
    """Train multiple models with and without SMOTE"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    trained_models = {}
    
    # Train on original data
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    # Train on SMOTE data
    for name, model in models.items():
        print(f"Training {name} with SMOTE...")
        model_smote = type(model)(random_state=42, n_estimators=100 if 'Forest' in name else None, 
                                  max_iter=1000 if 'Logistic' in name else None,
                                  eval_metric='logloss' if 'XGB' in name else None)
        model_smote.fit(X_train_smote, y_train_smote)
        trained_models[name + ' (SMOTE)'] = model_smote
    
    return trained_models
```

### Step 5: Model Evaluation

```python
def evaluate_models(models, X_test, y_test):
    """Evaluate all trained models"""
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:")
        print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # Detailed metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
    
    return pd.DataFrame(results).T
```

### Step 6: Hyperparameter Tuning

```python
def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for Random Forest"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    
    print("Performing hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_
```

### Step 7: Model Explainability

```python
def explain_model_shap(model, X_test, feature_names):
    """Explain model predictions using SHAP"""
    import shap
    
    # Create SHAP explainer
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    
    # Global feature importance
    print("Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values.values).mean(0)
    }).sort_values('importance', ascending=False).head(10)
    
    print(feature_importance)
    
    # Summary plot (would show in notebook)
    # shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    
    return shap_values

def explain_model_lime(model, X_test, feature_names, idx=0):
    """Explain individual predictions using LIME"""
    from lime.lime_tabular import LimeTabularExplainer
    
    explainer = LimeTabularExplainer(
        X_test.values,
        feature_names=feature_names,
        class_names=['Normal', 'Fraud'],
        mode='classification'
    )
    
    # Explain a specific instance
    explanation = explainer.explain_instance(
        X_test.iloc[idx].values,
        model.predict_proba,
        num_features=10
    )
    
    print(f"LIME Explanation for instance {idx}:")
    print(explanation.as_list())
    
    return explanation
```

### Step 8: Deployment Simulation

```python
def create_fraud_detection_api(model, scaler, threshold=0.5):
    """Create a fraud detection API simulation"""
    
    def predict_fraud(transaction_features):
        """Predict if a transaction is fraudulent"""
        # Preprocess the transaction
        transaction_scaled = scaler.transform([transaction_features])
        
        # Make prediction
        fraud_probability = model.predict_proba(transaction_scaled)[0][1]
        is_fraud = fraud_probability > threshold
        
        return {
            'is_fraud': bool(is_fraud),
            'fraud_probability': float(fraud_probability),
            'confidence': float(max(fraud_probability, 1 - fraud_probability)),
            'recommendation': 'BLOCK' if is_fraud else 'APPROVE'
        }
    
    return predict_fraud

# Example usage
def demonstrate_real_time_prediction(predict_func):
    """Demonstrate real-time fraud prediction"""
    # Example new transactions
    new_transactions = [
        # Normal transaction
        [100.0, 3600, 0.1, -0.2, 0.5, -0.3, 0.8, -0.1, 0.2, -0.4, 0.1, 0.3],
        # Suspicious transaction
        [5000.0, 86400, 3.2, -1.8, 2.1, -0.9, 1.5, -2.1, 0.8, -1.2, 2.3, -0.8]
    ]
    
    for i, transaction in enumerate(new_transactions):
        result = predict_func(transaction)
        print(f"Transaction {i+1}:")
        print(f"  Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"  Recommendation: {result['recommendation']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print()
```

## Expected Performance Results

Based on typical fraud detection scenarios:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|---------|----------|---------|
| Logistic Regression | 0.85 | 0.75 | 0.42 | 0.54 | 0.88 |
| Logistic Regression (SMOTE) | 0.82 | 0.65 | 0.89 | 0.75 | 0.91 |
| Random Forest | 0.89 | 0.82 | 0.58 | 0.68 | 0.92 |
| Random Forest (SMOTE) | 0.86 | 0.71 | 0.91 | 0.80 | 0.94 |
| XGBoost | 0.91 | 0.85 | 0.62 | 0.72 | 0.93 |
| XGBoost (SMOTE) | 0.88 | 0.74 | 0.93 | 0.82 | 0.95 |

## Key Insights

1. **SMOTE Impact**: Generally improves recall (fraud detection) but may reduce precision
2. **Model Performance**: XGBoost typically provides the best overall performance
3. **Metric Trade-offs**: Balance between precision and recall is crucial for fraud detection
4. **Class Imbalance**: SMOTE effectively addresses the class imbalance problem
5. **Feature Importance**: Anonymized features V1-V28 provide different levels of predictive power

## Deployment Considerations

### Real-Time Fraud Detection System Architecture

```python
# Example production deployment structure
class FraudDetectionSystem:
    def __init__(self, model_path, scaler_path, threshold=0.5):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.threshold = threshold
        
    def preprocess_transaction(self, transaction_data):
        """Preprocess incoming transaction data"""
        # Data validation
        # Feature engineering
        # Scaling
        return processed_data
    
    def predict_fraud(self, transaction_data):
        """Real-time fraud prediction"""
        processed_data = self.preprocess_transaction(transaction_data)
        fraud_probability = self.model.predict_proba([processed_data])[0][1]
        
        return {
            'fraud_probability': fraud_probability,
            'is_fraud': fraud_probability > self.threshold,
            'risk_level': self.get_risk_level(fraud_probability)
        }
    
    def get_risk_level(self, probability):
        """Determine risk level based on probability"""
        if probability < 0.3:
            return 'LOW'
        elif probability < 0.7:
            return 'MEDIUM'
        else:
            return 'HIGH'
```

### Model Monitoring and Maintenance

- **Performance Monitoring**: Track model accuracy, precision, recall over time
- **Data Drift Detection**: Monitor for changes in transaction patterns
- **Model Retraining**: Scheduled retraining with new data
- **A/B Testing**: Compare new models with production models
- **Audit Trail**: Maintain logs of all predictions and decisions

## Conclusion

This comprehensive fraud detection project demonstrates:

1. **Data Preprocessing**: Proper handling of imbalanced datasets
2. **Model Selection**: Comparison of multiple algorithms
3. **Performance Evaluation**: Comprehensive metrics analysis
4. **Explainability**: Understanding model decisions with SHAP/LIME
5. **Deployment**: Real-time prediction capabilities

The project provides a solid foundation for implementing fraud detection systems in production environments, with emphasis on both performance and interpretability.

## Next Steps

1. **Enhanced Feature Engineering**: Create domain-specific features
2. **Deep Learning Models**: Implement neural networks for complex patterns
3. **Ensemble Methods**: Combine multiple models for improved performance
4. **Real-time Processing**: Implement streaming data processing
5. **Regulatory Compliance**: Ensure model explanations meet regulatory requirements