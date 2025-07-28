

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv("C:\\Users\\ajmaq\\Downloads\\creditcard.csv")  

X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("STEP 4: MODEL TRAINING AND EVALUATION")
print("-" * 40)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

trained_models = {}
print("Training models on original (imbalanced) data...")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model

smote_models = {}
print("\nTraining models on SMOTE (balanced) data...")
for name, model in models.items():
    print(f"Training {name} with SMOTE...")
    if 'Forest' in name:
        model_smote = RandomForestClassifier(random_state=42, n_estimators=100)
    else:
        model_smote = LogisticRegression(random_state=42, max_iter=1000)
    model_smote.fit(X_train_smote, y_train_smote)
    smote_models[name + ' (SMOTE)'] = model_smote

print("\nModel training completed!")

