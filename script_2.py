# Create a comprehensive fraud detection project implementation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("COMPREHENSIVE FRAUD DETECTION PROJECT")
print("=" * 80)
print()

# Step 1: Create synthetic fraud detection dataset (simulating real-world data)
print("STEP 1: DATA GENERATION (Simulating Real Dataset)")
print("-" * 50)

# Generate synthetic dataset similar to credit card fraud detection
def generate_fraud_dataset(n_samples=10000):
    """
    Generate a synthetic fraud detection dataset similar to real credit card data
    """
    # Generate normal transactions
    normal_samples = int(n_samples * 0.998)  # 99.8% normal
    fraud_samples = n_samples - normal_samples  # 0.2% fraud
    
    # Normal transactions
    normal_data = {
        'Amount': np.random.lognormal(mean=2.5, sigma=1.2, size=normal_samples),
        'Time': np.random.uniform(0, 172800, size=normal_samples),  # 2 days in seconds
        'V1': np.random.normal(0, 1, size=normal_samples),
        'V2': np.random.normal(0, 1, size=normal_samples),
        'V3': np.random.normal(0, 1, size=normal_samples),
        'V4': np.random.normal(0, 1, size=normal_samples),
        'V5': np.random.normal(0, 1, size=normal_samples),
        'V6': np.random.normal(0, 1, size=normal_samples),
        'V7': np.random.normal(0, 1, size=normal_samples),
        'V8': np.random.normal(0, 1, size=normal_samples),
        'V9': np.random.normal(0, 1, size=normal_samples),
        'V10': np.random.normal(0, 1, size=normal_samples),
        'Class': np.zeros(normal_samples)
    }
    
    # Fraudulent transactions (with different patterns)
    fraud_data = {
        'Amount': np.random.lognormal(mean=3.2, sigma=1.5, size=fraud_samples),
        'Time': np.random.uniform(0, 172800, size=fraud_samples),
        'V1': np.random.normal(2, 1.5, size=fraud_samples),  # Different distribution
        'V2': np.random.normal(-1, 1.2, size=fraud_samples),
        'V3': np.random.normal(1.5, 1.3, size=fraud_samples),
        'V4': np.random.normal(-0.5, 1.1, size=fraud_samples),
        'V5': np.random.normal(0.8, 1.4, size=fraud_samples),
        'V6': np.random.normal(-1.2, 1.2, size=fraud_samples),
        'V7': np.random.normal(0.5, 1.3, size=fraud_samples),
        'V8': np.random.normal(-0.8, 1.1, size=fraud_samples),
        'V9': np.random.normal(1.2, 1.4, size=fraud_samples),
        'V10': np.random.normal(-0.6, 1.2, size=fraud_samples),
        'Class': np.ones(fraud_samples)
    }
    
    # Combine normal and fraud data
    combined_data = {}
    for key in normal_data.keys():
        combined_data[key] = np.concatenate([normal_data[key], fraud_data[key]])
    
    # Create DataFrame
    df = pd.DataFrame(combined_data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# Generate the dataset
df = generate_fraud_dataset(10000)

print(f"Dataset generated successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Normal transactions: {len(df[df['Class'] == 0])} ({len(df[df['Class'] == 0])/len(df)*100:.2f}%)")
print(f"Fraudulent transactions: {len(df[df['Class'] == 1])} ({len(df[df['Class'] == 1])/len(df)*100:.2f}%)")
print()

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())
print()

# Basic statistics
print("Dataset statistics:")
print(df.describe())
print()

print("Dataset generation completed successfully!")
print("=" * 80)