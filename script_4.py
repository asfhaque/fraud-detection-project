import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

print("CREATING LARGER DATASET FOR PROPER SMOTE APPLICATION")
print("-" * 55)

# Generate a larger dataset with more fraud samples
def generate_larger_fraud_dataset(n_samples=50000):
    """Generate a larger synthetic fraud detection dataset"""
    
    # Generate normal transactions (99.5% normal)
    normal_samples = int(n_samples * 0.995)
    fraud_samples = n_samples - normal_samples  # 0.5% fraud
    
    # Normal transactions
    normal_data = {
        'Amount': np.random.lognormal(mean=2.5, sigma=1.2, size=normal_samples),
        'Time': np.random.uniform(0, 172800, size=normal_samples),
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
        'V1': np.random.normal(2, 1.5, size=fraud_samples),
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
    
    # Combine data
    combined_data = {}
    for key in normal_data.keys():
        combined_data[key] = np.concatenate([normal_data[key], fraud_data[key]])
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(combined_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# Generate larger dataset
df_large = generate_larger_fraud_dataset(50000)

print(f"Large dataset created!")
print(f"Dataset shape: {df_large.shape}")
print(f"Normal transactions: {len(df_large[df_large['Class'] == 0])} ({len(df_large[df_large['Class'] == 0])/len(df_large)*100:.2f}%)")
print(f"Fraudulent transactions: {len(df_large[df_large['Class'] == 1])} ({len(df_large[df_large['Class'] == 1])/len(df_large)*100:.2f}%)")

# Now proceed with the analysis using the larger dataset
print("\n" + "="*80)
print("STEP 2: DATA EXPLORATION AND PREPROCESSING (UPDATED)")
print("-" * 50)

# Use the larger dataset
df = df_large

# Data exploration
print("Data Shape:", df.shape)
print("\nClass Distribution:")
class_counts = df['Class'].value_counts()
print(class_counts)

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

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training set fraud samples: {sum(y_train == 1)}")
print(f"Test set fraud samples: {sum(y_test == 1)}")

print("\n" + "="*80)

# Step 3: Apply SMOTE (should work now with more samples)
print("STEP 3: HANDLING CLASS IMBALANCE WITH SMOTE")
print("-" * 45)

print("Before SMOTE:")
print(f"Normal transactions: {sum(y_train == 0)}")
print(f"Fraud transactions: {sum(y_train == 1)}")

# Apply SMOTE with adjusted parameters
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"Normal transactions: {sum(y_train_smote == 0)}")
print(f"Fraud transactions: {sum(y_train_smote == 1)}")
print(f"Total training samples: {len(X_train_smote)}")

print("\nSMOTE application successful!")
print("=" * 80)