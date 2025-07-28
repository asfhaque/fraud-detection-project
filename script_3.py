import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load the dataset
df = pd.read_csv("C:\\Users\\ajmaq\\Downloads\\creditcard.csv")  

print("STEP 2: DATA EXPLORATION AND PREPROCESSING")
print("-" * 50)

# Data exploration
print("Data Shape:", df.shape)
print("\nClass Distribution:")
class_counts = df['Class'].value_counts()
print(class_counts)
print(f"\nClass Distribution Percentages:")
for class_val, count in class_counts.items():
    percentage = (count / len(df)) * 100
    class_name = "Normal" if class_val == 0 else "Fraud"
    print(f"{class_name}: {count} ({percentage:.2f}%)")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTransaction Amount Analysis:")
print(f"Normal transactions - Mean: ${df[df['Class']==0]['Amount'].mean():.2f}")
print(f"Fraud transactions - Mean: ${df[df['Class']==1]['Amount'].mean():.2f}")
print(f"Normal transactions - Median: ${df[df['Class']==0]['Amount'].median():.2f}")
print(f"Fraud transactions - Median: ${df[df['Class']==1]['Amount'].median():.2f}")

print("\n" + "="*80)

# Step 3: Data Preprocessing
print("STEP 3: DATA PREPROCESSING")
print("-" * 30)

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("Feature scaling completed")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training set fraud rate: {y_train.mean():.4f}")
print(f"Test set fraud rate: {y_test.mean():.4f}")

print("\n" + "="*80)

# Step 4: Handling Class Imbalance with SMOTE
print("STEP 4: HANDLING CLASS IMBALANCE WITH SMOTE")
print("-" * 45)

# Before SMOTE
print("Before SMOTE:")
print(f"Normal transactions: {sum(y_train == 0)}")
print(f"Fraud transactions: {sum(y_train == 1)}")

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"Normal transactions: {sum(y_train_smote == 0)}")
print(f"Fraud transactions: {sum(y_train_smote == 1)}")
print(f"Total training samples: {len(X_train_smote)}")

print("\n" + "="*80)

# Step 5: Model Training
print("STEP 5: MODEL TRAINING")
print("-" * 25)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

# Train models on original imbalanced data
trained_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} training completed")

# Train models on SMOTE data with correct initialization
smote_models = {}
for name, model in models.items():
    print(f"Training {name} with SMOTE...")
    if 'Forest' in name:
        model_smote = RandomForestClassifier(random_state=42, n_estimators=100)
    elif 'Logistic' in name:
        model_smote = LogisticRegression(random_state=42, max_iter=1000)
    else:
        model_smote = type(model)(random_state=42)

    model_smote.fit(X_train_smote, y_train_smote)
    smote_models[name + ' (SMOTE)'] = model_smote
    print(f"{name} with SMOTE training completed")

print("\nModel training completed!")
print("=" * 80)
