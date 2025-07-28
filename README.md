# Fraud Detection Using Machine Learning

## Project Overview

This project implements an end-to-end machine learning pipeline to detect fraudulent financial transactions. It simulates a real-world fraud detection scenario by generating a synthetic, highly imbalanced dataset and building models to classify fraud versus normal transactions effectively.

## Features

- **Synthetic Dataset Generation:** Creates realistic credit card transaction data with normal and fraudulent activities.
- **Data Exploration:** Explores dataset shape, class distribution, missing values, and transaction statistics.
- **Preprocessing Pipeline:** Includes feature scaling and stratified train-test split.
- **Imbalance Handling:** Applies SMOTE (Synthetic Minority Oversampling Technique) to balance the training set.
- **Model Training:** Trains Logistic Regression and Random Forest classifiers.
- **Evaluation:** Comprehensive model evaluation using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
- **Subset Training for Speed:** Allows training on subsets of data for faster experiments.

## Getting Started

### Prerequisites

- numpy
- pandas
- scikit-learn
- imbalanced-learn



