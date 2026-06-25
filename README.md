# Fruit Classification — KNN Classifier

A supervised machine learning project for **multi-class fruit classification** using the K-Nearest Neighbors algorithm, with hyperparameter tuning and performance evaluation.

## Business Context

TropicTaste Inc., a distributor of exotic fruits, aims to automate its manual fruit classification process. An accurate ML model reduces human error, optimizes inventory management, and ensures consistent quality standards.

## Dataset

Numerical features describing fruit characteristics (weight, size, color metrics, etc.) with a `Frutto` target column.

> Dataset: place `fruits.csv` in the same folder as the notebook before running.

## Approach

1. **EDA** — class distribution, descriptive statistics, data types check
2. **Preprocessing** — LabelEncoder on target, train/test split (80/20, stratified), StandardScaler
3. **Model selection** — KNN trained with k ∈ {1, 2, 3, 5, 7, 8, 9, 10}, comparing train vs test accuracy to detect overfitting
4. **Evaluation** — classification report (precision, recall, F1-score per class)

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn (`KNeighborsClassifier`, `LabelEncoder`, `StandardScaler`, `classification_report`)
- matplotlib

## Project Structure

```
fruit_classification_knn.ipynb   # Full pipeline: EDA → preprocessing → KNN tuning → evaluation
fruits.csv                       # Dataset (not included — see note above)
```

## Key Concepts

- Multi-class classification
- K-Nearest Neighbors and the effect of k on bias/variance
- Stratified train/test split
- Evaluation metrics beyond accuracy (precision, recall, F1)
