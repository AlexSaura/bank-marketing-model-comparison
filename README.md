# Bank Marketing Classifier Comparison

This project compares the performance of several classification algorithms to predict whether a customer will subscribe to a term deposit based on data collected from a Portuguese banking institution's marketing campaigns.

## Dataset

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- Size: 41,188 rows, 21 columns
- Period: 17 direct marketing campaigns conducted between 2008 and 2013
- Target variable: `y` (yes/no) — indicates if the client subscribed to a term deposit

## Objective

The goal is to predict client subscription outcomes using machine learning models. By doing so, the bank can target clients more effectively, reduce campaign costs, and improve overall success rates. This project compares baseline accuracy, logistic regression, decision trees, K-nearest neighbors, and support vector machines.

## Project Workflow

### 1. Data Understanding

- Used `.info()` and `.dtypes` to inspect data types
- Checked for null values and “unknown” labels in categorical features
- Identified and excluded the `duration` feature due to data leakage concerns

### 2. Feature Engineering

- Selected key banking-related features
- Encoded categorical variables using one-hot encoding
- Converted the target variable to binary (`yes` = 1, `no` = 0)

### 3. Train/Test Split

- Performed an 80/20 train-test split with stratification on the target
- Resulting shapes:  
  - Train: (32,950 rows, 42 features)  
  - Test: (8,238 rows, 42 features)

### 4. Baseline Model

- Majority class baseline: always predict 'no' (class 0)
- **Baseline accuracy**: **0.8874**

### 5. Logistic Regression

- Model trained using default parameters with `max_iter=1000`
- **Accuracy**: 0.8878
- **Classification Report**:
  - Precision: 0.89 (class 0), 0.51 (class 1)
  - Recall: 0.99 (class 0), 0.07 (class 1)
  - F1-score: 0.94 (class 0), 0.13 (class 1)
- **Confusion Matrix**:
[[7245 65]
[ 859 69]]


### 6. Model Comparisons

| Model               | Train Time (s) | Train Accuracy | Test Accuracy |
|--------------------|----------------|----------------|---------------|
| Logistic Regression| 0.2514         | 0.8872         | 0.8878        |
| K-Nearest Neighbors| 0.0030         | 0.9030         | 0.8795        |
| Decision Tree      | 0.1054         | 0.9774         | 0.8328        |
| SVM                | 7.2432         | 0.8873         | 0.8874        |

### 7. Insights

- Logistic Regression and SVM provided the best balance of accuracy and generalization.
- Decision Tree overfit the training data despite its speed.
- KNN performed reasonably well with minimal training time but slightly lower test accuracy.

### 8. Improvements Explored

- Suggested future enhancements:
- Feature selection: drop unhelpful features such as high-cardinality or low-signal attributes
- Hyperparameter tuning: use GridSearchCV for KNN (`n_neighbors`) or Decision Trees (`max_depth`)
- Alternative metrics: use F1-score or ROC-AUC to better capture minority class performance

## Tools Used

- Python
- pandas, NumPy
- scikit-learn
- Jupyter Notebook
