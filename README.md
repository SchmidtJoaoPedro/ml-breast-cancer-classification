# ML Breast Cancer Classification

This project is a **complete Machine Learning pipeline example** using Python and scikit-learn, designed to **classify breast cancer tumors** as benign or malignant.

---

## What the code does

1. **Loads the breast cancer dataset** (`breast_cancer`) from scikit-learn.  
2. **Splits the dataset** into input features (`X`) and target labels (`y`).  
3. **Splits the data** into training (80%) and testing (20%) sets.  
4. **Creates a ML pipeline** with:
   - `StandardScaler` to normalize the data.
   - `RandomForestClassifier` for classification.
5. **Finds the best hyperparameters** using `GridSearchCV`.  
6. **Trains the final model** with the best parameters.  
7. **Makes predictions** on the test data.  
8. **Evaluates the model** using metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC AUC
   - Confusion matrix
9. **Displays results** in the terminal.  
10. **Saves the trained model** as `model_breast_cancer_rf.joblib` for future use.

---

## Requirements

- Python 3.8 or higher  
- Python libraries:
  ```bash
  pip install scikit-learn pandas joblib
