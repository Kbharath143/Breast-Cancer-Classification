# Breast-Cancer-Classification

#Breast Cancer Classification using SVM 

##Objective 

To build and evaluate a Support Vector Machine (SVM) model to classify breast cancer tumors as Benign or Malignant using 
the Breast Cancer dataset from Scikit-learn. 

##Problem Statement 

Early detection of breast cancer is critical. This task uses machine learning to classify tumors based on medical features. 
SVM is chosen because it performs well on high-dimensional data and binary classification problems. 

##Dataset Details 

• Dataset Source: sklearn.datasets.load_breast_cancer 

• Total Samples: 569 

• Features: 30 numerical features 

• Target Variable: 

o 0 → Malignant 

o 1 → Benign 

##Tools & Libraries Used 

• Python 

• NumPy 

• Pandas 

• Scikit-learn 

• Matplotlib 

• Jupyter Notebook 

##Step-by-Step Procedure 
1. Load the Dataset 

The breast cancer dataset is loaded using Scikit-learn and split into features (X) and target (y). 

2. Data Preprocessing 

• Checked dataset shape and target distribution 

• Applied StandardScaler to normalize feature values 

3. Train-Test Split 

The dataset is split into: 

• 80% Training data 

• 20% Testing data 

4. Train SVM (Linear Kernel) 

A baseline SVM model with a linear kernel is trained and evaluated. 

5. Train SVM (RBF Kernel) 

An SVM model with RBF kernel is trained to capture non-linear patterns. 

6. Hyperparameter Tuning 

GridSearchCV is used to find optimal values for: 

• C (Regularization parameter) 

• Gamma (Kernel coefficient) 

7. Model Evaluation 

The best model is evaluated using: 

• Accuracy Score 

• Confusion Matrix 

• Classification Report 

• ROC Curve and AUC Score 

##Results 

• RBF kernel outperformed linear kernel 

• Achieved high accuracy (above 95%) 

• ROC-AUC score indicated strong classification performance 

##Files Included 

• SVM_Breast_Cancer.ipynb – Implementation Notebook 

• README.md – Project Documentation 

##Conclusion 

This task demonstrates the effectiveness of Support Vector Machines in medical diagnosis problems. Proper feature scaling 
and hyperparameter tuning significantly improve model performance. 
