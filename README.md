### 🕵️‍♂️ Fraud Detection using Reusable Classification & EDA Modules
This project focuses on detecting fraudulent transactions using various machine learning classification models. The solution is designed to be modular and reusable, with separate files for preprocessing, EDA, and modeling. It provides a framework to plug in different datasets and quickly apply end-to-end fraud detection analysis.

### 📁 Project Structure
Fraud_detection.ipynb:
Main Jupyter Notebook demonstrating the full workflow – from data loading, cleaning, and visualization to model training and evaluation.

-Classification_models_reusable.py:
A Python module containing reusable functions for classification models including Logistic Regression, Random Forest, XGBoost, SVC, KNN, and MLP. Each model has its own function with built-in preprocessing, cross-validation, and evaluation (confusion matrix, classification report, ROC/AUC).

-EDA_template.py:
A reusable EDA (Exploratory Data Analysis) module providing functions to:

Inspect data structure

Identify missing values

Plot distributions and boxplots

Normalize or standardize features

Apply log transformations

Compute correlations

Group and summarize values

🔍 Features
Supports multiple classifiers via a single-line function call per model.

Built-in scaling (StandardScaler).

Evaluation metrics include Accuracy, Precision, Recall, F1 Score, and AUC.

Visualizations: Confusion Matrix and ROC Curves.

EDA support with histograms, boxplots, and correlation analysis.

