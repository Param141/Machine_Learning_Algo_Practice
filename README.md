# Machine Learning Supervised Learning Benchmarks

This repository contains practical implementations and benchmark experiments for foundational supervised learning algorithms, ranging from basic regression and classification to end-to-end data pipelines, hyperparameter tuning, and ensemble modeling. Notebooks include step-by-step preprocessing, modeling, hyperparameter search and evaluation.

---

## 📌 Project 1: Medical Insurance Cost Prediction (Regression)

Objective: Predict individual medical insurance charges (`charges`) based on physical, demographic, and lifestyle features (`age`, `bmi`, `children`, `sex`, `smoker`, `region`).

Data Preprocessing & Cleaning:
- Handled target skewness and target outliers in `y_train` using the Interquartile Range (IQR) method (Q1 − 1.5 × IQR to Q3 + 1.5 × IQR).
- Scaled continuous numerical features using `StandardScaler` fitted strictly on training data.
- Encoded categorical features (`sex`, `smoker`, `region`) using One-Hot Encoding (`pd.get_dummies(drop_first=True)`).

Model Iterations & Evaluation:
- Simple Linear Regression (age only): baseline linear relationship between age and charges.
- Multiple Linear Regression (age, bmi, children): expanded feature set.
- Full Multiple Linear Regression (all features): evaluated using Mean Squared Error (MSE) and R² score.

---

## 📌 Project 2: Social Network Ads (Binary Classification)

Objective: Predict user purchase intent (`Purchased`: 0 or 1) from `Age` and `EstimatedSalary`.

Preprocessing:
- Standardized features with `StandardScaler` (important for distance-based classifiers).

Hyperparameter Optimization:
- `GridSearchCV` with 5-fold CV to tune logistic-regression hyperparameters (C, penalty, solver).

Model Progression & Benchmarking:
- Logistic Regression (Baseline): 86.00% Test Accuracy
- Linear SVM: 86.25% Test Accuracy
- Logistic Regression (GridSearchCV): 88.75% Test Accuracy
- RBF Kernel SVM (Best): 92.50% Test Accuracy 🏆

Key Finding: Non-linear decision boundaries (RBF SVM) substantially improved separation of age / salary clusters.

---

## 📌 Project 3: Regression Benchmarks (Simple & Multiple Linear Regression)

Datasets: `Salary_Data.csv` (simple) and `50_Startups.csv` (multiple).

Objectives:
- Predict salary from years of experience (simple linear regression).
- Predict startup `Profit` from `R&D Spend`, `Administration`, `Marketing Spend` and `State`.

Preprocessing & Feature Engineering:
- `StandardScaler` applied after splitting to avoid leakage.
- One-hot encode `State` (`pd.get_dummies(drop_first=True)`), align train/test columns with `df.align(...)`.

Key Findings:
- Linear relationships captured baseline effects; `R&D Spend` often emerges as the most influential predictor.

---

## 📌 Project 4: Used Car Price Prediction (Cardekho Regression Benchmark)

Dataset: `cardekho_imputated.csv` (~15,411 records).

Objective: Predict `selling_price` using vehicle specs, age, mileage, engine, fuel, transmission, seller type.

Preprocessing & Pipeline:
- Dropped high-cardinality strings (`car_name`, `brand`), `LabelEncoder` for `model`, `OneHotEncoder(drop='first')` for categorical features.
- `ColumnTransformer` combining `StandardScaler` for numerical columns and encoders for categorical columns.
- 80/20 train/test split (12,328 train / 3,083 test).

Model Evaluation & Tuning:
- Linear Regression / Lasso / Ridge: Test R² ≈ 0.6645
- K-Neighbors Regressor: Test R² ≈ 0.9149
- Decision Tree Regressor: Test R² ≈ 0.8728 (overfitting observed)
- AdaBoost Regressor: Test R² ≈ 0.5739
- Gradient Boosting Regressor: Test R² ≈ 0.9126
- Random Forest Regressor (tuned): Test R² ≈ 0.9406 (RandomizedSearchCV)
- XGBoost Regressor (best): Test R² ≈ 0.9511 (RandomizedSearchCV) 🏆

---

## 📌 Project 5: Holiday Package Purchase Prediction (Wellness Tourism Classification)

Dataset: `Travel.csv` (4,888 customer profiles).

Objective: Predict `ProdTaken` (0/1) to optimize marketing.

Preprocessing & Feature Engineering:
- Cleaned inconsistent category labels (e.g., `'Fe Male'` → `'Female'`).
- Imputed medians for continuous features and mode for categorical/discrete attributes.
- Engineered `TotalVisiting = NumberOfPersonVisiting + NumberOfChildrenVisiting`.
- Applied `ColumnTransformer` with `OneHotEncoder(drop='first')` across categorical columns and `StandardScaler` to numerical features.

Model Benchmarking:
- Logistic Regression: Test Accuracy = 83.54%, ROC-AUC = 0.6301
- AdaBoost: Test Accuracy = 83.54%, ROC-AUC = 0.6400
- Gradient Boosting: Test Accuracy = 85.89%, ROC-AUC = 0.6824
- Decision Tree: Test Accuracy = 91.92%, ROC-AUC = 0.8626
- Random Forest (tuned): Test Accuracy = 93.05%, Precision = 0.9624
- XGBoost (best): Test Accuracy = 95.09%, Precision = 0.9554, Recall = 0.7853, F1 = 0.9490, ROC-AUC = 0.8882 🏆

---

## 🛠️ Tech Stack & Libraries
- Language: Python 3.8+
- Data processing: pandas, numpy
- Visualization: matplotlib, seaborn, plotly
- ML & pipelines: scikit-learn (LinearRegression, LogisticRegression, SVC, DecisionTree*, RandomForest*, KNeighbors*, AdaBoost*, GradientBoost*, ColumnTransformer, StandardScaler, OneHotEncoder, LabelEncoder, GridSearchCV, RandomizedSearchCV), XGBoost

---

## 📁 Repository Structure (actual top-level)

```text
├── 3.0-KNNClassifier.ipynb
├── 3.0-KNNRegressor.ipynb
├── 3.0-Naive+Bayes+Implementation.ipynb
├── Adaboost+Classification+Implementation.ipynb
├── Adaboost+Regression+Implementation.ipynb
├── Basic+SVC+Implementation.ipynb
├── DBSCAN+Implementation.ipynb
├── Decision+Tree+Classifier+Practical+Implementation.ipynb
├── Diabetes+Prediction+Using+Decision+Tree+Regressor.ipynb
├── GradientBoost+Classification+Implementation.ipynb
├── Gradientboost+Regression+Implementation.ipynb
├── Hierarichal+Clustering+Implementation.ipynb
├── K+Means+Clustering+Algorithms+implementation.ipynb
├── LogisticRegression_Social_Network_Ads.ipynb
├── MultipleLR_StartupDataset.ipynb
├── Principal+Component+Analysis+(PCA)+Implementation.ipynb
├── Random+Forest+Classification+Implementation.ipynb
├── Random+Forest+Regression+Implementation.ipynb
├── SVM+Kernels+Implementation.ipynb
├── Simple_Linear_Regression_SalaryDataset.ipynb
├── Support+Vector+Regression+Implementation.ipynb
├── Xgboost+Regression+Implementation.ipynb
├── XgboostBoost+Classification+Implementation.ipynb
├── logisticregression_social_network_ads.py
├── multiplelinearregression.py
├── multiplelr_startupdataset.py
├── simple_linear_regression_salarydataset.py
└── README.md
