# Machine Learning Supervised Learning Benchmarks

This repository contains practical implementations and benchmark experiments for foundational Supervised Learning algorithms, ranging from basic regression and classification to end-to-end data pipelines, hyperparameter tuning, and ensemble modeling.

---

## 📌 Project 1: Medical Insurance Cost Prediction (Regression)

**Objective:** Predict individual medical insurance charges (`charges`) based on physical, demographic, and lifestyle features (`age`, `bmi`, `children`, `sex`, `smoker`, `region`).

**Data Preprocessing & Cleaning:**
* Handled target skewness and target outliers in `y_train` using the Interquartile Range (IQR) method ($Q_1 - 1.5 \times \text{IQR}$ to $Q_3 + 1.5 \times \text{IQR}$).
* Scaled continuous numerical features using `StandardScaler` fitted strictly on training data.
* Encoded categorical features (`sex`, `smoker`, `region`) using One-Hot Encoding (`pd.get_dummies(drop_first=True)`).

**Model Iterations & Evaluation:**
* **Simple Linear Regression (age only):** Evaluated baseline linear relationship between age and charges.
* **Multiple Linear Regression (age, bmi, children):** Expanded feature set to capture additional physical metrics.
* **Full Multiple Linear Regression (All Features Included):** Combined scaled numerical metrics with one-hot encoded categorical variables to evaluate overall model performance using Mean Squared Error (MSE) and $R^2$ score.

---

## 📌 Project 2: Social Network Ads (Binary Classification)

**Objective:** Predict user purchase intent (`Purchased`: 0 or 1) based on user demographic features (`Age` and `EstimatedSalary`).

**Preprocessing:** Standardized feature distributions using `StandardScaler` for distance-based classification algorithms.

**Hyperparameter Optimization:** Utilized `GridSearchCV` with 5-fold cross-validation to fine-tune inverse regularization ($C$), penalty norms, and solvers, improving Logistic Regression accuracy from 86.00% to 88.75%.

**Model Progression & Benchmarking:**
* **Logistic Regression (Baseline):** 86.00% Test Accuracy
* **Linear SVM:** 86.25% Test Accuracy
* **Logistic Regression (GridSearchCV Tuned):** 88.75% Test Accuracy
* **RBF Kernel SVM (Best Model):** 92.50% Test Accuracy 🏆

**Key Finding:** Transitioning from linear models to an RBF Kernel SVM yielded a +6.5% performance boost, proving that non-linear decision boundaries are required to separate user age and salary interaction clusters.

---

## 📌 Project 3: Regression Benchmarks (Simple & Multiple Linear Regression)

**Datasets Used:** `Salary_Data.csv` (Simple) and `50_Startups.csv` (Multiple).

**Objectives:**
* Predict salary based on years of experience (1D Linear Regression).
* Predict startup Profit based on operational spending (`R&D Spend`, `Administration`, `Marketing Spend`) and `State` location.

**Preprocessing & Feature Engineering:**
* Applied `StandardScaler` on numerical features after splitting data to prevent data leakage.
* Encoded categorical location (`State`) via One-Hot Encoding (`pd.get_dummies(drop_first=True)`), preventing the Dummy Variable Trap.
* Synchronized feature columns between train and test splits using `df.align(join='left', axis=1, fill_value=0)`.

**Model Evaluation & Key Findings:**
* **Simple Linear Regression:** Established baseline $y = mx + c$ relationship between experience and income.
* **Multiple Linear Regression:** Evaluated partial regression coefficients (`model.coef_`) to rank spending impact (`R&D Spend` emerged as the primary profit driver).
* **Low-Correlation Feature Impact:** Bivariate correlation showed `Administration` had only ~9% linear correlation with `Profit`, but model weights demonstrated its role as a control variable alongside `R&D` and `Marketing` spending.

---

## 📌 Project 4: Used Car Price Prediction (Cardekho Regression Benchmark)

**Dataset Used:** `cardekho_imputated.csv` (15,411 records, scraped from CarDekho.com)[cite: 1].

**Objective:** Predict second-hand car market selling prices (`selling_price`) based on vehicle specifications, age, mileage, engine size, and fuel/transmission types[cite: 1].

**Preprocessing & Feature Pipeline:**
* **Categorical Encoding:** Applied `LabelEncoder` for high-cardinality model names (120 unique models)[cite: 1]. Used `ColumnTransformer` with `OneHotEncoder(drop='first')` for categorical attributes (`seller_type`, `fuel_type`, `transmission_type`)[cite: 1].
* **Feature Scaling:** Standardized continuous/discrete numerical columns (`vehicle_age`, `km_driven`, `mileage`, `engine`, `max_power`, `seats`, `model`) via `StandardScaler`[cite: 1].
* **Data Leakage Prevention:** Built a unified `ColumnTransformer` pipeline fit exclusively on `X_train`[cite: 1].

**Model Evaluation & Benchmarking:**
* **Linear / Ridge / Lasso Regression:** Test $R^2 \approx 0.6645$[cite: 1].
* **K-Neighbors Regressor:** Test $R^2 = 0.9149$[cite: 1].
* **Decision Tree Regressor:** Test $R^2 = 0.8725$ (High training overfit at $R^2 = 0.9995$)[cite: 1].
* **Random Forest Regressor (Tuned via `RandomizedSearchCV`):** Test $R^2 = 0.9307$ (Test RMSE: 228,415) 🏆[cite: 1]

---

## 📌 Project 5: Holiday Package Purchase Prediction (Wellness Tourism Classification)

**Dataset Used:** `Travel.csv` (4,888 customer profiles from Trips & Travel.Com)[cite: 2].

**Objective:** Predict whether a customer will purchase a new Wellness Tourism Package (`ProdTaken`: 0 or 1) to optimize targeted marketing spend[cite: 2].

**Preprocessing & Feature Engineering:**
* **Data Cleaning:** Cleaned category typos (e.g., merging `'Fe Male'` to `'Female'`, `'Single'` to `'Unmarried'`)[cite: 2]. Imputed missing values using Medians (for continuous features like `Age`, `MonthlyIncome`) and Modes (for categorical/discrete features like `TypeofContact`, `NumberOfFollowups`)[cite: 2].
* **Feature Extraction:** Engineered a composite interaction variable `TotalVisiting = NumberOfPersonVisiting + NumberOfChildrenVisiting`[cite: 2].
* **Pipeline Transformation:** Used `ColumnTransformer` combining `OneHotEncoder(drop='first')` across categorical columns (`Occupation`, `Gender`, `ProductPitched`, `MaritalStatus`, `Designation`) and `StandardScaler` across numeric metrics[cite: 2].

**Model Evaluation & Benchmarking:**
* **Logistic Regression:** Test Accuracy = 83.54%, Test ROC-AUC = 0.6301[cite: 2]
* **Gradient Boosting Classifier:** Test Accuracy = 85.89%, Test ROC-AUC = 0.6824[cite: 2]
* **Decision Tree Classifier:** Test Accuracy = 92.54%, Test ROC-AUC = 0.8723[cite: 2]
* **Random Forest Classifier (Tuned via `RandomizedSearchCV`):** Test Accuracy = 93.15%, Test Precision = 0.9697, Test ROC-AUC = 0.8325 🏆[cite: 2]

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.x[cite: 1, 2]
* **Data Manipulation & EDA:** Pandas, NumPy[cite: 1, 2]
* **Data Visualization:** Matplotlib, Seaborn, Plotly Express[cite: 1, 2]
* **Machine Learning:** Scikit-Learn (`LinearRegression`, `LogisticRegression`, `SVC`, `KNeighborsRegressor`, `DecisionTreeRegressor`, `DecisionTreeClassifier`, `RandomForestRegressor`, `RandomForestClassifier`, `GradientBoostingClassifier`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `LabelEncoder`, `GridSearchCV`, `RandomizedSearchCV`)[cite: 1, 2]

---

## 📁 Repository Structure

```text
├── 50_Startups.csv                         # Startups Expenditure & Profit Dataset
├── Salary_Data.csv                         # Salary vs Experience Dataset
├── Social_Network_Ads.csv                  # Social Network Ads Classification Dataset
├── cardekho_imputated.csv                  # Cardekho Used Cars Price Dataset
├── Travel.csv                              # Holiday Package Purchase Prediction Dataset
├── Simple_And_Multiple_Linear_Regression.ipynb # Notebook for Regression Experiments
├── Social_Network_Ads_Classification.ipynb # Notebook for Classification Experiments
├── Used_Car_Price_Prediction.ipynb         # Notebook for Car Price Regression (Random Forest & KNN)
├── Holiday_Package_Prediction.ipynb        # Notebook for Tourism Package Classification (Ensemble Methods)
└── README.md                               # Project documentation
