Here is an updated **README.md** tailored for your GitHub repository. It expands on your existing content, integrating **Project 4** and **Project 5** with exact data attributes, preprocessing pipelines, model evaluation metrics, and hyperparameter results derived directly from your notebooks.

---

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

**Dataset Used:** `cardekho_imputated.csv` (15,411 records scraped from CarDekho.com).

**Objective:** Predict second-hand car market selling prices (`selling_price`) based on vehicle specifications, age, mileage, engine size, fuel type, transmission type, and seller type.

**Data Preprocessing & Feature Pipeline:**

* **Feature Selection & Encoding:** Dropped high-cardinality/redundant string attributes (`car_name`, `brand`). Encoded high-cardinality `model` (120 unique models) using `LabelEncoder`. Applied `OneHotEncoder(drop='first')` across low-cardinality categorical features (`seller_type`, `fuel_type`, `transmission_type`).


* **Feature Scaling:** Scaled continuous and discrete numerical columns (`vehicle_age`, `km_driven`, `mileage`, `engine`, `max_power`, `seats`, `model`) via `StandardScaler` inside a unified `ColumnTransformer` pipeline.


* **Train/Test Split:** Evaluated performance using an 80/20 train-test split (12,328 train / 3,083 test samples).



**Model Evaluation & Hyperparameter Tuning:**

* **Linear / Ridge / Lasso Regression:** Baseline test $R^2 = 0.6645$ (Test RMSE: 502,543).


* **K-Neighbors Regressor (Tuned $K=10$):** Test $R^2 = 0.9075$ (Test MAE: 117,483).


* **Decision Tree Regressor:** Test $R^2 = 0.8759$ (High overfit with Train $R^2 = 0.9995$).


* **AdaBoost Regressor:** Test $R^2 = 0.7284$.


* **Random Forest Regressor (Best Model):** Achieved **Test $R^2 = 0.9315$** (Test MAE: 102,202). Tuned via `RandomizedSearchCV` (`n_estimators=500`, `max_features=8`, `min_samples_split=2`). 🏆



---

## 📌 Project 5: Holiday Package Purchase Prediction (Wellness Tourism Classification)

**Dataset Used:** `Travel.csv` (4,888 customer profiles from Trips & Travel.Com).

**Objective:** Predict whether a customer will purchase a newly introduced Wellness Tourism Package (`ProdTaken`: 0 or 1) to optimize targeted marketing efficiency.

**Data Preprocessing & Feature Engineering:**

* **Category Cleaning:** Standardized inconsistent labels (e.g., merged `'Fe Male'` into `'Female'`, `'Single'` into `'Unmarried'`).


* **Missing Value Imputation:** Imputed median values for continuous features (`Age`, `DurationOfPitch`, `NumberOfTrips`, `MonthlyIncome`) and mode values for categorical/discrete attributes (`TypeofContact`, `NumberOfFollowups`, `PreferredPropertyStar`, `NumberOfChildrenVisiting`).


* **Feature Extraction:** Engineered composite interaction variable `TotalVisiting = NumberOfPersonVisiting + NumberOfChildrenVisiting`.


* **Preprocessing Pipeline:** Integrated a unified `ColumnTransformer` combining `OneHotEncoder(drop='first')` across 6 categorical columns (`TypeofContact`, `Occupation`, `Gender`, `ProductPitched`, `MaritalStatus`, `Designation`) and `StandardScaler` across 11 numerical features.



**Model Benchmarking & Tuning Results:**

* **Logistic Regression:** Test Accuracy = 83.54%, Test ROC-AUC = 0.6301, Test Precision = 0.6829


* **AdaBoost Classifier:** Test Accuracy = 83.64%, Test ROC-AUC = 0.6049


* **Gradient Boosting Classifier:** Test Accuracy = 85.89%, Test ROC-AUC = 0.6824


* **Decision Tree Classifier:** Test Accuracy = 91.92%, Test ROC-AUC = 0.8606


* **Random Forest Classifier (Best Model):** Fine-tuned via `RandomizedSearchCV` (`n_estimators=1000`, `max_features=7`, `max_depth=None`), yielding **Test Accuracy = 93.56%**, **Test Precision = 0.9706**, and **Test ROC-AUC = 0.8430** 🏆



---

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.8+


* **Data Processing:** Pandas, NumPy


* **Visualization:** Matplotlib, Seaborn, Plotly Express


* **Machine Learning & Pipeline:** Scikit-Learn (`LinearRegression`, `Lasso`, `Ridge`, `LogisticRegression`, `SVC`, `KNeighborsRegressor`, `DecisionTreeRegressor`, `DecisionTreeClassifier`, `RandomForestRegressor`, `RandomForestClassifier`, `AdaBoostRegressor`, `AdaBoostClassifier`, `GradientBoostingClassifier`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `LabelEncoder`, `GridSearchCV`, `RandomizedSearchCV`)



---

## 📁 Repository Structure

```text
├── data/
│   ├── 50_Startups.csv                         # Startups Expenditure & Profit Dataset
│   ├── Salary_Data.csv                         # Salary vs Experience Dataset
│   ├── Social_Network_Ads.csv                  # Social Network Ads Classification Dataset
│   ├── cardekho_imputated.csv                  # Cardekho Used Cars Price Dataset
│   └── Travel.csv                              # Holiday Package Purchase Prediction Dataset
├── Simple_And_Multiple_Linear_Regression.ipynb # Notebook for Regression Experiments
├── Social_Network_Ads_Classification.ipynb     # Notebook for Classification Experiments
├── Used_Car_Price_Prediction.ipynb             # Notebook for Car Price Regression (Random Forest & KNN)
├── Holiday_Package_Prediction.ipynb            # Notebook for Tourism Package Classification (Ensemble Methods)
└── README.md                                   # Project documentation

```
