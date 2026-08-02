
# Machine Learning Supervised Learning Benchmarks

This repository contains practical implementations and benchmark experiments for foundational Supervised Learning algorithms, including **Linear & Multiple Regression** and **Binary Classification (Logistic Regression vs. Support Vector Machines)**.

---

## 📌 Project 1: Medical Insurance Cost Prediction (Regression)

* **Objective:** Predict individual medical insurance charges (`charges`) based on physical, demographic, and lifestyle features (`age`, `bmi`, `children`, `sex`, `smoker`, `region`).
* **Data Preprocessing & Cleaning:** 
  * Handled target skewness and target outliers in `y_train` using the **Interquartile Range (IQR)** method (`Q1 - 1.5*IQR` to `Q3 + 1.5*IQR`).
  * Scaled continuous numerical features using `StandardScaler` fitted strictly on training data.
  * Encoded categorical features (`sex`, `smoker`, `region`) using **One-Hot Encoding** (`pd.get_dummies(drop_first=True)`).
* **Model Iterations & Evaluation:**
  * **Simple Linear Regression (`age` only):** Evaluated baseline linear relationship between age and charges.
  * **Multiple Linear Regression (`age`, `bmi`, `children`):** Expanded feature set to capture additional physical metrics.
  * **Full Multiple Linear Regression (All Features Included):** Combined scaled numerical metrics with one-hot encoded categorical variables to evaluate overall model performance using Mean Squared Error (MSE) and $R^2$ score.

---

## 📌 Project 2: Social Network Ads (Binary Classification)

* **Objective:** Predict user purchase intent (`Purchased`: 0 or 1) based on user demographic features (`Age` and `EstimatedSalary`).
* **Preprocessing:** Standardized feature distributions using `StandardScaler` for distance-based classification algorithms.
* **Hyperparameter Optimization:** Utilized `GridSearchCV` with 5-fold cross-validation to fine-tune inverse regularization (`C`), penalty norms, and solvers, improving Logistic Regression accuracy from **86.00% to 88.75%**.
* **Model Progression & Benchmarking:**
  * **Logistic Regression (Baseline):** 86.00% Test Accuracy
  * **Linear SVM:** 86.25% Test Accuracy
  * **Logistic Regression (GridSearchCV Tuned):** 88.75% Test Accuracy
  * **RBF Kernel SVM (Best Model):** **92.50% Test Accuracy** 🏆
* **Key Finding:** Transitioning from linear models to an **RBF Kernel SVM** yielded a **+6.5% performance boost**, proving that non-linear decision boundaries are required to separate user age and salary interaction clusters.

---

📌 Project 3: Regression Benchmarks (Simple & Multiple Linear Regression)

* **Datasets Used:** `Salary_Data.csv` (Simple) and `50_Startups.csv` (Multiple).
* **Objectives:** 
  * Predict salary based on years of experience (1D Linear Regression).
  * Predict startup `Profit` based on operational spending (`R&D Spend`, `Administration`, `Marketing Spend`) and `State` location.
* **Preprocessing & Feature Engineering:**
  * Applied `StandardScaler` on numerical features after splitting data to prevent data leakage.
  * Encoded categorical location (`State`) via One-Hot Encoding (`pd.get_dummies(drop_first=True)`), preventing the Dummy Variable Trap.
  * Synchronized feature columns between train and test splits using `df.align(join='left', axis=1, fill_value=0)`.
* **Model Evaluation & Key Findings:**
  * **Simple Linear Regression:** Established baseline $y = mx + c$ relationship between experience and income.
  * **Multiple Linear Regression:** Evaluated partial regression coefficients (`model.coef_`) to rank spending impact (`R&D Spend` emerged as the primary profit driver).
  * **Low-Correlation Feature Impact:** Bivariate correlation showed `Administration` had only ~9% linear correlation with `Profit`, but model weights demonstrated its role as a control variable alongside R&D and Marketing spending.

---
## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Data Manipulation & EDA:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (`LinearRegression`, `LogisticRegression`, `SVC`, `StandardScaler`, `GridSearchCV`, `mean_squared_error`, `r2_score`,`confusion matrix`,`classification report`)

## 📁 Repository Structure
```text
├── 50_Startups.csv                         # Startups Expenditure & Profit Dataset
├── Salary_Data.csv                         # Salary vs Experience Dataset
├── Social_Network_Ads.csv                  # Social Network Ads Classification Dataset
├── Simple_And_Multiple_Linear_Regression.ipynb # Notebook for Regression Experiments
├── Social_Network_Ads_Classification.ipynb # Notebook for Classification Experiments
└── README.md                               # Project documentation















