# 🚗 Rusty Bargain – Used Car Market Value Prediction

## 📖 Project Overview

This project focuses on predicting the **market value of used vehicles** based on their technical specifications and historical data.  
The goal was to assist a used-car marketplace in developing a pricing model capable of estimating fair prices for listings, improving transparency and buyer trust.

The dataset includes information such as brand, model, year of manufacture, mileage, fuel type, engine power, and other vehicle characteristics.  
Several regression models were trained, evaluated, and compared to determine which provided the best predictive accuracy.

---

## 🎯 Objectives

- Predict the **market price** of used cars based on technical and categorical features.
- Compare and evaluate the performance of different **machine learning regression algorithms**.
- Select the model that minimizes the prediction error (RMSE) and maximizes the explained variance (R²).
- Analyze the **influence of key features** on car price determination.

---

## 🧠 Machine Learning Workflow

### 1. **Data Preprocessing**

- Handled missing values and outliers.
- Encoded categorical features using **Label Encoding** and **One-Hot Encoding**.
- Scaled numerical variables with **StandardScaler**.
- Split data into training and testing sets (80/20).

### 2. **Exploratory Data Analysis (EDA)**

- Distribution of car prices and feature correlations.
- Detection of skewness and log transformation for better feature scaling.
- Visualization of variable importance through **correlation heatmaps** and **pairplots**.

### 3. **Modeling**

The following **regression models** were implemented and compared:

- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**
- **CatBoost Regressor**
- **LightGBM Regressor**

All models were evaluated using cross-validation and fine-tuned through **GridSearchCV**.

### 4. **Model Evaluation**

The models were compared using the following metrics:

- **R² (Coefficient of Determination)**
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**

| Model             |    R²    |    RMSE     |     MAE     |
| :---------------- | :------: | :---------: | :---------: |
| Linear Regression |   0.74   |   2050.32   |   1480.25   |
| Random Forest     |   0.86   |   1623.18   |   1135.52   |
| XGBoost           |   0.87   |   1587.63   |   1089.40   |
| CatBoost          |   0.88   |   1579.49   |   1075.68   |
| LightGBM          | **0.88** | **1579.49** | **1075.68** |

> 💡 **LightGBM Regressor** achieved the best performance with an R² of 0.88 and RMSE of 1579.49,  
> representing an average error of around 30% relative to the average car price — a strong result for real-world data variability.

---

## 📊 Feature Importance

Feature importance analysis using LightGBM revealed that:

- **Year of manufacture**, **engine power**, and **mileage** were the most influential predictors.
- Categorical features such as **brand** and **fuel type** also contributed significantly.

---

## 🧩 Tech Stack

| Category                | Tools                                                            |
| :---------------------- | :--------------------------------------------------------------- |
| **Language**            | Python 3.10                                                      |
| **Data Processing**     | Pandas, NumPy                                                    |
| **Visualization**       | Matplotlib, Seaborn                                              |
| **Modeling**            | Scikit-learn, LightGBM, XGBoost, CatBoost, RandomForestRegressor |
| **Feature Engineering** | LabelEncoder, OneHotEncoder, StandardScaler                      |
| **Optimization**        | GridSearchCV                                                     |
| **Evaluation**          | R², RMSE, MAE                                                    |
| **Development Tools**   | Jupyter Notebook, VS Code, Git & GitHub                          |

---

## 🧪 Key Insights

- Ensemble tree-based models (LightGBM, XGBoost, CatBoost) significantly outperformed linear regression.
- Price prediction accuracy improves notably with proper feature encoding and hyperparameter tuning.
- The model demonstrates robustness against overfitting while maintaining interpretability.

---
