# FinSentImpact: A News-Driven Multi-Stock Forecasting Framework

## 📊 Overview

**FinSentImpact** is a multi-model machine learning framework designed to forecast stock prices for multiple companies by integrating historical financial data with sentiment analysis from financial news. This project aims to provide more accurate stock predictions by capturing both numerical trends and qualitative market signals.

The framework supports:

* Multi-output regression (predicting multiple future stock metrics)
* Integration of news sentiment scores
* Multiple model implementations ( XGBoost, Random Forest, SVR )
* Model explainability via SHAP, PDP, ICE, and LIME

---

## 🧠 Team Members and Assigned Models

| Name                 | Model Implemented | 
| -------------------- | ----------------- | 
| Abdulrahman Omar     | XGBoost Regressor | 
| Abdulrahman Elattar  | Random Forest     | 
| Retal Ali            | SVM Regressor     | 

> ℹ️ Each notebook is self-contained and includes full model implementation, evaluation, and 4+ explainability techniques.


---

## 🧪 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Abdulrahmann-Omar/FinSentImpact-A-News-Driven-Multi-Stock-Forecasting-Framework.git
cd FinSentImpact-A-News-Driven-Multi-Stock-Forecasting-Framework
```

### 2. Create Virtual Environment or you can use Colab

```bash
python3 -m venv venv
source venv\Scripts\activate  
```

### 3. Open Notebooks

Launch Jupyter or Colab and open any notebook you need.

---

## 🧠 Machine Learning Models

* 📈 **XGBoost**: Gradient boosting with SHAP & PDP for explainability.
* 🌲 **Random Forest**: Ensemble model for robustness.
* 📉 **Support Vector Regressor (SVR)**: For nonlinear regression using RBF kernel.

Each model targets forecasting the following outputs:

* `Volume_tomorrow`, `Open_tomorrow`, `High_tomorrow`
* `Low_tomorrow`, `Close_tomorrow`, `Adj close_tomorrow`

---

## 🔍 Explainability Techniques

| Technique | Description                                      |
| --------- | ------------------------------------------------ |
| SHAP      | Global and local feature importance              |
| LIME      | Local surrogate models for instance explanations |
| PDP       | Shows marginal effect of a feature               |
| ICE       | Displays variation across individual predictions |

Each notebook includes at least four interpretability techniques as required.

---

## 📊 Evaluation Metrics

* **RMSE**: Root Mean Squared Error for prediction accuracy
* **R² Score**: Goodness of fit metric
* **Visualizations**: SHAP summary plots, ICE plots, and feature importance bar charts

---

## ⚙️ Feature Engineering

* Ratios: `high_to_low`, `close_to_open`, `volume_per_price`
* Sentiment: News sentiment scores aggregated per date
* Log-transformed volumes for normalization
* Scaled features using `StandardScaler`

