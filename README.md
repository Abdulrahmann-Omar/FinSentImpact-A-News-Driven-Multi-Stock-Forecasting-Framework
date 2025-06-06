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

## Challenges and Solutions in Data Handling

Integrating and preparing the FNSPID dataset presented several challenges. Below, we outline these hurdles and the robust solutions we implemented to ensure our forecasting models are accurate and reliable.

### 1. Data Integration

**Challenge**: The FNSPID dataset separates stock price data (in multiple CSV files within full_history.zip) and news sentiment data (in nasdaq_exteral_data.csv). Concatenating these files directly would misalign data due to differing structures and frequencies, leading to inaccurate predictions.

**Solution**: We merged the stock price and news data using Pandas’ merge function, joining on the 'date' and 'ticker' columns. A left join ensured all stock price records were retained, even for days without news. This approach aligned each stock’s price data with its corresponding sentiment scores.


### 2. Defining the Target Variable

**Challenge**: Setting "tomorrow" as the target (e.g., next day’s closing price) required identifying the next trading day, as stock markets are closed on weekends and holidays. Incorrectly using calendar days could introduce errors in the time series.

**Solution**: We used the pandas_market_calendars library to filter data to valid NYSE trading days and set the target as the next trading day’s closing price, grouped by ticker to maintain stock-specific sequences.


### 3. Feature Engineering

**Challenge**: Creating meaningful features from raw data while avoiding look-ahead bias was critical. Using future data in features (e.g., tomorrow’s sentiment) would make predictions unrealistic.

**Solution**: We engineered features like high-to-low ratio, close-to-open ratio, and aggregated sentiment scores per date and ticker, using only past and present data. Sentiment scores were averaged for days with multiple news articles to capture overall sentiment.


### 4. Handling Missing Values

**Challenge**: Missing sentiment scores for days without news could disrupt model training if not addressed properly.

**Solution**: We applied forward-filling to impute missing sentiment scores, assuming sentiment persists until new information is available. For other missing values, we evaluated context-specific imputation or removed incomplete records to maintain data quality.



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

