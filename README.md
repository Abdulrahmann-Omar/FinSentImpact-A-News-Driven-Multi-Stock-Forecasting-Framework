# FinSentImpact-A-News-Driven-Multi-Stock-Forecasting-Framework
FinSentImpact is a data-driven forecasting framework that integrates historical stock price data and financial news indicators to analyze and predict next-day stock movements across multiple companies. The project leverages advanced feature engineering, time series visualization, exploratory data analysis, and class imbalance handling to uncover the correlation between news sentiment and price volatility. Using predictive modeling techniques (such as Random Forest and SHAP interpretability), the project aims to develop explainable models that can forecast future stock performance (Open, Close, High, Low, Volume, and Adjusted Close) by learning from both market behavior and news triggers.

## Key components include:

  Automated data pipeline from 47 publicly traded stocks
  
  Feature shifts to model next-day predictions
  
  News event tagging and impact visualization
  
  Rolling statistics, KDE, outlier detection, and PCA clustering
  
  SMOTE-based balancing for news event classification
  
  Model-ready dataset with engineered candle shape and volatility features
