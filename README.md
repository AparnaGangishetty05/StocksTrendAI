# StocksTrendAI
The Stock Prediction Dashboard is a web-based application that provides real-time stock market analysis and future price predictions using machine learning. It combines The Stock Prediction Dashboard is a web-based application that provides real-time stock market analysis and future price predictions using machine learning.
It combines financial data visualization, technical analysis indicators, and ML forecasting models to help investors make data-driven decisions.

 Features
 User Authentication – Secure login for personalized dashboards
 Real-Time Stock Data – Fetch live data via Yahoo Finance API (yfinance)
 Technical Indicators – SMA, RSI, and MACD visualizations
 ML Forecasting – Stock price prediction using Random Forest Regressor
Interactive Charts – Historical prices, moving averages, RSI, MACD, and 6-month forecast

 Tech Stack
Frontend/Backend: Streamlit (Python)
Data Processing: Pandas, NumPy
Machine Learning: Scikit-learn
Visualization: Matplotlib, Seaborn
API: YFinance
Docs: python-docx

How to Run
Clone the repo:
git clone https://github.com/your-username/stock-prediction-dashboard.git
cd stock-prediction-dashboard

Install dependencies:
pip install -r requirements.txt
Run the app:
streamlit run app.py
Login and select a stock ticker to start analysis.

Future Enhancements
Support for multiple ML models with comparison
Sentiment analysis from financial news
Portfolio tracking and price alerts
More interactive charting (e.g., Plotly, D3.js)
