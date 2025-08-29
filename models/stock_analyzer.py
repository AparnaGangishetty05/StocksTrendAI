import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import config


class StockAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.period = config.YFINANCE_PERIOD
        self.interval = config.YFINANCE_INTERVAL

    def fetch_data(self):
        df = yf.download(self.symbol, period=self.period,
                         interval=self.interval, progress=False)
        if df.empty:
            raise ValueError("Invalid stock symbol or no data found.")
        return df

    def days_until_december(self):
        today = pd.Timestamp.today().normalize()
        year_end = pd.Timestamp(year=today.year, month=12, day=31)
        return len(pd.bdate_range(today, year_end))

    def iterative_forecast(self, df, model, n_days):
        df_future = df.copy()
        last_window = df_future["Close"].values[-30:]
        scaler = StandardScaler()
        last_window = scaler.fit_transform(
            last_window.reshape(-1, 1)).flatten()

        preds = []
        for _ in range(n_days):
            X_pred = np.array(last_window[-30:]).reshape(1, -1)
            pred = model.predict(X_pred)[0]
            preds.append(pred)
            last_window = np.append(last_window, pred)

        future_dates = pd.bdate_range(
            df.index[-1], periods=n_days+1, freq="B")[1:]
        df_future = pd.DataFrame({"Date": future_dates, "Close": preds})
        df_future.set_index("Date", inplace=True)
        return df_future

    def analyze(self):
        df = self.fetch_data()
        df["Return"] = df["Close"].pct_change()

        X, y = [], []
        for i in range(30, len(df)):
            X.append(df["Close"].values[i-30:i])
            y.append(df["Close"].values[i])
        X, y = np.array(X), np.array(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(
            n_estimators=config.N_ESTIMATORS, random_state=42)
        model.fit(X_scaled, y)

        forecast_days = self.days_until_december()
        df_future = self.iterative_forecast(df, model, forecast_days)

        latest_price = df["Close"].iloc[-1]
        final_pred_price = df_future["Close"].iloc[-1]
        exp_return = (final_pred_price - latest_price) / latest_price

        if exp_return > 0.05:
            signal = "Buy"
        elif exp_return < -0.05:
            signal = "Sell"
        else:
            signal = "Hold"

        # Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], mode="lines", name="Historical"))
        fig.add_trace(go.Scatter(x=df_future.index,
                      y=df_future["Close"], mode="lines", name="Forecast"))
        chart_html = fig.to_html(full_html=False)

        return {
            "symbol": self.symbol,
            "latest_price": round(latest_price, 2),
            "final_pred_price": round(final_pred_price, 2),
            "expected_return": round(exp_return*100, 2),
            "signal": signal,
            "chart": chart_html
        }
