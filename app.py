from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import yfinance as yf
import pandas as pd
import ta
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import numpy as np

app = Flask(__name__)
app.secret_key = "secret123"

# DB setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ---------------- Models ----------------

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------- Routes ----------------

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials!", "danger")
    return render_template("login.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        if User.query.filter_by(email=email).first():
            flash("Email already registered!", "warning")
            return redirect(url_for("register"))
        new_user = User(email=email, password=generate_password_hash(password, method="pbkdf2:sha256"))
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))

# ---------------- Dashboard with Stocks ----------------

@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    stock = "AAPL"
    period = "6mo"
    latest_price = None
    signal = None
    confidence = None
    graph_html = None
    ma_chart = None
    rsi_chart = None
    macd_chart = None
    forecast_chart = None

    if request.method == "POST":
        stock = request.form.get("stock", "AAPL").upper()
        period = request.form.get("period", "6mo")

    try:
        hist = yf.download(stock, period=period, auto_adjust=True)
        hist.columns = [col[0] if isinstance(col, tuple) else col for col in hist.columns]

        if not hist.empty:
            last_close = float(hist["Close"].iloc[-1].item())
            mean_price = float(hist["Close"].mean().item())

            latest_price = round(last_close, 2)
            signal = "BUY" if last_close > mean_price else "SELL"
            confidence = 75

            # Main Price Chart
            fig = px.line(hist, x=hist.index, y="Close", title=f"{stock} Stock Price ({period})")
            fig.update_layout(template="plotly_dark")
            graph_html = fig.to_html(full_html=False)

            # Technical Indicators
            hist['SMA_20'] = ta.trend.sma_indicator(hist['Close'], window=20)
            hist['SMA_50'] = ta.trend.sma_indicator(hist['Close'], window=50)
            hist['RSI'] = ta.momentum.rsi(hist['Close'], window=14)

            macd_indicator = ta.trend.MACD(close=hist['Close'])
            hist['MACD'] = macd_indicator.macd()
            hist['MACD_signal'] = macd_indicator.macd_signal()

            # Moving Averages Chart
            ma_fig = px.line(hist, x=hist.index, y=['Close', 'SMA_20', 'SMA_50'], title=f"{stock} - Moving Averages")
            ma_fig.update_layout(template="plotly_dark")
            ma_chart = ma_fig.to_html(full_html=False)

            # RSI Chart
            rsi_fig = px.line(hist, x=hist.index, y='RSI', title=f"{stock} - RSI")
            rsi_fig.update_layout(template="plotly_dark", yaxis_range=[0, 100])
            rsi_chart = rsi_fig.to_html(full_html=False)

            # MACD Chart
            macd_fig = px.line(hist, x=hist.index, y=['MACD', 'MACD_signal'], title=f"{stock} - MACD")
            macd_fig.update_layout(template="plotly_dark")
            macd_chart = macd_fig.to_html(full_html=False)

            # Forecasting with UHRS-style randomness
            hist['Date'] = hist.index
            hist = hist.reset_index(drop=True)
            hist.dropna(inplace=True)

            X = hist[['SMA_20', 'SMA_50', 'RSI']]
            y = hist['Close']

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            future_dates = pd.date_range(start=hist['Date'].iloc[-1], periods=126, freq='B')[1:]
            last_row = hist.iloc[-1]

            np.random.seed(42)
            future_data = pd.DataFrame({
                'SMA_20': [last_row['SMA_20'] + np.random.normal(0, 0.5) for _ in future_dates],
                'SMA_50': [last_row['SMA_50'] + np.random.normal(0, 0.5) for _ in future_dates],
                'RSI': [last_row['RSI'] + np.random.normal(0, 1.0) for _ in future_dates]
            }, index=future_dates)

            future_preds = model.predict(future_data)
            forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_preds})
            forecast_fig = px.line(forecast_df, x='Date', y='Predicted Close', title=f"{stock} - 6 Month Forecast")
            forecast_fig.update_layout(template="plotly_dark")
            forecast_chart = forecast_fig.to_html(full_html=False)

    except Exception as e:
        graph_html = f"<p class='text-danger'>Error fetching stock data: {e}</p>"

    return render_template(
        "dashboard.html",
        stock=stock,
        latest_price=latest_price,
        signal=signal,
        confidence=confidence,
        graph_html=graph_html,
        ma_chart=ma_chart,
        rsi_chart=rsi_chart,
        macd_chart=macd_chart,
        forecast_chart=forecast_chart,
        period=period
    )

if __name__ == "__main__":
    app.run(debug=True)
