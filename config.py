import os

SECRET_KEY = "supersecretkey"
SQLALCHEMY_DATABASE_URI = "sqlite:///stocktrendai.db"
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Yahoo Finance parameters
YFINANCE_PERIOD = "1y"
YFINANCE_INTERVAL = "1d"

# ML Model parameters
N_ESTIMATORS = 100
