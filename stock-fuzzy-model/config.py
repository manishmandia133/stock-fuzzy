"""Project-wide constants: ticker universe, dates, hyper-parameters."""
import numpy as np
import torch

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

US_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "BAC", "VZ", "ADBE",
    "CMCSA", "NFLX", "KO", "NKE", "MRK", "PFE", "T", "INTC", "CSCO", "XOM",
    "CVX", "WMT", "MCD", "ABT", "CRM", "ORCL", "IBM", "QCOM", "TXN", "HON",
    "AMD", "COST", "PEP", "ABBV", "TMO", "AVGO", "ACN", "LLY", "DHR", "NEE",
    "LIN", "UNP", "PM", "RTX", "LOW", "UPS", "SBUX", "BA", "GE", "CAT",
]

INDIA_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "NESTLEIND.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "JSWSTEEL.NS",
    "GRASIM.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "CIPLA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "BRITANNIA.NS",
]

TICKERS = US_TICKERS + INDIA_TICKERS

START_DATE = "2015-01-01"
END_DATE = None  # None -> up to the most recent available session

WINDOW = 60  # sliding window length (days)

FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume", "return_1d", "volatility_14d"]
TARGET_COL = "target_direction"

VOL_COL_IDX = FEATURE_COLS.index("volatility_14d")

BATCH_SIZE = 128
EPOCHS = 15
FIXED_LR = 1e-3
