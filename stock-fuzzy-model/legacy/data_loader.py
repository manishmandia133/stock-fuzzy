"""Data extraction: batched yfinance download (one HTTP round-trip per chunk)."""
import yfinance as yf

from config import WINDOW


def batch_download(tickers, start, end, chunk_size=20):
    """Download OHLCV for many tickers efficiently, in chunks.

    Returns:
        dict {ticker: DataFrame[Open, High, Low, Close, Volume]}
    """
    frames = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        print(f"Downloading {i + 1}-{i + len(chunk)} / {len(tickers)} ...")
        raw = yf.download(
            tickers=chunk, start=start, end=end, group_by="ticker",
            auto_adjust=True, threads=True, progress=False,
        )
        for t in chunk:
            try:
                if len(chunk) == 1:
                    df = raw.copy()
                else:
                    df = raw[t].copy()
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
                if len(df) > (WINDOW + 30):  # need enough history to be usable
                    frames[t] = df
                else:
                    print(f"  skipping {t}: insufficient history ({len(df)} rows)")
            except Exception as e:
                print(f"  skipping {t}: {e}")
    return frames
