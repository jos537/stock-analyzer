"""חישוב אינדיקטורים טכניים בפנדס טהור, ללא ספריות חיצוניות.

כל האינדיקטורים מסתכלים אחורה בלבד (rolling / ewm) כך שהערך בשורה t
אינו תלוי בנתונים עתידיים - חשוב הן לתצוגה והן כדי שה-ML לא "יראה את העתיד".
"""

import pandas as pd


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # החלקת Wilder (ewm עם alpha=1/period) - השיטה הסטנדרטית ל-RSI
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    if "EMA12" not in df.columns or "EMA26" not in df.columns:
        df = add_moving_averages(df)
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    sma = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    df["BB_MID"] = sma
    df["BB_UPPER"] = sma + num_std * std
    df["BB_LOWER"] = sma - num_std * std
    # מיקום המחיר בתוך הרצועה: 0 = רצועה תחתונה, 1 = רצועה עליונה
    band_width = df["BB_UPPER"] - df["BB_LOWER"]
    df["BB_PCTB"] = (df["Close"] - df["BB_LOWER"]) / band_width
    return df


def add_volume_trend(df: pd.DataFrame) -> pd.DataFrame:
    avg_volume_20 = df["Volume"].rolling(20).mean()
    df["VOL_RATIO"] = df["Volume"] / avg_volume_20
    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """מקבל DataFrame עם עמודות OHLCV ומחזיר אותו עם כל האינדיקטורים מחושבים."""
    df = df.copy()
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_trend(df)
    return df
