"""הנדסת features, טעינת המודל המאומן, וחיזוי הסתברות עלייה.

המודל עצמו מאומן *מחוץ* לאפליקציה (ראה scripts/train_model.py) ונטען כאן
כקובץ סטטי. חשוב: כל הפיצ'רים מחושבים מנתונים עד יום t (כולל) בלבד - שום
פיצ'ר לא "רואה" את העתיד.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MARKET_TICKER

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

STOCK_FEATURE_COLUMNS = [
    "RET_1D", "RET_5D", "RET_10D", "RET_20D",
    "RSI14",
    "MACD", "MACD_SIGNAL", "MACD_HIST",
    "PRICE_TO_SMA20", "PRICE_TO_SMA50", "PRICE_TO_SMA200",
    "BB_PCTB",
    "VOLATILITY_20D",
    "VOL_RATIO",
]

# features שמתארים את הקשר השוק הרחב (S&P 500), לא רק את המניה הבודדת
MARKET_FEATURE_COLUMNS = ["MKT_RET_5D", "MKT_RET_20D", "MKT_ABOVE_SMA50"]

FEATURE_COLUMNS = STOCK_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """מוסיף את עמודות ה-features הנגזרות של המניה עצמה. מניח שהאינדיקטורים
    הבסיסיים (SMA/RSI/MACD/Bollinger/Volume) כבר חושבו על ידי indicators.py."""
    df = df.copy()
    df["RET_1D"] = df["Close"].pct_change(1)
    df["RET_5D"] = df["Close"].pct_change(5)
    df["RET_10D"] = df["Close"].pct_change(10)
    df["RET_20D"] = df["Close"].pct_change(20)
    df["PRICE_TO_SMA20"] = df["Close"] / df["SMA20"] - 1
    df["PRICE_TO_SMA50"] = df["Close"] / df["SMA50"] - 1
    df["PRICE_TO_SMA200"] = df["Close"] / df["SMA200"] - 1
    df["VOLATILITY_20D"] = df["Close"].pct_change().rolling(20).std()
    return df


def build_market_features(market_df_with_indicators: pd.DataFrame) -> pd.DataFrame:
    """מקבל DataFrame של מדד השוק (למשל SPY) עם אינדיקטורים כבר מחושבים,
    ומחזיר רק את עמודות ה-features של הקשר השוק, לשימוש כ-join למניות בודדות."""
    m = market_df_with_indicators
    out = pd.DataFrame(index=m.index)
    out["MKT_RET_5D"] = m["Close"].pct_change(5)
    out["MKT_RET_20D"] = m["Close"].pct_change(20)
    out["MKT_ABOVE_SMA50"] = (m["Close"] > m["SMA50"]).astype(float)
    return out


def add_market_features(df: pd.DataFrame, market_features: pd.DataFrame | None) -> pd.DataFrame:
    """מצרף את features הקשר-השוק ל-DataFrame של מניה בודדת, לפי תאריך.
    אם אין נתוני שוק זמינים, העמודות מתמלאות ב-NaN (השורה תושמט בהמשך)."""
    df = df.copy()
    if market_features is None:
        for col in MARKET_FEATURE_COLUMNS:
            df[col] = np.nan
        return df
    df = df.join(market_features, how="left")
    df[MARKET_FEATURE_COLUMNS] = df[MARKET_FEATURE_COLUMNS].ffill()
    return df


def build_labels(df: pd.DataFrame, horizon_days: int, move_threshold: float = 0.0) -> pd.Series:
    """1 אם המחיר עלה מעל move_threshold תוך horizon_days, 0 אם ירד מתחתיו.
    תזוזות קטנות מ-move_threshold (רעש) ושורות בלי label עתידי מקבלות NaN
    ומושמטות מהאימון - זה מפחית תוויות שרירותיות סביב אפס."""
    future_ret = df["Close"].shift(-horizon_days) / df["Close"] - 1
    labels = pd.Series(np.nan, index=df.index)
    labels[future_ret > move_threshold] = 1.0
    labels[future_ret < -move_threshold] = 0.0
    return labels


_model_cache = None
_metadata_cache = None
_market_features_cache = None


def _load():
    global _model_cache, _metadata_cache
    if _model_cache is None:
        _model_cache = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata_cache = json.load(f)
    return _model_cache, _metadata_cache


def model_is_available() -> bool:
    return MODEL_PATH.exists() and METADATA_PATH.exists()


def get_metadata() -> dict:
    _, metadata = _load()
    return metadata


def _get_live_market_features() -> pd.DataFrame | None:
    """שולף (עם caching) את היסטוריית מדד השוק ובונה ממנה features, לשימוש
    בזמן חיזוי חי באפליקציה. ייבוא מקומי כדי למנוע import מעגלי עם data.py."""
    global _market_features_cache
    if _market_features_cache is not None:
        return _market_features_cache
    from src.data import get_history
    from src.indicators import compute_all_indicators

    spy = get_history(MARKET_TICKER)
    if spy is None:
        return None
    spy = compute_all_indicators(spy)
    _market_features_cache = build_market_features(spy)
    return _market_features_cache


def predict_proba_up(df_with_indicators: pd.DataFrame) -> float | None:
    """מקבל DataFrame עם אינדיקטורים מחושבים (indicators.py) ומחזיר
    הסתברות (0-1) שהמחיר יעלה תוך horizon ימי מסחר, לפי השורה האחרונה.
    מחזיר None אם אין מספיק נתונים (למשל מניה חדשה בלי היסטוריית 200 יום)."""
    model, _ = _load()
    df = add_ml_features(df_with_indicators)
    df = add_market_features(df, _get_live_market_features())
    last_row = df.iloc[-1]
    features = last_row[FEATURE_COLUMNS]
    if features.isna().any():
        return None
    x = pd.DataFrame([features.to_dict()], columns=FEATURE_COLUMNS)
    proba = model.predict_proba(x)[0][1]
    return float(proba)
