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

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

FEATURE_COLUMNS = [
    "RET_1D", "RET_5D", "RET_10D", "RET_20D",
    "RSI14",
    "MACD", "MACD_SIGNAL", "MACD_HIST",
    "PRICE_TO_SMA20", "PRICE_TO_SMA50", "PRICE_TO_SMA200",
    "BB_PCTB",
    "VOLATILITY_20D",
    "VOL_RATIO",
]


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """מוסיף את עמודות ה-features הנגזרות. מניח שהאינדיקטורים הבסיסיים
    (SMA/RSI/MACD/Bollinger/Volume) כבר חושבו על ידי indicators.py."""
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


def build_labels(df: pd.DataFrame, horizon_days: int) -> pd.Series:
    """1 אם המחיר בעוד horizon_days ימי מסחר גבוה מהיום, אחרת 0.
    השורות האחרונות (שאין להן label עתידי) יהיו NaN - יש להשמיט אותן."""
    future_close = df["Close"].shift(-horizon_days)
    return (future_close > df["Close"]).astype("float")


_model_cache = None
_metadata_cache = None


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


def predict_proba_up(df_with_indicators: pd.DataFrame) -> float | None:
    """מקבל DataFrame עם אינדיקטורים מחושבים (indicators.py) ומחזיר
    הסתברות (0-1) שהמחיר יעלה תוך horizon ימי מסחר, לפי השורה האחרונה.
    מחזיר None אם אין מספיק נתונים (למשל מניה חדשה בלי היסטוריית 200 יום)."""
    model, _ = _load()
    df = add_ml_features(df_with_indicators)
    last_row = df.iloc[-1]
    features = last_row[FEATURE_COLUMNS]
    if features.isna().any():
        return None
    x = pd.DataFrame([features.to_dict()], columns=FEATURE_COLUMNS)
    proba = model.predict_proba(x)[0][1]
    return float(proba)
