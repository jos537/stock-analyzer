"""שליפת נתוני מניות מ-yfinance, עם caching וטיפול בשגיאות לכל מניה בנפרד."""

import pandas as pd
import streamlit as st
import yfinance as yf

from src.config import HISTORY_PERIOD


@st.cache_data(ttl=3600)  # שעה - מספיק לנתוני סוף-יום, לא מכביד על yfinance
def get_summary_data(tickers: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """שולף היסטוריה קצרה (6 חודשים) לכל טיקר, לשימוש בעמוד הבית.

    מחזיר dict בין טיקר ל-DataFrame. טיקר שנכשל פשוט לא מופיע ב-dict,
    כדי שמניה בעייתית אחת לא תפיל את כל טבלת הסיכום.
    """
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period="6mo", interval="1d")
            if not hist.empty:
                result[ticker] = hist
        except Exception:
            continue
    return result


@st.cache_data(ttl=86400)  # יום - היסטוריה ארוכה, לא צריך רענון תכוף
def get_history(ticker: str, period: str = HISTORY_PERIOD) -> pd.DataFrame | None:
    """שולף היסטוריה מלאה למניה בודדת (לעמוד הפירוט ולאימון המודל)."""
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d")
        if hist.empty:
            return None
        return hist
    except Exception:
        return None
