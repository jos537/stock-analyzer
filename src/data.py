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


@st.cache_data(ttl=86400)  # יום - הסקטור של חברה כמעט לא משתנה
def get_ticker_sector(ticker: str) -> str | None:
    """שם הסקטור (GICS) של הטיקר, לשימוש במיפוי ל-ETF סקטוריאלי (config.SECTOR_ETF_MAP).
    מחזיר None אם לא ידוע - לא מנחשים."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("sector")
    except Exception:
        return None


@st.cache_data(ttl=86400)  # יום - תאריך דוח לא משתנה תכוף
def get_next_earnings_date(ticker: str) -> pd.Timestamp | None:
    """מועד הדוח הרבעוני הקרוב, אם ידוע. משתמש ב-get_earnings_dates ולא ב-
    Ticker.calendar, שידוע כלא עקבי בין גרסאות yfinance. מחזיר None אם לא נמצא -
    זה *לא* אומר "אין סיכון דוח", רק ש"לא ידוע" - חשוב להבדיל בתצוגה."""
    try:
        dates = yf.Ticker(ticker).get_earnings_dates(limit=8)
        if dates is None or dates.empty:
            return None
        now = pd.Timestamp.now(tz=dates.index.tz)
        future = dates[dates.index >= now]
        return future.index.min() if not future.empty else None
    except Exception:
        return None
