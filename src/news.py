"""חדשות חברה חיות דרך Finnhub, עם caching. לעולם לא זורק חריגה - אם אין
מפתח API או שהקריאה נכשלת, מחזיר רשימה ריקה והעמוד מציג הודעה ידידותית.

תיוג ה-sentiment הוא heuristic מקומי (מילות מפתח) - לא בינה מלאכותית,
לא תחליף לניתוח אמיתי. הטיר החינמי של Finnhub לא כולל sentiment מובנה.
"""

from datetime import datetime, timedelta

import requests
import streamlit as st

from src.config import NEWS_LOOKBACK_DAYS

NEWS_API_BASE = "https://finnhub.io/api/v1"

_POSITIVE_WORDS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies",
    "upgrade", "upgraded", "record", "growth", "profit", "gain", "gains",
    "strong", "outperform", "buy", "bullish", "expansion", "partnership",
}
_NEGATIVE_WORDS = {
    "miss", "misses", "plunge", "plunges", "drop", "drops", "downgrade",
    "downgraded", "loss", "losses", "weak", "warning", "lawsuit", "probe",
    "investigation", "recall", "layoff", "layoffs", "bearish", "sell",
    "cut", "cuts", "decline", "declines", "concern", "concerns",
}


def tag_headline_sentiment(headline: str) -> str:
    """🟢/🔴/⚪ לפי ספירת מילות מפתח חיוביות/שליליות באנגלית. heuristic גס
    בלבד - מוצג ב-UI ככזה, לא כניתוח אמיתי."""
    words = set(headline.lower().replace(",", " ").replace(".", " ").split())
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    if pos > neg:
        return "🟢"
    if neg > pos:
        return "🔴"
    return "⚪"


@st.cache_data(ttl=1800)  # חצי שעה - חדשות מתעדכנות, אבל אין צורך ברענון קבוע
def get_company_news(ticker: str, days_back: int = NEWS_LOOKBACK_DAYS) -> list[dict]:
    """מחזיר [] אם אין מפתח API, הקריאה נכשלה, או שאין תוצאות - לעולם לא
    זורק. כל פריט: {headline, source, url, datetime (datetime|None), summary}."""
    try:
        api_key = st.secrets.get("FINNHUB_API_KEY")
    except Exception:
        # קורה אם אין קובץ secrets.toml בכלל (למשל: פריסה טרייה בלי ה-Secret
        # עדיין מוגדר) - זה מצב תקין, לא שגיאה. פשוט אין חדשות להציג.
        api_key = None
    if not api_key:
        return []

    today = datetime.now().date()
    start = today - timedelta(days=days_back)
    try:
        resp = requests.get(
            f"{NEWS_API_BASE}/company-news",
            params={
                "symbol": ticker,
                "from": start.isoformat(),
                "to": today.isoformat(),
                "token": api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        raw_items = resp.json()
    except Exception:
        return []

    if not isinstance(raw_items, list):
        return []

    items = []
    for item in raw_items:
        ts = item.get("datetime")
        items.append({
            "headline": item.get("headline", ""),
            "source": item.get("source", "לא ידוע"),
            "url": item.get("url", ""),
            "datetime": datetime.fromtimestamp(ts) if ts else None,
            "summary": item.get("summary", ""),
        })
    items.sort(key=lambda x: x["datetime"] or datetime.min, reverse=True)
    return items
