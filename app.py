"""עמוד הבית: רשימת מעקב אישית וכרטיסי סיכום לכל מניה."""

import streamlit as st

from src.config import DEFAULT_WATCHLIST
from src.data import get_summary_data
from src.indicators import compute_all_indicators
from src.ml_model import model_is_available, predict_proba_up
from src.signals import BEARISH, BULLISH, evaluate

st.set_page_config(page_title="ניתוח מניות", page_icon="📊", layout="centered")

st.warning(
    "⚠️ **כלי חינוכי בלבד - אינו ייעוץ השקעות.** "
    "הנתונים וההערכות מבוססים על היסטוריה בלבד ואינם מבטיחים תוצאות עתידיות.",
    icon="⚠️",
)

st.title("📊 ניתוח מניות")

if "watchlist" not in st.session_state:
    st.session_state.watchlist = list(DEFAULT_WATCHLIST)

with st.sidebar:
    st.header("רשימת מעקב")
    new_ticker = st.text_input("הוסף טיקר (למשל AAPL)", key="new_ticker_input").strip().upper()
    if st.button("➕ הוסף") and new_ticker:
        if new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)
        st.rerun()

    st.divider()
    for t in list(st.session_state.watchlist):
        col1, col2 = st.columns([3, 1])
        col1.write(t)
        if col2.button("🗑️", key=f"remove_{t}"):
            st.session_state.watchlist.remove(t)
            st.rerun()

    st.caption("רשימת המעקב נשמרת רק בדפדפן הנוכחי, לא בין מכשירים.")

if not model_is_available():
    st.error(
        "מודל ה-ML לא נמצא. יש להריץ תחילה `python scripts/train_model.py` "
        "כדי לייצר את models/model.joblib."
    )

watchlist = st.session_state.watchlist
if not watchlist:
    st.info("הוסף טיקר בסרגל הצד כדי להתחיל.")
    st.stop()

summary_data = get_summary_data(tuple(watchlist))

for ticker in watchlist:
    with st.container(border=True):
        hist = summary_data.get(ticker)
        if hist is None or hist.empty:
            st.write(f"**{ticker}** - ⚠️ לא ניתן היה לשלוף נתונים כרגע. נסה שוב מאוחר יותר.")
            continue

        hist = compute_all_indicators(hist)
        last_price = hist["Close"].iloc[-1]
        prev_price = hist["Close"].iloc[-2] if len(hist) > 1 else last_price
        pct_change = (last_price / prev_price - 1) * 100 if prev_price else 0.0

        signal = evaluate(hist)
        proba = predict_proba_up(hist) if model_is_available() else None

        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            st.subheader(ticker)
            st.metric("מחיר אחרון", f"${last_price:.2f}", f"{pct_change:+.2f}%")
        with col2:
            st.write("**פסק דין טכני**")
            st.write(signal["verdict"])
        with col3:
            st.write("**הסתברות ML לעלייה**")
            if proba is not None:
                st.progress(proba, text=f"{proba * 100:.0f}%")
            else:
                st.caption("אין מספיק נתונים")

        st.page_link("pages/1_Stock_Detail.py", label=f"➡️ פירוט מלא ל-{ticker}")
