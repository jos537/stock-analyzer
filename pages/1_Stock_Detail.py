"""עמוד פירוט למניה בודדת: גרף נרות, אינדיקטורים, פסק דין והסתברות ML."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_WATCHLIST, PREDICTION_HORIZON_DAYS  # noqa: E402
from src.data import get_history  # noqa: E402
from src.indicators import compute_all_indicators  # noqa: E402
from src.ml_model import get_metadata, model_is_available, predict_proba_up  # noqa: E402
from src.signals import evaluate  # noqa: E402

st.set_page_config(page_title="פירוט מניה", page_icon="📈", layout="centered")

st.warning("⚠️ כלי חינוכי בלבד - אינו ייעוץ השקעות.", icon="⚠️")

watchlist = st.session_state.get("watchlist", list(DEFAULT_WATCHLIST))
ticker = st.selectbox("בחר מניה", watchlist)

if not ticker:
    st.stop()

hist = get_history(ticker)
if hist is None or hist.empty:
    st.error(f"לא ניתן לשלוף נתונים עבור {ticker}. נסה שוב מאוחר יותר.")
    st.stop()

hist = compute_all_indicators(hist)

st.title(f"📈 {ticker}")

# --- גרף נרות + ממוצעים נעים + רצועות בולינגר, עם תת-גרפים ל-RSI ו-MACD ---
plot_df = hist.tail(180)  # כ-9 חודשים אחרונים, כדי שהגרף יישאר קריא

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
    row_heights=[0.6, 0.2, 0.2],
    subplot_titles=("מחיר", "RSI", "MACD"),
)

fig.add_trace(go.Candlestick(
    x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
    low=plot_df["Low"], close=plot_df["Close"], name="מחיר",
), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA20"], name="SMA20", line=dict(width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA50"], name="SMA50", line=dict(width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_UPPER"], name="Bollinger עליון", line=dict(width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_LOWER"], name="Bollinger תחתון", line=dict(width=1, dash="dot")), row=1, col=1)

fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["RSI14"], name="RSI14", line=dict(width=1)), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["MACD_HIST"], name="MACD Hist"), row=3, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD"], name="MACD", line=dict(width=1)), row=3, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD_SIGNAL"], name="Signal", line=dict(width=1)), row=3, col=1)

fig.update_layout(height=750, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# --- פסק דין טכני ---
signal = evaluate(hist)
st.subheader("פסק דין טכני")
st.write(signal["verdict"])
with st.expander("למה?", expanded=True):
    for reason in signal["reasons"]:
        st.write(f"- {reason}")

# --- הסתברות ML ---
st.subheader("הערכת מודל למידת מכונה")
if not model_is_available():
    st.info("המודל עדיין לא אומן. יש להריץ `python scripts/train_model.py`.")
else:
    proba = predict_proba_up(hist)
    if proba is None:
        st.info("אין מספיק היסטוריה עבור מניה זו כדי לחשב הערכה (נדרשות כ-200 ימי מסחר).")
    else:
        meta = get_metadata()
        st.progress(proba, text=f"{proba * 100:.0f}% הסתברות לעלייה תוך {PREDICTION_HORIZON_DAYS} ימי מסחר")
        st.caption(
            "ההערכה מבוססת על דפוסים היסטוריים במניות גדולות בארה\"ב, ואינה תחזית ודאית."
        )
        with st.expander("על המודל"):
            st.write(f"דיוק על נתוני בדיקה היסטוריים: {meta['test_accuracy'] * 100:.1f}%")
            st.write(f"ROC-AUC: {meta['test_roc_auc']:.2f}")
            if "test_brier_score" in meta:
                st.write(f"Brier score (איכות כיול ההסתברות): {meta['test_brier_score']:.3f}")
            if "diagnostic_cv_mean_accuracy" in meta:
                st.write(
                    f"דיוק ממוצע על פני {5} חלונות זמן שונים: "
                    f"{meta['diagnostic_cv_mean_accuracy'] * 100:.1f}% "
                    f"(± {meta['diagnostic_cv_std_accuracy'] * 100:.1f}%)"
                )
            st.write(
                "דיוק סביב 53-58% הוא ריאלי וצפוי לבעיה הזו - "
                "אין מודל שיכול לחזות מניות בוודאות גבוהה."
            )
