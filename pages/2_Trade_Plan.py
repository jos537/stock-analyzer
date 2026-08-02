"""עמוד תוכנית מסחר: דשבורד כמותי - מגמה, תמיכה/התנגדות, נזילות, חדשות,
דוח כספי/מאקרו, בדיקה היסטורית בסיסית, תוכנית מבוססת ATR, תוחלת, גודל
פוזיציה, ודגלי אזהרה. הכל מספרים/טבלאות/דגלים - בלי ניתוח בפרוזה (בלי LLM)."""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DEFAULT_RISK_PCT, DEFAULT_WATCHLIST, MARKET_TICKER,
    PREDICTION_HORIZON_DAYS, SECTOR_ETF_MAP,
)
from src.data import get_history, get_next_earnings_date, get_ticker_sector  # noqa: E402
from src.indicators import compute_all_indicators  # noqa: E402
from src.macro_calendar import check_macro_conflicts  # noqa: E402
from src.ml_model import model_is_available, predict_proba_up  # noqa: E402
from src.news import get_company_news, tag_headline_sentiment  # noqa: E402
from src.signals import evaluate  # noqa: E402
from src.trade_plan import (  # noqa: E402
    build_trade_plan, compute_expected_value, compute_red_flags,
    compute_relative_strength, compute_support_resistance,
    historical_analog_stats, liquidity_check, position_size,
)

st.set_page_config(page_title="תוכנית מסחר", page_icon="🎯", layout="centered")

st.warning(
    "⚠️ **כלי מחקר בלבד - לא ייעוץ השקעות, ולא הבטחת רווח.** "
    "כל הערכה כאן מבוססת על נתונים היסטוריים וכללים אוטומטיים, לא על שיקול דעת אנושי או בינה מלאכותית.",
    icon="⚠️",
)
st.title("🎯 תוכנית מסחר")
st.caption(f"טווח החזקה קבוע: {PREDICTION_HORIZON_DAYS} ימי מסחר. ברירת מחדל: קנייה בלבד.")

watchlist = st.session_state.get("watchlist", list(DEFAULT_WATCHLIST))
col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.selectbox("בחר מניה", watchlist).strip().upper() if watchlist else ""
    custom = st.text_input("או הקלד טיקר אחר")
    if custom:
        ticker = custom.strip().upper()
with col2:
    direction = st.radio("כיוון", ["long", "short"], format_func=lambda d: "קנייה" if d == "long" else "מכירה בחסר")

generate = st.button("🔍 צור ניתוח", type="primary")

if not generate:
    st.info("בחר טיקר וכיוון, ולחץ 'צור ניתוח'. (לא רץ אוטומטית - חוסך קריאות API.)")
    st.stop()

if not ticker:
    st.error("יש להזין טיקר.")
    st.stop()

hist = get_history(ticker)
if hist is None or hist.empty:
    st.error(f"לא ניתן לשלוף נתונים עבור {ticker}. ייתכן שהסימול שגוי.")
    st.stop()

hist = compute_all_indicators(hist)
last = hist.iloc[-1]
last_date = hist.index[-1]

st.caption(f"מקור: Yahoo Finance (yfinance) | נתון אחרון: {last_date.date()} (סוף יום)")

# ===== 1. מצב שוק וסקטור =====
st.header("1. מצב שוק וסקטור")
spy_hist = get_history(MARKET_TICKER)
sector = get_ticker_sector(ticker)
sector_etf = SECTOR_ETF_MAP.get(sector) if sector else None
sector_hist = get_history(sector_etf) if sector_etf else None

market_rs = sector_rs = None
if spy_hist is not None:
    spy_hist_ind = compute_all_indicators(spy_hist)
    market_rs = compute_relative_strength(hist, spy_hist_ind)
if sector_hist is not None:
    sector_hist_ind = compute_all_indicators(sector_hist)
    sector_rs = compute_relative_strength(hist, sector_hist_ind)

mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("סקטור", sector or "לא ידוע")
mcol2.metric(f"עוצמה יחסית מול {MARKET_TICKER} (20d)", f"{market_rs*100:+.2f}%" if market_rs is not None else "אין נתון")
mcol3.metric(f"עוצמה יחסית מול {sector_etf or 'סקטור'} (20d)", f"{sector_rs*100:+.2f}%" if sector_rs is not None else "אין נתון (סקטור לא ממופה)")

# ===== 2. מגמה ומומנטום =====
st.header("2. מגמה ומומנטום")
signal = evaluate(hist)
st.write(f"**פסק דין טכני**: {signal['verdict']}")
with st.expander("פירוט"):
    for r in signal["reasons"]:
        st.write(f"- {r}")

tcol1, tcol2, tcol3, tcol4 = st.columns(4)
tcol1.metric("RSI14", f"{last['RSI14']:.1f}" if pd.notna(last["RSI14"]) else "—")
tcol2.metric("MACD Hist", f"{last['MACD_HIST']:.2f}" if pd.notna(last["MACD_HIST"]) else "—")
tcol3.metric("ATR14", f"${last['ATR14']:.2f}" if pd.notna(last["ATR14"]) else "—")
tcol4.metric("נפח מול ממוצע 20י", f"{last['VOL_RATIO']:.2f}x")

# ===== 3. תמיכה/התנגדות =====
st.header("3. תמיכה והתנגדות")
sr = compute_support_resistance(hist)
sr_rows = [
    {"טווח": "שבועי", "שפל": f"${sr['weekly']['low']:.2f}", "שיא": f"${sr['weekly']['high']:.2f}"},
    {"טווח": "חודשי", "שפל": f"${sr['monthly']['low']:.2f}", "שיא": f"${sr['monthly']['high']:.2f}"},
    {"טווח": "רבעוני", "שפל": f"${sr['quarterly']['low']:.2f}", "שיא": f"${sr['quarterly']['high']:.2f}"},
]
st.table(sr_rows)

# ===== 4. נזילות =====
st.header("4. נזילות")
liq = liquidity_check(hist)
st.metric("מחזור כספי יומי ממוצע (20י)", f"${liq['avg_dollar_volume_20d']:,.0f}")
if not liq["passes"]:
    st.warning("⚠️ נזילות נמוכה מהרצפה שהוגדרה למניה נסחרת קלות.")

# ===== 5. חדשות =====
st.header("5. חדשות אחרונות")
news_items = get_company_news(ticker)
if not news_items:
    st.caption("אין חדשות זמינות (ייתכן שחסר מפתח API, או שאין ידיעות בטווח הזמן).")
else:
    st.caption("תיוג הרגש (🟢/🔴/⚪) הוא הערכה גסה מבוססת מילות מפתח - לא ניתוח אמיתי.")
    for item in news_items[:8]:
        tag = tag_headline_sentiment(item["headline"])
        dt = item["datetime"].strftime("%Y-%m-%d") if item["datetime"] else "—"
        st.write(f"{tag} [{dt}] **{item['source']}**: [{item['headline']}]({item['url']})")

# ===== 6. דוח כספי ואירועי מאקרו =====
st.header("6. דוח כספי ואירועי מאקרו")
earnings_date = get_next_earnings_date(ticker)
hold_start = date.today()
hold_end = hold_start + timedelta(days=int(PREDICTION_HORIZON_DAYS * 1.5))  # מרווח ביטחון לימי מסחר מול ימי לוח

earnings_in_window = None
if earnings_date is not None:
    earnings_in_window = hold_start <= earnings_date.date() <= hold_end
    st.write(f"דוח כספי קרוב: **{earnings_date.date()}**"
             + (" ⚠️ בתוך חלון ההחזקה המשוער" if earnings_in_window else " (מחוץ לחלון ההחזקה)"))
else:
    st.write("⚠️ מועד הדוח הכספי הבא לא ידוע - לא ניתן לשלול סיכון דוח.")

macro_warnings = check_macro_conflicts(hold_start, hold_end)
if macro_warnings:
    st.write("אירועי מאקרו בתוך החלון:")
    for w in macro_warnings:
        st.write(f"- ⚠️ {w}")
else:
    st.write("לא נמצאו אירועי מאקרו ידועים בתוך חלון ההחזקה.")

# ===== 7. בדיקה היסטורית בסיסית =====
st.header("7. בדיקה היסטורית בסיסית")
st.info(
    "⚠️ **זו לא בדיקת מסלול (path-dependent).** לא נבדק אם היעד או הסטופ "
    "היו נפגעים קודם בפועל - רק פילוג תשואה גולמית קדימה במצבים דומים "
    "(RSI דומה, אותו צד ממוצע נע 50, אותו כיוון MACD)."
)
analog = historical_analog_stats(hist)
if analog["sample_size"] == 0:
    st.write("לא נמצאו מספיק נתונים לבדיקה.")
else:
    acol1, acol2, acol3 = st.columns(3)
    acol1.metric("מספר מקרים דומים", analog["sample_size"])
    acol2.metric(f"תשואת {PREDICTION_HORIZON_DAYS} ימים ממוצעת", f"{analog['mean_return']*100:+.2f}%")
    acol3.metric("אחוז מקרים חיוביים", f"{analog['pct_positive']*100:.1f}%")
    if analog["small_sample_warning"]:
        st.warning("⚠️ מדגם קטן מדי (< 20 מקרים) - אין להסתמך על המספרים האלה.")

# ===== 8. מודל ML =====
st.header("8. הסתברות מודל למידת מכונה")
proba = predict_proba_up(hist) if model_is_available() else None
if proba is None:
    st.write("אין הערכת ML זמינה (מודל לא זמין או שאין מספיק היסטוריה).")
else:
    st.progress(proba, text=f"{proba*100:.0f}% הסתברות לעלייה תוך {PREDICTION_HORIZON_DAYS} ימים")
    st.caption("דיוק המודל עצמו על נתוני בדיקה הוא רק ~50-52% - קרוב לניחוש. ראו הרחבה בעמוד 'פירוט מניה'.")

# ===== 9. תוכנית מסחר (ATR) =====
st.header("9. תוכנית מסחר מוצעת (מבוססת ATR)")
plan = build_trade_plan(hist, direction=direction)
if plan is None:
    st.error("אין מספיק היסטוריה לחישוב ATR - לא ניתן לבנות תוכנית.")
else:
    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("כניסה (אחרון)", f"${plan['entry']:.2f}")
    pcol2.metric("Stop Loss", f"${plan['stop']:.2f}")
    pcol3.metric("ATR14", f"${plan['atr']:.2f}")
    tcol1, tcol2 = st.columns(2)
    tcol1.metric("יעד 1", f"${plan['target1']:.2f}", f"R:R = {plan['rr1']:.1f}:1")
    tcol2.metric("יעד 2", f"${plan['target2']:.2f}", f"R:R = {plan['rr2']:.1f}:1")
    if plan["meets_min_rr"]:
        st.success("✅ עומד ביחס סיכון-סיכוי מינימלי (1:2) ליעד הראשון.")
    else:
        st.error("❌ לא עומד ביחס סיכון-סיכוי מינימלי (1:2) ליעד הראשון.")

    ev = compute_expected_value(proba, plan)
    if ev is not None:
        st.write(
            f"**תוחלת משוערת (טווח, לפני עמלות)**: "
            f"${ev['ev_low']:.2f} עד ${ev['ev_high']:.2f} למניה "
            f"(לפי הסתברות {ev['proba_range'][0]*100:.0f}%-{ev['proba_range'][1]*100:.0f}%)"
        )
        st.caption("טווח, לא מספר יחיד - כי דיוק מודל ה-ML נמוך (~50-52%), הצגת מספר מדויק הייתה מטעה.")

# ===== 10. דגלי אזהרה =====
st.header("10. דגלי אזהרה")
macd_bullish = bool(last["MACD_HIST"] > 0) if pd.notna(last["MACD_HIST"]) else None
flags = compute_red_flags(
    direction=direction,
    macd_bullish=macd_bullish,
    sector_rs=sector_rs,
    market_rs=market_rs,
    earnings_in_window=earnings_in_window,
    analog_stats=analog,
    liquidity=liq,
    macro_conflicts=macro_warnings,
)
if flags:
    for f in flags:
        st.write(f)
else:
    st.success("✅ לא נמצאו דגלי אזהרה אוטומטיים.")

# ===== 11. מחשבון גודל פוזיציה =====
st.header("11. מחשבון גודל פוזיציה")
st.caption("חישוב מקומי בלבד - לא נשלח לשום מקום.")
pcol1, pcol2 = st.columns(2)
portfolio_value = pcol1.number_input("שווי התיק ($)", min_value=0.0, value=10000.0, step=100.0)
risk_pct = pcol2.number_input("סיכון מרבי לעסקה (%)", min_value=0.0, max_value=10.0,
                                value=DEFAULT_RISK_PCT * 100, step=0.1) / 100

if plan is not None and portfolio_value > 0:
    size = position_size(portfolio_value, risk_pct, plan["entry"], plan["stop"])
    st.write(f"**כמות מניות מוצעת**: {size['shares']}")
    st.write(f"**שווי פוזיציה**: ${size['position_value']:,.2f} ({size['position_pct_of_portfolio']*100:.1f}% מהתיק)")
    st.write(f"**סכום בסיכון**: ${size['risk_amount']:,.2f}")
