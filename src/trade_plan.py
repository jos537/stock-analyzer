"""לוגיקת תוכנית מסחר: תמיכה/התנגדות, עוצמה יחסית, תוכנית מבוססת ATR,
בדיקה היסטורית בסיסית, דגלי אזהרה, תוחלת, וגודל פוזיציה.

מודול חישוב טהור בלבד - אין כאן קריאות רשת. כל שליפת הנתונים (מחירים,
סקטור, דוח כספי, חדשות) קורית ב-data.py / news.py ומועברת הנה כבר מוכנה.
"""

import numpy as np
import pandas as pd

from src.config import (
    ANALOG_MIN_SAMPLE, ANALOG_RSI_BAND, ATR_STOP_MULT, ATR_TARGET1_MULT,
    ATR_TARGET2_MULT, MIN_DOLLAR_VOLUME, MIN_RR_RATIO, PREDICTION_HORIZON_DAYS,
)


def compute_support_resistance(df: pd.DataFrame) -> dict:
    """שיא/שפל שבועי (5 ימי מסחר), חודשי (21) ורבעוני (63)."""
    def high_low(window: int) -> dict:
        tail = df.tail(window)
        return {"high": float(tail["High"].max()), "low": float(tail["Low"].min())}

    return {
        "weekly": high_low(5),
        "monthly": high_low(21),
        "quarterly": high_low(63),
    }


def compute_relative_strength(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame,
                               lookback_days: int = 20) -> float | None:
    """תשואת lookback_days של המניה פחות תשואת אותו טווח של אמת-המידה
    (SPY או ETF סקטור). None אם אין מספיק נתונים באחד מהם."""
    if stock_df is None or benchmark_df is None:
        return None
    if len(stock_df) <= lookback_days or len(benchmark_df) <= lookback_days:
        return None
    stock_ret = stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[-lookback_days - 1] - 1
    bench_ret = benchmark_df["Close"].iloc[-1] / benchmark_df["Close"].iloc[-lookback_days - 1] - 1
    return float(stock_ret - bench_ret)


def build_trade_plan(df: pd.DataFrame, direction: str = "long", *,
                      stop_mult: float = ATR_STOP_MULT,
                      target1_mult: float = ATR_TARGET1_MULT,
                      target2_mult: float = ATR_TARGET2_MULT,
                      min_rr: float = MIN_RR_RATIO) -> dict | None:
    """בונה תוכנית מבוססת ATR: entry = מחיר אחרון, סטופ/יעדים לפי כפולות ATR14.
    direction: 'long' או 'short'. מחזיר None אם אין ATR14 זמין (היסטוריה קצרה מדי)."""
    last = df.iloc[-1]
    entry = float(last["Close"])
    atr = last.get("ATR14")
    if pd.isna(atr):
        return None
    atr = float(atr)

    sign = 1 if direction == "long" else -1
    stop = entry - sign * stop_mult * atr
    target1 = entry + sign * target1_mult * atr
    target2 = entry + sign * target2_mult * atr

    risk = abs(entry - stop)
    reward1 = abs(target1 - entry)
    reward2 = abs(target2 - entry)
    rr1 = reward1 / risk if risk > 0 else None
    rr2 = reward2 / risk if risk > 0 else None

    return {
        "direction": direction,
        "entry": entry,
        "atr": atr,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "risk_per_share": risk,
        "rr1": rr1,
        "rr2": rr2,
        "meets_min_rr": (rr1 is not None and rr1 >= min_rr),
    }


def historical_analog_stats(df: pd.DataFrame, *, horizon_days: int = PREDICTION_HORIZON_DAYS,
                             rsi_band: float = ANALOG_RSI_BAND,
                             min_sample: int = ANALOG_MIN_SAMPLE) -> dict:
    """*** לא בדיקת מסלול (path-dependent) - לא בודקת אם סטופ/יעד ספציפיים
    היו נפגעים קודם בתוך החלון. רק מפלגת תשואה גולמית ל-horizon_days קדימה
    במצבים היסטוריים "דומים" (RSI קרוב, אותו צד SMA50, אותו סימן MACD). ***

    מיועד כתצפית כיוונית גסה בלבד, לא כהסתברות מדויקת."""
    required = ["RSI14", "SMA50", "MACD_HIST", "Close"]
    h = df.dropna(subset=required).copy()
    h["fwd_return"] = h["Close"].shift(-horizon_days) / h["Close"] - 1
    h = h.dropna(subset=["fwd_return"])

    last = df.iloc[-1]
    cur_rsi = last["RSI14"]
    cur_above_sma50 = last["Close"] > last["SMA50"]
    cur_macd_sign = np.sign(last["MACD_HIST"]) if pd.notna(last["MACD_HIST"]) else None

    if pd.isna(cur_rsi) or cur_macd_sign is None:
        return {"sample_size": 0, "small_sample_warning": True}

    similar = h[
        h["RSI14"].between(cur_rsi - rsi_band, cur_rsi + rsi_band)
        & ((h["Close"] > h["SMA50"]) == cur_above_sma50)
        & (np.sign(h["MACD_HIST"]) == cur_macd_sign)
    ]

    sample_size = len(similar)
    result = {"sample_size": sample_size, "small_sample_warning": sample_size < min_sample}
    if sample_size > 0:
        result.update({
            "mean_return": float(similar["fwd_return"].mean()),
            "median_return": float(similar["fwd_return"].median()),
            "pct_positive": float((similar["fwd_return"] > 0).mean()),
            "min_return": float(similar["fwd_return"].min()),
            "max_return": float(similar["fwd_return"].max()),
        })
    return result


def liquidity_check(df: pd.DataFrame, min_dollar_volume: float = MIN_DOLLAR_VOLUME) -> dict:
    """מחזור כספי יומי ממוצע (20 יום) - נזילות מתחת לרצפה מסומנת כפוסלת."""
    avg_dollar_volume = float((df["Close"] * df["Volume"]).tail(20).mean())
    return {"avg_dollar_volume_20d": avg_dollar_volume, "passes": avg_dollar_volume >= min_dollar_volume}


def compute_red_flags(*, direction: str, macd_bullish: bool | None,
                       sector_rs: float | None, market_rs: float | None,
                       earnings_in_window: bool | None, analog_stats: dict,
                       liquidity: dict, macro_conflicts: list[str]) -> list[str]:
    """מרכז את כל בדיקות ה"ביקורת הנגדית" לרשימת אזהרות ⚠️ בעברית -
    גרסה דטרמיניסטית של שלב ה-red-team שנעשה ידנית בצ'אט."""
    flags: list[str] = []
    wants_up = direction == "long"

    if macd_bullish is not None and macd_bullish != wants_up:
        flags.append(
            "⚠️ MACD לא תואם את כיוון העסקה - "
            + ("MACD דובי בזמן שמחפשים קנייה" if wants_up else "MACD שורי בזמן שמחפשים מכירה בחסר")
        )

    if sector_rs is not None and market_rs is not None:
        if wants_up and sector_rs < 0 and sector_rs < market_rs:
            flags.append("⚠️ הסקטור חלש מהשוק הרחב - קנייה נגד רוח סקטוריאלית")
        if not wants_up and sector_rs > 0 and sector_rs > market_rs:
            flags.append("⚠️ הסקטור חזק מהשוק הרחב - מכירה בחסר נגד רוח סקטוריאלית")

    if earnings_in_window is None:
        flags.append("⚠️ מועד הדוח הכספי הבא לא ידוע - לא ניתן לשלול סיכון דוח")
    elif earnings_in_window:
        flags.append("⚠️ דוח כספי צפוי בתוך חלון ההחזקה - סיכון אירוע גבוה")

    if analog_stats.get("small_sample_warning", True):
        flags.append(
            f"⚠️ מדגם היסטורי קטן מדי ({analog_stats.get('sample_size', 0)} מקרים) - "
            "הערכה לא אמינה"
        )

    if not liquidity.get("passes", False):
        flags.append("⚠️ נזילות נמוכה - מחזור כספי יומי מתחת לרצפה שהוגדרה")

    flags.extend(f"⚠️ {c}" for c in macro_conflicts)

    return flags


def compute_expected_value(proba_up: float | None, trade_plan: dict | None,
                            proba_margin: float = 0.05) -> dict | None:
    """EV = p*reward - (1-p)*risk, לפי יעד 1. מציג טווח [p-margin, p+margin]
    ולא מספר יחיד - כי דיוק המודל עצמו נמוך (~50-52%), הצגת דיוק מזויף
    הייתה מטעה. מחזיר None אם חסר proba או trade_plan."""
    if proba_up is None or trade_plan is None:
        return None
    reward = trade_plan["risk_per_share"] * trade_plan.get("rr1") if trade_plan.get("rr1") else None
    risk = trade_plan["risk_per_share"]
    if reward is None:
        return None

    p_low = max(0.0, proba_up - proba_margin)
    p_high = min(1.0, proba_up + proba_margin)

    def ev(p: float) -> float:
        return p * reward - (1 - p) * risk

    return {
        "proba_used": proba_up,
        "proba_range": (p_low, p_high),
        "ev_low": ev(p_low),
        "ev_high": ev(p_high),
        "ev_mid": ev(proba_up),
    }


def position_size(portfolio_value: float, risk_pct: float, entry: float, stop: float) -> dict:
    """חישוב אריתמטי טהור - כמה מניות לקנות כדי לא לחרוג מסיכון מוגדר."""
    risk_amount = portfolio_value * risk_pct
    risk_per_share = abs(entry - stop)
    shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
    position_value = shares * entry
    return {
        "risk_amount": risk_amount,
        "shares": shares,
        "position_value": position_value,
        "position_pct_of_portfolio": (position_value / portfolio_value) if portfolio_value > 0 else 0.0,
    }
