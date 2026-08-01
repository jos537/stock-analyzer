"""פסק דין מבוסס-כללים (heuristic) - נפרד לגמרי מהערכת ה-ML.

הרעיון: כמה תנאי אצבע פשוטים ומוכרים בניתוח טכני, כל אחד תורם ניקוד,
והסכום הכולל קובע תווית. זה *לא* מודל סטטיסטי - זה כלל אצבע שקוף וניתן להסבר.
"""

import pandas as pd

BULLISH = "🟢 חיובי (Bullish)"
NEUTRAL = "🟡 ניטרלי / למעקב"
BEARISH = "🔴 שלילי (Bearish)"


def evaluate(df: pd.DataFrame) -> dict:
    """מקבל DataFrame עם אינדיקטורים מחושבים (ראה indicators.py) ומחזיר
    dict עם: score, verdict, reasons (רשימת הסברים בעברית)."""
    last = df.iloc[-1]
    score = 0.0
    reasons: list[str] = []

    # מגמה מול ממוצעים נעים
    if pd.notna(last.get("SMA50")) and pd.notna(last.get("SMA200")):
        if last["Close"] > last["SMA50"] > last["SMA200"]:
            score += 1
            reasons.append("המחיר מעל הממוצע הנע ל-50 ול-200 יום - מגמת עלייה.")
        elif last["Close"] < last["SMA50"] < last["SMA200"]:
            score -= 1
            reasons.append("המחיר מתחת לממוצע הנע ל-50 ול-200 יום - מגמת ירידה.")

    # RSI
    if pd.notna(last.get("RSI14")):
        rsi = last["RSI14"]
        if rsi < 30:
            score += 1
            reasons.append(f"RSI נמוך ({rsi:.0f}) - המניה במצב 'מכירת יתר', ייתכן תיקון כלפי מעלה.")
        elif rsi > 70:
            score -= 1
            reasons.append(f"RSI גבוה ({rsi:.0f}) - המניה במצב 'קניית יתר', ייתכן תיקון כלפי מטה.")
        else:
            reasons.append(f"RSI ניטרלי ({rsi:.0f}).")

    # MACD
    if pd.notna(last.get("MACD")) and pd.notna(last.get("MACD_SIGNAL")):
        if last["MACD"] > last["MACD_SIGNAL"]:
            score += 1
            reasons.append("MACD מעל קו הסיגנל - חציית שור (Bullish crossover).")
        else:
            score -= 1
            reasons.append("MACD מתחת לקו הסיגנל - חציית דוב (Bearish crossover).")

    # מיקום ברצועות בולינגר
    if pd.notna(last.get("BB_PCTB")):
        pctb = last["BB_PCTB"]
        if pctb >= 1.0:
            score -= 1
            reasons.append("המחיר נוגע/חורג מהרצועה העליונה - זהירות מקניית יתר.")
        elif pctb <= 0.0:
            score -= 1
            reasons.append("המחיר נוגע/חורג מהרצועה התחתונה - מגמת ירידה חזקה, לא בהכרח סימן קנייה.")

    # אישור נפח מסחר
    if pd.notna(last.get("VOL_RATIO")) and last["VOL_RATIO"] > 1.5:
        score += 0.5
        reasons.append("נפח המסחר גבוה משמעותית מהממוצע - תנועה נתמכת בעניין מוגבר.")

    if score >= 2:
        verdict = BULLISH
    elif score <= -2:
        verdict = BEARISH
    else:
        verdict = NEUTRAL

    return {"score": score, "verdict": verdict, "reasons": reasons}
