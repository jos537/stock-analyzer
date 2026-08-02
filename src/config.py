"""קבועים משותפים לכל האפליקציה."""

# רשימת מעקב ברירת מחדל - מוצגת מיד עם הכניסה הראשונה, בלי צורך בהגדרה
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# מספר ימי מסחר קדימה שעבורם מוערכת הסתברות העלייה
PREDICTION_HORIZON_DAYS = 5

# תזוזת מחיר מינימלית (5D) כדי להיחשב "עלייה"/"ירידה" ברורה; תזוזות קטנות
# יותר (רעש) מושמטות מהאימון לגמרי - מפחית תוויות שרירותיות/רועשות
LABEL_MOVE_THRESHOLD = 0.01

# מדד השוק הכללי (S&P 500 ETF) המשמש להוספת features של הקשר שוק רחב
MARKET_TICKER = "SPY"

# יקום המניות עליו מאומן מודל ה-ML (מניות גדולות ונזילות, פרוסות על פני סקטורים)
TRAINING_UNIVERSE = [
    # טכנולוגיה
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ADBE", "CSCO",
    "NFLX", "AMD", "INTC", "QCOM", "TXN", "INTU", "PANW", "PYPL", "MU", "ORCL",
    "CRM", "IBM", "NOW", "ADSK", "SNPS", "CDNS",
    # צריכה
    "COST", "PEP", "SBUX", "MDLZ", "HD", "NKE", "MCD", "WMT", "PG", "KO", "DIS", "BKNG",
    # בריאות
    "AMGN", "GILD", "REGN", "VRTX", "JNJ", "UNH", "PFE", "TMO", "ABT", "ABBV",
    "DHR", "LLY", "MRK", "BMY",
    # פיננסים
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "SCHW", "BLK",
    # תעשייה ואנרגיה
    "HON", "XOM", "CVX", "BA", "CAT", "DE", "UPS", "LMT", "RTX", "GE", "MMM", "EMR",
]

# כמה שנות היסטוריה נשלפות לכל מניה (לאימון ולתצוגה)
HISTORY_PERIOD = "5y"


# ============================================================
# תוכנית מסחר (Trade Plan) - קבועים
# ============================================================

# מיפוי סקטור GICS (כפי שמוחזר מ-yfinance) ל-ETF סקטוריאלי, להשוואת עוצמה
# יחסית. טיקר עם סקטור שלא ברשימה -> מדלגים על השוואת סקטור בכנות.
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# תוכנית עסקה מבוססת ATR: סטופ = entry - 1×ATR, יעדים = entry + 2×ATR / 3×ATR
ATR_STOP_MULT = 1.0
ATR_TARGET1_MULT = 2.0
ATR_TARGET2_MULT = 3.0
MIN_RR_RATIO = 2.0  # יחס סיכון-סיכוי מתחת לזה -> העסקה מסומנת כפסולה

# ניהול סיכונים - ברירת מחדל שמרנית
DEFAULT_RISK_PCT = 0.005  # 0.5% מהתיק לעסקה בודדת

# רצפת נזילות (מחזור כספי יומי ממוצע, $) מתחת לה מסמנים אזהרת נזילות
MIN_DOLLAR_VOLUME = 5_000_000

# בדיקה היסטורית בסיסית (analog) - לא backtest מלא של מסלול יעד/סטופ
ANALOG_RSI_BAND = 10.0
ANALOG_MIN_SAMPLE = 20  # מתחת לזה -> דגל "מדגם קטן מדי"

# חדשות: כמה ימים אחורה לשלוף
NEWS_LOOKBACK_DAYS = 14
