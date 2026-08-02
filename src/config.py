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
