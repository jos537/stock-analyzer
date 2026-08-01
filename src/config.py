"""קבועים משותפים לכל האפליקציה."""

# רשימת מעקב ברירת מחדל - מוצגת מיד עם הכניסה הראשונה, בלי צורך בהגדרה
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# מספר ימי מסחר קדימה שעבורם מוערכת הסתברות העלייה
PREDICTION_HORIZON_DAYS = 5

# יקום המניות עליו מאומן מודל ה-ML (מניות גדולות ונזילות)
TRAINING_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "COST", "PEP",
    "ADBE", "CSCO", "NFLX", "AMD", "INTC", "QCOM", "TXN", "AMGN", "INTU", "HON",
    "SBUX", "GILD", "MDLZ", "ADI", "BKNG", "REGN", "VRTX", "PANW", "PYPL", "MU",
    "JPM", "V", "MA", "HD", "PG", "JNJ", "UNH", "XOM", "BAC", "KO",
]

# כמה שנות היסטוריה נשלפות לכל מניה (לאימון ולתצוגה)
HISTORY_PERIOD = "5y"
