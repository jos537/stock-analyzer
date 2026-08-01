"""אימון מודל ה-ML. מריצים פעם אחת מקומית (לא בענן):

    python scripts\\train_model.py

הפלט: models/model.joblib + models/model_metadata.json, שנשמרים בגיט
ונטענים על ידי האפליקציה. לאימון מחדש בעתיד - פשוט מריצים שוב ואז git push.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HISTORY_PERIOD, PREDICTION_HORIZON_DAYS, TRAINING_UNIVERSE  # noqa: E402
from src.indicators import compute_all_indicators  # noqa: E402
from src.ml_model import FEATURE_COLUMNS, add_ml_features, build_labels  # noqa: E402

import joblib  # noqa: E402


def fetch_and_prepare(ticker: str) -> pd.DataFrame | None:
    try:
        hist = yf.Ticker(ticker).history(period=HISTORY_PERIOD, interval="1d")
    except Exception as e:
        print(f"  [דילוג] {ticker}: שגיאת שליפה - {e}")
        return None
    if hist.empty or len(hist) < 250:
        print(f"  [דילוג] {ticker}: מעט מדי נתונים ({len(hist)} שורות)")
        return None

    hist = compute_all_indicators(hist)
    hist = add_ml_features(hist)
    hist["LABEL"] = build_labels(hist, PREDICTION_HORIZON_DAYS)
    hist["TICKER"] = ticker
    return hist


def main():
    print(f"אימון מודל על {len(TRAINING_UNIVERSE)} מניות, אופק תחזית = {PREDICTION_HORIZON_DAYS} ימי מסחר\n")

    frames = []
    for ticker in TRAINING_UNIVERSE:
        print(f"שולף נתונים: {ticker}")
        df = fetch_and_prepare(ticker)
        if df is not None:
            frames.append(df)

    if len(frames) < 5:
        print("שגיאה: יותר מדי מניות נכשלו בשליפה, לא ניתן לאמן מודל.")
        sys.exit(1)

    combined = pd.concat(frames, axis=0)
    combined = combined.reset_index().rename(columns={"index": "Date"})

    # השמטת שורות ללא label (הימים האחרונים של כל מניה) וללא features מלאים
    # (השורות הראשונות, לפני שהחלונות המתגלגלים כמו SMA200 התמלאו)
    required_cols = FEATURE_COLUMNS + ["LABEL"]
    before = len(combined)
    combined = combined.dropna(subset=required_cols)
    print(f"\nשורות לאחר ניקוי NaN: {len(combined)} (מתוך {before})")

    # חלוקה כרונולוגית - קריטי כדי למנוע דליפת מידע מהעתיד.
    # לא מערבבים אקראית בין תאריכים; המבחן הוא תמיד "אחרי" האימון בזמן.
    combined = combined.sort_values("Date")
    cutoff_idx = int(len(combined) * 0.8)
    cutoff_date = combined.iloc[cutoff_idx]["Date"]
    train_df = combined[combined["Date"] < cutoff_date]
    test_df = combined[combined["Date"] >= cutoff_date]
    print(f"חיתוך כרונולוגי בתאריך: {cutoff_date.date()}")
    print(f"אימון: {len(train_df)} שורות | בדיקה: {len(test_df)} שורות")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["LABEL"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["LABEL"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=40,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, test_pred)
    auc = roc_auc_score(y_test, test_proba)

    print(f"\nתוצאות על נתוני בדיקה (לא נראו באימון):")
    print(f"  דיוק (accuracy): {acc:.3f}")
    print(f"  ROC-AUC: {auc:.3f}")
    print("  (דיוק סביב 53-60% ריאלי לבעיה הזו - המודל לא אמור 'לדעת' לחזות מניות בוודאות)")

    MODEL_DIR = PROJECT_ROOT / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "horizon_days": PREDICTION_HORIZON_DAYS,
        "feature_columns": FEATURE_COLUMNS,
        "training_universe_size": len(TRAINING_UNIVERSE),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_accuracy": round(acc, 4),
        "test_roc_auc": round(auc, 4),
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nנשמר: {MODEL_DIR / 'model.joblib'}")
    print(f"נשמר: {MODEL_DIR / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
