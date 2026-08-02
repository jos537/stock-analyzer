"""אימון מודל ה-ML. מריצים פעם אחת מקומית (לא בענן):

    python scripts\\train_model.py

הפלט: models/model.joblib + models/model_metadata.json, שנשמרים בגיט
ונטענים על ידי האפליקציה. לאימון מחדש בעתיד - פשוט מריצים שוב ואז git push.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    HISTORY_PERIOD, LABEL_MOVE_THRESHOLD, MARKET_TICKER,
    PREDICTION_HORIZON_DAYS, TRAINING_UNIVERSE,
)
from src.indicators import compute_all_indicators  # noqa: E402
from src.ml_model import (  # noqa: E402
    FEATURE_COLUMNS, add_market_features, add_ml_features,
    build_labels, build_market_features,
)

import joblib  # noqa: E402


def fetch_and_prepare(ticker: str, market_features: pd.DataFrame) -> pd.DataFrame | None:
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
    hist = add_market_features(hist, market_features)
    hist["LABEL"] = build_labels(hist, PREDICTION_HORIZON_DAYS, LABEL_MOVE_THRESHOLD)
    hist["TICKER"] = ticker
    return hist


def run_diagnostic_cv(combined: pd.DataFrame, n_splits: int = 5):
    """אבחון: מריץ TimeSeriesSplit על הנתונים המאוחדים (ממוינים לפי תאריך)
    ומדפיס דיוק לכל חלון זמן בנפרד. עוזר להבין אם תוצאה חלשה נובעת מ'מזל רע'
    של חלוקה בודדת (רגים שוק חריג בתקופת הבדיקה) או מבעיה עקבית."""
    print(f"\n--- אבחון: {n_splits} חלונות זמן נפרדים (TimeSeriesSplit) ---")
    X = combined[FEATURE_COLUMNS].to_numpy()
    y = combined["LABEL"].to_numpy()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, aucs = [], []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        clf = RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_leaf=40,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        acc = accuracy_score(y[test_idx], pred)
        auc = roc_auc_score(y[test_idx], proba)
        accs.append(acc)
        aucs.append(auc)
        test_dates = combined.iloc[test_idx]["Date"]
        print(f"  חלון {i}: {test_dates.min().date()} עד {test_dates.max().date()} | "
              f"דיוק={acc:.3f} | AUC={auc:.3f}")
    print(f"  ממוצע דיוק: {np.mean(accs):.3f} (סטיית תקן: {np.std(accs):.3f})")
    print(f"  ממוצע AUC: {np.mean(aucs):.3f} (סטיית תקן: {np.std(aucs):.3f})")
    print("  סטיית תקן גבוהה בין חלונות = הביצועים תלויים מאוד בתקופת השוק שנבדקה,")
    print("  לא בהכרח בעיה בקוד. זה צפוי בבעיית ניבוי מניות.")
    return {"cv_mean_accuracy": round(float(np.mean(accs)), 4),
            "cv_std_accuracy": round(float(np.std(accs)), 4),
            "cv_mean_auc": round(float(np.mean(aucs)), 4),
            "cv_std_auc": round(float(np.std(aucs)), 4)}


def main():
    print(f"אימון מודל על {len(TRAINING_UNIVERSE)} מניות, אופק תחזית = {PREDICTION_HORIZON_DAYS} ימי מסחר")
    print(f"סף תווית (label threshold): {LABEL_MOVE_THRESHOLD * 100:.1f}%\n")

    print(f"שולף נתוני שוק כלליים: {MARKET_TICKER}")
    spy_hist = yf.Ticker(MARKET_TICKER).history(period=HISTORY_PERIOD, interval="1d")
    spy_hist = compute_all_indicators(spy_hist)
    market_features = build_market_features(spy_hist)

    frames = []
    for ticker in TRAINING_UNIVERSE:
        print(f"שולף נתונים: {ticker}")
        df = fetch_and_prepare(ticker, market_features)
        if df is not None:
            frames.append(df)

    if len(frames) < 5:
        print("שגיאה: יותר מדי מניות נכשלו בשליפה, לא ניתן לאמן מודל.")
        sys.exit(1)

    combined = pd.concat(frames, axis=0)
    combined = combined.reset_index().rename(columns={"index": "Date"})

    # השמטת שורות ללא label (הימים האחרונים של כל מניה, ותזוזות קטנות מדי -
    # ראו LABEL_MOVE_THRESHOLD) וללא features מלאים (תחילת ההיסטוריה של כל מניה,
    # לפני שהחלונות המתגלגלים כמו SMA200 התמלאו)
    required_cols = FEATURE_COLUMNS + ["LABEL"]
    before = len(combined)
    combined = combined.dropna(subset=required_cols)
    print(f"\nשורות לאחר ניקוי NaN: {len(combined)} (מתוך {before})")
    combined = combined.sort_values("Date").reset_index(drop=True)

    diagnostic = run_diagnostic_cv(combined)

    # חלוקה כרונולוגית סופית ל-3: אימון / כיול / בדיקה. קריטי כדי למנוע דליפת
    # מידע מהעתיד - לעולם לא מערבבים אקראית בין תאריכים.
    n = len(combined)
    train_end = int(n * 0.6)
    calib_end = int(n * 0.8)
    train_df = combined.iloc[:train_end]
    calib_df = combined.iloc[train_end:calib_end]
    test_df = combined.iloc[calib_end:]
    print(f"\nחיתוך כרונולוגי: אימון עד {train_df['Date'].max().date()}, "
          f"כיול עד {calib_df['Date'].max().date()}, בדיקה עד {test_df['Date'].max().date()}")
    print(f"אימון: {len(train_df)} | כיול: {len(calib_df)} | בדיקה: {len(test_df)}")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["LABEL"]
    X_calib, y_calib = calib_df[FEATURE_COLUMNS], calib_df["LABEL"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["LABEL"]

    base_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=40,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    base_model.fit(X_train, y_train)

    # כיול הסתברויות: מוודא שכש-המודל אומר "60%" זה קרוב בפועל ל-60% הצלחה,
    # לא רק מספר גולמי. מכויל על סט נפרד (calib) שהמודל לא התאמן עליו.
    calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method="isotonic")
    calibrated_model.fit(X_calib, y_calib)

    test_pred = calibrated_model.predict(X_test)
    test_proba = calibrated_model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, test_pred)
    auc = roc_auc_score(y_test, test_proba)
    brier = brier_score_loss(y_test, test_proba)

    print(f"\nתוצאות סופיות על נתוני בדיקה (לא נראו באימון ולא בכיול):")
    print(f"  דיוק (accuracy): {acc:.3f}")
    print(f"  ROC-AUC: {auc:.3f}")
    print(f"  Brier score (איכות כיול, נמוך יותר=טוב יותר, 0.25=ניחוש אקראי): {brier:.3f}")
    print("  דיוק סביב 53-58% ריאלי לבעיה הזו - המודל לא אמור 'לדעת' לחזות מניות בוודאות")

    MODEL_DIR = PROJECT_ROOT / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(calibrated_model, MODEL_DIR / "model.joblib")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "horizon_days": PREDICTION_HORIZON_DAYS,
        "label_move_threshold": LABEL_MOVE_THRESHOLD,
        "feature_columns": FEATURE_COLUMNS,
        "training_universe_size": len(TRAINING_UNIVERSE),
        "train_rows": len(train_df),
        "calib_rows": len(calib_df),
        "test_rows": len(test_df),
        "test_accuracy": round(acc, 4),
        "test_roc_auc": round(auc, 4),
        "test_brier_score": round(brier, 4),
        **{f"diagnostic_{k}": v for k, v in diagnostic.items()},
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nנשמר: {MODEL_DIR / 'model.joblib'}")
    print(f"נשמר: {MODEL_DIR / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
