"""לוח אירועי מאקרו-כלכליים. FOMC מדויק (מתפרסם מראש רשמית ע"י הפד),
CPI/NFP הערכה בלבד (אין להם לוח קבוע כמו FOMC).

*** יש לעדכן את FOMC_MEETING_DATES_2026 פעם בשנה (בערך בדצמבר) מול:
    https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm ***
"""

from datetime import date, timedelta

# תאריך ההחלטה (היום השני של הפגישה, 14:00 שעון מזרח ארה"ב) - אומת מול
# federalreserve.gov ב-2026-08-02
FOMC_MEETING_DATES_2026 = [
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
]


def _first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    offset = (4 - d.weekday()) % 7  # 4 = יום שישי
    return d + timedelta(days=offset)


def get_upcoming_macro_events(start: date, end: date) -> list[dict]:
    """מחזיר אירועי מאקרו שחלים בטווח [start, end]. FOMC ברמת אמינות
    'exact' (מקור: הפד), NFP/CPI ברמת אמינות 'approx' (הערכה בלבד)."""
    events = []

    for d in FOMC_MEETING_DATES_2026:
        if start <= d <= end:
            events.append({"name": "החלטת ריבית (FOMC)", "date": d, "confidence": "exact"})

    months = set()
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        months.add((cursor.year, cursor.month))
        cursor = date(cursor.year + (cursor.month == 12), (cursor.month % 12) + 1, 1)

    for year, month in months:
        nfp_date = _first_friday(year, month)
        if start <= nfp_date <= end:
            events.append({"name": "דוח תעסוקה (NFP, הערכה)", "date": nfp_date, "confidence": "approx"})

        cpi_date = date(year, month, 12)  # הערכה גסה: אמצע-חודש
        if start <= cpi_date <= end:
            events.append({"name": "מדד המחירים לצרכן (CPI, הערכה)", "date": cpi_date, "confidence": "approx"})

    return sorted(events, key=lambda e: e["date"])


def check_macro_conflicts(hold_start: date, hold_end: date) -> list[str]:
    """הודעות ⚠️ מוכנות לתצוגה - לצריכה ישירה ע"י trade_plan.compute_red_flags."""
    events = get_upcoming_macro_events(hold_start, hold_end)
    warnings = []
    for e in events:
        certainty = "" if e["confidence"] == "exact" else " (תאריך משוער, לא מאומת)"
        warnings.append(f"{e['name']} ב-{e['date'].isoformat()} נופל בתוך חלון ההחזקה{certainty}")
    return warnings
