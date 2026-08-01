# ניתוח מניות

אפליקציית Streamlit לניתוח טכני והערכת הסתברות (למידת מכונה) למניות אמריקאיות.

⚠️ **כלי חינוכי בלבד. אינו ייעוץ השקעות.** ההערכות מבוססות על נתונים היסטוריים
בלבד ואינן מבטיחות תוצאות עתידיות.

## הרצה מקומית

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\train_model.py   # פעם ראשונה בלבד - מייצר את models/model.joblib
streamlit run app.py
```

## אימון מחדש של המודל

```powershell
python scripts\train_model.py
git add models\model.joblib models\model_metadata.json
git commit -m "Retrain model"
git push
```
