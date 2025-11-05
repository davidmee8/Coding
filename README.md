# מנוע חילוץ תוויות לחצנים

יישום Flask שמחלץ תוויות לחצני תאורה מקובצי PDF ומייצא אותן לקובץ Excel בודד.

## הפעלה

1. צרו סביבה וירטואלית והתקינו את התלויות:

   ```bash
   pip install -r requirements.txt
   ```

2. הפעילו את השרת:

   ```bash
   flask --app app run
   ```

   כברירת מחדל השרת זמין ב-`http://127.0.0.1:5000`.

## קלט ופלט

- היישום מקבל קובץ PDF יחיד.
- הטקסט מחולץ תחילה באמצעות PyMuPDF; אם לא נמצאה כמות טקסט מספקת מופעל מנגנון OCR (pytesseract + pdf2image).
- לאחר נרמול וספירה נוצרת חוברת Excel אחת בשם `button_expanded_split_<timestamp>.xlsx` בתיקיית `outputs/`.
- קובץ ה-Excel כולל גיליון יחיד בשם `expanded` עם עמודות A,B ושורה עבור כל מופע תווית.

## דרישות OCR

לשימוש במנגנון ה-OCR יש לוודא שהרכיבים הבאים מותקנים ברמת מערכת ההפעלה:

- **Tesseract OCR** עם חבילות השפה `heb` ו-`eng`.
- **Poppler** (מספק את כלי `pdftoppm` בהם משתמשת pdf2image).

ניתן להגדיר את המיקום של Tesseract ו-Poppler בעזרת משתני הסביבה הבאים:

- `TESSERACT_CMD` – נתיב מלא להפעלת `tesseract` (אם אינו זמין במשתנה PATH).
- `POPPLER_PATH` – נתיב לתיקיית bin של Poppler.

## משתני סביבה נוספים

- `APP_SECRET` – מחרוזת לשימוש כמפתח סשן של Flask (ברירת המחדל: `dev-secret`).
