import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import fitz  # PyMuPDF
import pandas as pd
import pytesseract
from dotenv import load_dotenv
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from pdf2image import convert_from_bytes


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET", "dev-secret")

HEBREW_LETTER_PATTERN = re.compile(r"[\u0590-\u05FF]")
DIRECTIONAL_MARKS_PATTERN = re.compile(r"[\u200e\u200f]")
ROOM_PREFIX_PATTERNS: Iterable[Tuple[re.Pattern[str], str]] = (
    (re.compile(r"^\s*כ\.?\s*חדר\s*$"), "כ.חדר"),
    (re.compile(r"^\s*ח\.?\s*חדר\s*$"), "ח.חדר"),
)
READING_LIGHT_PATTERNS: Iterable[re.Pattern[str]] = (
    re.compile(r"^מ\.?$"),
    re.compile(r"^מ\.?\s*קריאה\.?$"),
)
PAS_LED_COMPACT = {"פסלד", "לדפס"}


def sanitize_line(line: str) -> str:
    """Normalize whitespace, remove bullets, and trim helper characters."""
    line = DIRECTIONAL_MARKS_PATTERN.sub("", line)
    line = line.replace("־", " ")
    line = re.sub(r"[•·●◦▪▫►]+", " ", line)
    line = re.sub(r"^[\s\-\u2022•·●]+", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip(" -\t")


def has_meaningful_text(text: str) -> bool:
    if not text:
        return False
    hebrew_letters = HEBREW_LETTER_PATTERN.findall(text)
    if len(hebrew_letters) >= 10:
        return True
    words = [word for word in re.split(r"\s+", text) if HEBREW_LETTER_PATTERN.search(word)]
    return len(words) >= 5


def extract_text_with_pymupdf(pdf_bytes: bytes) -> str:
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            page_texts = [page.get_text("text") for page in document]
        return "\n".join(page_texts)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError("כשל בקריאת ה-PDF (חילוץ טקסט).") from exc


def ocr_pdf(pdf_bytes: bytes) -> str:
    poppler_path = os.getenv("POPPLER_PATH") or None
    try:
        images = convert_from_bytes(pdf_bytes, dpi=300, poppler_path=poppler_path)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError("כשל בהמרת PDF לתמונות. ודאו ש-Poppler מותקן.") from exc

    texts: List[str] = []
    for index, image in enumerate(images):
        try:
            texts.append(pytesseract.image_to_string(image, lang="heb+eng"))
        except pytesseract.TesseractError as exc:
            raise RuntimeError(
                "כשל בהרצת Tesseract OCR. ודאו ש-Tesseract מותקן עם חבילות השפה heb+eng."
            ) from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"כשל ב-OCR בעמוד {index + 1}.") from exc
    return "\n".join(texts)


def extract_pdf_content(pdf_bytes: bytes) -> Tuple[str, str]:
    text = ""
    extraction_mode = "טקסט"
    try:
        text = extract_text_with_pymupdf(pdf_bytes)
    except RuntimeError:
        text = ""

    if has_meaningful_text(text):
        return text, extraction_mode

    extraction_mode = "OCR"
    text = ocr_pdf(pdf_bytes)
    if has_meaningful_text(text):
        return text, extraction_mode

    raise RuntimeError("לא נמצאו נתונים ניתנים לקריאה ב-PDF שסופק.")


def _flush_block(block: List[str], labels: List[str]) -> None:
    if not block:
        return
    seen = set()
    for item in block:
        cleaned = re.sub(r"\s+", " ", item).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        labels.append(cleaned)
    block.clear()


def parse_labels_from_text(text: str) -> List[str]:
    labels: List[str] = []
    current_block: List[str] = []

    for raw_line in text.splitlines():
        sanitized = sanitize_line(raw_line)
        if not sanitized:
            _flush_block(current_block, labels)
            continue

        if ":" in sanitized:
            potential = sanitized.split(":", 1)[1].strip()
            if potential and HEBREW_LETTER_PATTERN.search(potential):
                sanitized = potential
            else:
                continue

        parts = [part.strip() for part in re.split(r"[\\/\|]+", sanitized) if part.strip()]
        if not parts:
            continue

        for part in parts:
            if not HEBREW_LETTER_PATTERN.search(part):
                continue
            current_block.append(part)

    _flush_block(current_block, labels)
    return labels


def normalize_label(label: str) -> str:
    label = re.sub(r"\s+", " ", label.strip())
    if not label:
        return ""

    label = label.replace("ילון", "וילון")

    if re.search(r"מראה", label) and re.search(r"נסתר", label):
        label = "מראה ונסתרת"

    for pattern in READING_LIGHT_PATTERNS:
        if pattern.fullmatch(label):
            label = "מ.קריאה"
            break

    if re.fullmatch(r"מ\.?\s*קריאה\.?", label):
        label = "מ.קריאה"

    for pattern, replacement in ROOM_PREFIX_PATTERNS:
        if pattern.match(label):
            label = replacement
            break

    compact = re.sub(r"\s+", "", label)
    skip_flip = False
    if compact in PAS_LED_COMPACT:
        label = "פס לד"
        skip_flip = True
    elif label in {"לד פס", "פס לד"}:
        label = "פס לד"
        skip_flip = True

    words = label.split(" ")
    if len(words) == 2 and not skip_flip:
        label = " ".join(reversed(words))

    return re.sub(r"\s+", " ", label.strip())


def build_combined_counter(labels: List[str]) -> Counter:
    counter = Counter()
    for raw in labels:
        normalized = normalize_label(raw)
        if not normalized:
            continue
        if normalized in {"וילון קיר", "קיר וילון"}:
            counter["וילון"] += 1
            counter["קיר"] += 1
        elif normalized in {"וילון לילה", "לילה וילון"}:
            counter["וילון"] += 1
            counter["לילה"] += 1
        else:
            counter[normalized] += 1
    return counter


def export_expanded(counter: Counter, path: Path) -> None:
    rows: List[Dict[str, str]] = []
    for label, count in sorted(counter.items(), key=lambda item: item[0]):
        words = label.split(" ")
        if len(words) == 2:
            first, second = words
            for _ in range(count):
                rows.append({"A": first, "B": second})
        else:
            for _ in range(count):
                rows.append({"A": "", "B": label})
    df = pd.DataFrame(rows, columns=["A", "B"])
    df.to_excel(path, index=False, sheet_name="expanded")


@app.route("/")
def index():
    results = session.pop("results", None)
    return render_template("index.html", results=results)


@app.route("/process", methods=["POST"])
def process():
    pdf_file = request.files.get("pdf_file")
    if not pdf_file or not pdf_file.filename:
        flash("נא להעלות קובץ PDF אחד לעיבוד.", "error")
        return redirect(url_for("index"))

    try:
        pdf_bytes = pdf_file.read()
        if not pdf_bytes:
            raise ValueError("קובץ ה-PDF שסופק ריק.")

        text, extraction_mode = extract_pdf_content(pdf_bytes)
        labels_raw = parse_labels_from_text(text)

        if not labels_raw:
            raise ValueError("לא נמצאו תוויות בעמודים שנסרקו.")

        combined_counter = build_combined_counter(labels_raw)
        if not combined_counter:
            raise ValueError("לא נמצאו תוויות לאחר נרמול וספירה.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"button_expanded_split_{timestamp}.xlsx"
        output_path = OUTPUT_DIR / output_name
        export_expanded(combined_counter, output_path)

        session["results"] = {
            "file": {"label": "הורד את קובץ האקסל", "name": output_name},
            "total_labels": len(labels_raw),
            "total_rows": sum(combined_counter.values()),
            "extraction_mode": extraction_mode,
        }
        flash("העיבוד הסתיים בהצלחה!", "success")
    except Exception as exc:  # pylint: disable=broad-except
        flash(str(exc), "error")

    return redirect(url_for("index"))


@app.route("/download/<path:filename>")
def download_file(filename: str):
    target = OUTPUT_DIR / filename
    if not target.exists():
        flash("הקובץ המבוקש לא נמצא.", "error")
        return redirect(url_for("index"))
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
