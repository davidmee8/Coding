import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
from shutil import which

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pytesseract
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
from tkinter import filedialog, messagebox, ttk
import tkinter as tk


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

DEFAULT_TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEFAULT_POPPLER = r"C:\poppler\Library\bin"

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
else:
    auto_tesseract = which("tesseract")
    if auto_tesseract:
        pytesseract.pytesseract.tesseract_cmd = auto_tesseract

if not TESSERACT_CMD and Path(DEFAULT_TESS).exists():
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESS

POPPLER_PATH = os.getenv("POPPLER_PATH")
if not POPPLER_PATH:
    detected_poppler = which("pdftoppm")
    if detected_poppler:
        POPPLER_PATH = str(Path(detected_poppler).parent)
if not POPPLER_PATH and Path(DEFAULT_POPPLER).exists():
    POPPLER_PATH = DEFAULT_POPPLER

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
    # Count Hebrew/Latin letters
    letters = re.findall(r"[\u0590-\u05FFA-Za-z]", text)
    if len(letters) >= 200:
        return True
    # Or at least several Hebrew words
    words = [w for w in re.split(r"\s+", text) if HEBREW_LETTER_PATTERN.search(w)]
    return len(words) >= 15


def extract_text_with_pymupdf(pdf_bytes: bytes) -> str:
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            page_texts = [page.get_text("text") for page in document]
        return "\n".join(page_texts)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError("כשל בקריאת ה-PDF (חילוץ טקסט).") from exc


def _preprocess_image_for_ocr(pil_img) -> np.ndarray:
    """Grayscale + OTSU binarization to improve OCR on short labels."""
    im = np.array(pil_img.convert("L"))
    im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return im


def ocr_pdf(pdf_bytes: bytes) -> str:
    try:
        # Increase to dpi=400 if OCR is still weak
        images = convert_from_bytes(pdf_bytes, dpi=300, poppler_path=POPPLER_PATH)
    except Exception as exc:
        raise RuntimeError("PDF → image conversion failed. Make sure Poppler is installed.") from exc

    # psm 9 improves recognition of short separated words (better for labeling)
    ts_config = r'--oem 1 --psm 9 -l heb+eng'

    words: List[str] = []
    for index, image in enumerate(images):
        try:
            im = _preprocess_image_for_ocr(image)
            data = pytesseract.image_to_data(im, config=ts_config, output_type=pytesseract.Output.DICT)
            for txt in data.get("text", []):
                txt = (txt or "").strip()
                if txt:
                    words.append(txt)
        except pytesseract.TesseractError as exc:
            raise RuntimeError("Tesseract OCR failed. Make sure Hebrew language pack is installed.") from exc
        except Exception as exc:
            raise RuntimeError(f"OCR failed on page {index + 1}.") from exc

    return " ".join(words)


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


LABEL_QUOTE_RE = re.compile(r"""“([^”]+)”|"([^"]+)"|׳([^׳]+)׳|'([^']+)'""")
ANCHORS = [
    r"ENGRAVING",
    r"BUTTON",
    r"LABEL",
    r"LIGHTING\s+LOADS",
    r"SHADE\s+LOADS",
    r"אנגרייב",
    r"חריטה",
    r"תוויות",
]
ANCHOR_RE = re.compile(rf"({'|'.join(ANCHORS)})[:\s]*", re.IGNORECASE)


def parse_labels_from_text(text: str) -> List[str]:
    """
    Extracts short Hebrew labels from OCR/text extraction:
    - Drops English/metadata lines (Name/Description/etc.)
    - Extracts only short Hebrew segments (1–30 chars)
    - Cleans duplicates and noise
    """
    lines = [sanitize_line(ln) for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    labels: List[str] = []

    # Drop noisy English/metadata lines
    DROP_RE = re.compile(
        r'(?:\bName\b|\bDescription\b|\bOrder information\b|\bWhite\b|\bHZ2\b|\bKPCN\b|\d{2,}|\bLEFT\b|\bRIGHT\b|\bENTRY\b)',
        re.IGNORECASE
    )
    LATIN_RE = re.compile(r'[A-Za-z]')

    # Hebrew short segments extractor
    HEB_SEG_RE = re.compile(r'[\u0590-\u05FF][\u0590-\u05FF\s\.\-]{0,29}')

    for ln in lines:
        if DROP_RE.search(ln) or (LATIN_RE.search(ln) and not HEBREW_LETTER_PATTERN.search(ln)):
            continue

        for m in HEB_SEG_RE.finditer(ln):
            seg = sanitize_line(m.group(0))
            if not seg:
                continue
            if not HEBREW_LETTER_PATTERN.search(seg):
                continue
            if 1 <= len(seg) <= 30:
                labels.append(seg)

    cleaned: List[str] = []
    seen = set()
    for x in labels:
        k = re.sub(r'\s+', ' ', x).strip()
        if not k or len(k) == 1:
            continue
        if k not in seen:
            seen.add(k)
            cleaned.append(k)

    return cleaned


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


class LabelExtractorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("מחלץ תוויות מ-PDF")
        self.root.geometry("720x520")

        self.selected_file = tk.StringVar()
        self.status_var = tk.StringVar(value="בחרו קובץ PDF לעיבוד.")
        self.summary_var = tk.StringVar(value="")
        self._last_output_path: Optional[Path] = None

        self._build_ui()

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_frame, text="בחר קובץ PDF", command=self.select_file).pack(
            side=tk.LEFT
        )

        ttk.Label(file_frame, textvariable=self.selected_file, wraplength=420).pack(
            side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True
        )

        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=(0, 10))

        self.process_button = ttk.Button(
            action_frame, text="עבד קובץ", command=self.process_selected_file
        )
        self.process_button.pack(side=tk.LEFT)

        self.open_output_button = ttk.Button(
            action_frame,
            text="פתח תיקיית פלט",
            command=self.open_output_location,
            state=tk.DISABLED,
        )
        self.open_output_button.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(main_frame, textvariable=self.status_var).pack(fill=tk.X, pady=(0, 5))
        ttk.Label(main_frame, textvariable=self.summary_var).pack(fill=tk.X, pady=(0, 10))

        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("label", "count")
        self.tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            height=12,
        )
        self.tree.heading("label", text="תווית")
        self.tree.heading("count", text="כמות")
        self.tree.column("label", width=420, anchor=tk.W)
        self.tree.column("count", width=80, anchor=tk.CENTER)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

    def select_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="בחרו קובץ PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if file_path:
            self.selected_file.set(file_path)
            self.status_var.set("לחצו 'עבד קובץ' כדי להתחיל.")

    def process_selected_file(self) -> None:
        file_path = self.selected_file.get()
        if not file_path:
            messagebox.showerror("שגיאה", "אנא בחרו קובץ PDF לעיבוד.")
            return

        try:
            self.process_button.config(state=tk.DISABLED)
            self.status_var.set("מעבד את הקובץ, נא להמתין...")
            self.root.update_idletasks()

            pdf_bytes = Path(file_path).read_bytes()
            if not pdf_bytes:
                raise ValueError("קובץ ה-PDF שסופק ריק.")

            text, extraction_mode = extract_pdf_content(pdf_bytes)
            labels_raw = parse_labels_from_text(text)
            # Filter out obvious OCR garbage (very long unbroken strings)
            labels_raw = [x for x in labels_raw if len(x) <= 40]

            if not labels_raw:
                raise ValueError("לא נמצאו תוויות בעמודים שנסרקו.")

            combined_counter = build_combined_counter(labels_raw)
            if not combined_counter:
                raise ValueError("לא נמצאו תוויות לאחר נרמול וספירה.")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"button_expanded_split_{timestamp}.xlsx"
            output_path = OUTPUT_DIR / output_name
            export_expanded(combined_counter, output_path)
            self._last_output_path = output_path

            self.populate_results(combined_counter)
            self.summary_var.set(
                " | ".join(
                    [
                        f"מצב חילוץ: {extraction_mode}",
                        f"סה\"כ תוויות מקוריות: {len(labels_raw)}",
                        f"סה\"כ שורות בקובץ: {sum(combined_counter.values())}",
                        f"קובץ פלט: {output_name}",
                    ]
                )
            )

            self.status_var.set("העיבוד הסתיים בהצלחה!")
            self.open_output_button.config(state=tk.NORMAL)
            messagebox.showinfo(
                "הצלחה",
                "העיבוד הסתיים בהצלחה! הקובץ נשמר בתיקיית outputs.",
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.status_var.set("אירעה שגיאה במהלך העיבוד.")
            messagebox.showerror("שגיאה", str(exc))
        finally:
            self.process_button.config(state=tk.NORMAL)

    def populate_results(self, counter: Counter) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        for label, count in sorted(counter.items(), key=lambda item: item[0]):
            self.tree.insert("", tk.END, values=(label, count))

    def open_output_location(self) -> None:
        if not self._last_output_path:
            return

        target = self._last_output_path
        directory = target.parent
        try:
            if sys.platform.startswith("win"):
                os.startfile(directory)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(directory)], check=False)
            else:
                subprocess.run(["xdg-open", str(directory)], check=False)
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("שגיאה", f"לא ניתן לפתוח את התיקייה: {exc}")


def main() -> None:
    root = tk.Tk()
    LabelExtractorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
