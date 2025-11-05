import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
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


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET", "dev-secret")


def is_docupipe_configured() -> bool:
    return bool(os.getenv("docupipe_API_KEY") and os.getenv("docupipe_ENDPOINT"))


def clean_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = re.sub(r"\s+", " ", value.strip())
    return value or None


READING_LIGHT_VARIANTS = {
    "מ",
    "מ.",
    "מ קריאה",
    "מ. קריאה",
    "מ קריאה.",
    "מ.קריאה",
}


ROOM_PREFIX_PATTERNS = [
    (re.compile(r"^\s*כ\.?\s*חדר\s*$"), "כ.חדר"),
    (re.compile(r"^\s*ח\.?\s*חדר\s*$"), "ח.חדר"),
]


def normalize_label(label: str) -> str:
    label = re.sub(r"\s+", " ", label.strip())
    if not label:
        return label

    # Fix known spelling issues
    label = label.replace("ילון", "וילון")

    # Reading light prefix normalization
    if label in READING_LIGHT_VARIANTS:
        label = "מ.קריאה"

    # Mirror & Hidden combinations
    if "מראה" in label and "נסתר" in label:
        label = "מראה ונסתרת"

    # Room prefixes
    for pattern, replacement in ROOM_PREFIX_PATTERNS:
        if pattern.match(label):
            label = replacement
            break

    # Normalize פס לד variations
    normalized_for_pas_led = re.sub(r"\s+", "", label)
    pas_led_variants = {"פסלד", "לדפס"}
    if normalized_for_pas_led in pas_led_variants:
        label = "פס לד"
        skip_flip = True
    else:
        skip_flip = False
        if label.replace(" ", "") == "פסלד":
            label = "פס לד"
            skip_flip = True
        elif label.replace(" ", "") == "לדפס":
            label = "פס לד"
            skip_flip = True

    # Ensure standard פס לד even if already spaced
    if label in {"לד פס", "פס לד"}:
        label = "פס לד"
        skip_flip = True

    # Handle standalone מ as prefix before a word (e.g., "מ קריאה")
    label = re.sub(r"^מ\s*\.\s*קריאה$", "מ.קריאה", label)
    label = re.sub(r"^מ\s*קריאה$", "מ.קריאה", label)

    words = label.split(" ")
    if len(words) == 2 and not skip_flip:
        label = " ".join(reversed(words))

    return re.sub(r"\s+", " ", label.strip())


def collect_labels(data: Dict[str, Any], include_options: bool) -> List[str]:
    labels: List[str] = []
    rooms = data.get("rooms")
    if not isinstance(rooms, list):
        raise ValueError("מבנה הקובץ אינו תקין - חסר שדה 'rooms'.")

    for room in rooms:
        if not isinstance(room, dict):
            continue

        features = room.get("features", [])
        labels.extend(filter(None, (clean_label(f) for f in ensure_iterable(features))))

        options = room.get("options", [])
        if not include_options:
            continue
        for option in ensure_iterable(options):
            name = None
            selected = True
            if isinstance(option, dict):
                name = clean_label(option.get("name") or option.get("label"))
                selected = bool(option.get("selected"))
            else:
                name = clean_label(option)
            if not name:
                continue
            if isinstance(option, dict) and not selected:
                continue
            labels.append(name)
    return labels


def ensure_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return value
    return [value]


def apply_counting_rules(labels: List[str]) -> Tuple[Counter, Counter]:
    raw_counter = Counter()
    normalized_counter = Counter()

    for raw in labels:
        raw_counter[raw] += 1
        normalized = normalize_label(raw)
        if normalized in {"וילון קיר", "קיר וילון"}:
            normalized_counter["וילון"] += 1
            normalized_counter["קיר"] += 1
        elif normalized in {"וילון לילה", "לילה וילון"}:
            normalized_counter["וילון"] += 1
            normalized_counter["לילה"] += 1
        else:
            normalized_counter[normalized] += 1

    return raw_counter, normalized_counter


def sort_counter(counter: Counter) -> List[Tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def export_counts(counter: Counter, path: Path, value_header: str, sheet_name: str) -> None:
    rows = sort_counter(counter)
    df = pd.DataFrame(rows, columns=[value_header, "מופעים"])
    df.to_excel(path, index=False, sheet_name=sheet_name)


def export_expanded(counter: Counter, path: Path) -> None:
    rows: List[Dict[str, str]] = []
    for label, count in sort_counter(counter):
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


def map_to_internal_schema(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, list):
        return {"rooms": payload}

    if not isinstance(payload, dict):
        raise ValueError("תגובה לא תקינה מהשירות - פורמט לא נתמך.")

    if "rooms" in payload and isinstance(payload["rooms"], list):
        return {"rooms": payload["rooms"]}

    # Check nested under data
    data = payload.get("data") if isinstance(payload.get("data"), dict) else None
    if data and isinstance(data.get("rooms"), list):
        return {"rooms": data["rooms"]}

    # Some services return results under 'result'
    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("rooms"), list):
        return {"rooms": result["rooms"]}

    # Attempt to detect direct list of rooms
    if isinstance(payload.get("items"), list):
        return {"rooms": payload["items"]}

    raise ValueError("לא ניתן למפות את תגובת השירות למבנה המצופה.")


def parse_json_file(file_storage) -> Dict[str, Any]:
    try:
        data = json.load(file_storage)
    except json.JSONDecodeError as exc:
        raise ValueError(f"קובץ JSON לא תקין: {exc}") from exc
    return map_to_internal_schema(data)


def call_docupipe(pdf_file) -> Dict[str, Any]:
    api_key = os.getenv("docupipe_API_KEY")
    endpoint = os.getenv("docupipe_ENDPOINT")
    if not api_key or not endpoint:
        raise RuntimeError("נדרש מפתח Docupipe או העלאת קובץ JSON.")

    try:
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (pdf_file.filename, pdf_file.stream, pdf_file.mimetype or "application/pdf")},
            timeout=60,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"שגיאת תקשורת מול Docupipe: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Docupipe החזיר שגיאה ({response.status_code}): {response.text}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Docupipe החזיר תגובה שאינה JSON תקין.") from exc

    return map_to_internal_schema(payload)


def save_results(raw_counter: Counter, normalized_counter: Counter, include_raw: bool) -> Dict[str, Any]:
    counts_filename = "counts.xlsx"
    expanded_filename = "button_expanded_split.xlsx"
    raw_filename = "button_counts_exact.xlsx"

    counts_path = OUTPUT_DIR / counts_filename
    expanded_path = OUTPUT_DIR / expanded_filename

    export_counts(normalized_counter, counts_path, "ערך", sheet_name="counts")
    export_expanded(normalized_counter, expanded_path)

    files = [
        {"label": "counts.xlsx", "name": counts_filename},
        {"label": "button_expanded_split.xlsx", "name": expanded_filename},
    ]

    if include_raw:
        raw_path = OUTPUT_DIR / raw_filename
        export_counts(raw_counter, raw_path, "ערך (Raw)", sheet_name="raw_counts")
        files.append({"label": "button_counts_exact.xlsx", "name": raw_filename})

    top_20 = [
        {"label": label, "count": count}
        for label, count in sort_counter(normalized_counter)[:20]
    ]
    return {"files": files, "top_20": top_20}


@app.route("/")
def index():
    results = session.pop("results", None)
    docupipe_ready = is_docupipe_configured()
    return render_template("index.html", results=results, docupipe_ready=docupipe_ready)


@app.route("/process", methods=["POST"])
def process():
    json_file = request.files.get("json_file")
    pdf_file = request.files.get("pdf_file")
    include_options = request.form.get("include_options") == "on"
    include_raw = request.form.get("include_raw") == "on"

    try:
        if json_file and json_file.filename:
            json_file.stream.seek(0)
            internal_data = parse_json_file(json_file.stream)
        elif pdf_file and pdf_file.filename:
            if not is_docupipe_configured():
                raise RuntimeError("נא להעלות קובץ JSON או להגדיר פרטי Docupipe בקובץ הסביבה.")
            try:
                pdf_file.stream.seek(0)
            except (AttributeError, OSError):
                pass
            internal_data = call_docupipe(pdf_file)
        else:
            raise ValueError("נא להעלות קובץ JSON או PDF.")

        labels = collect_labels(internal_data, include_options=include_options)
        if not labels:
            raise ValueError("לא נמצאו תוויות לעיבוד בקובץ שסופק.")

        raw_counter, normalized_counter = apply_counting_rules(labels)
        results = save_results(raw_counter, normalized_counter, include_raw)
        session["results"] = results
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
