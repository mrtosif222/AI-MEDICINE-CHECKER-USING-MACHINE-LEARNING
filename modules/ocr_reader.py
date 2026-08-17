"""
ocr_reader.py
Extracts text (medicine names) from an uploaded prescription photo using
pytesseract (OCR).

Since prescriptions often list several medicines in one photo, this module
returns a cleaned list of likely medicine-name lines rather than just one
big text blob - each line is passed through medicine_lookup.py by the
caller (app.py) to confirm whether it's an actual known medicine.
"""

import pytesseract
from PIL import Image, ImageOps, ImageFilter
import re

# On Windows, uncomment and set this to your Tesseract install path if
# pytesseract can't find it automatically:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def _preprocess_image(image: Image.Image) -> Image.Image:
    """Cleanup + upscaling to improve OCR accuracy, especially on small or
    table-heavy prescription photos: convert to grayscale, upscale 2x
    (small text in tables is a common OCR failure point), boost contrast,
    and sharpen."""
    image = image.convert("L")  # grayscale

    width, height = image.size
    image = image.resize((width * 2, height * 2), Image.LANCZOS)

    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.SHARPEN)
    return image


def extract_text(image_path: str) -> str:
    """Runs OCR on the image and returns the raw extracted text.
    psm 6 (assume a single uniform block of text) works better than the
    default mode for prescription slips and tables."""
    image = Image.open(image_path)
    image = _preprocess_image(image)
    return pytesseract.image_to_string(image, config="--psm 6")


def extract_medicine_candidates(image_path: str) -> list[str]:
    """Returns a cleaned list of probable medicine-name lines from the
    photo. Filters out empty lines, pure numbers, and very short noise.
    Also strips numbering (e.g. "1)") and trailing dosage/duration text
    that OCR often merges onto the same line as the medicine name in
    table-style prescriptions."""
    raw_text = extract_text(image_path)

    candidates = []
    for line in raw_text.split("\n"):
        line = line.strip()

        if len(line) < 3:
            continue
        if re.fullmatch(r"[\d\s\-.,/]+", line):  # skip lines that are only numbers/punctuation
            continue

        # Strip leading OCR noise characters (quotes, brackets, bullets, etc.)
        line = re.sub(r"^[^a-zA-Z0-9]+", "", line)

        # Remove list numbering like "1)" or "2."
        line = re.sub(r"^\d+[).]\s*", "", line)

        # Remove common prescription noise like "Rx", "Tab.", "Sig:" prefixes
        line = re.sub(r"^(Rx|Tab\.?|Cap\.?|Sig:?|Dr\.?)\s*", "", line, flags=re.IGNORECASE)

        # Cut off trailing dosage/duration text that OCR often merges in
        # (e.g. "Medicine Name 1 Morning, 1 Night 10 Days" -> "Medicine Name")
        line = re.split(
            r"\s+\d+\s*(Morning|Night|Day|Days|Time|Tab|Cap)\b",
            line, maxsplit=1, flags=re.IGNORECASE
        )[0]

        line = line.strip()

        if line:
            candidates.append(line)

    return candidates


if __name__ == "__main__":
    # quick manual test - replace with an actual prescription image path to try
    test_image = "sample_prescription.jpg"
    try:
        print("Raw OCR text:")
        print(extract_text(test_image))
        print("\nMedicine candidates:")
        print(extract_medicine_candidates(test_image))
    except FileNotFoundError:
        print(f"No test image found at '{test_image}'. "
              f"Place a sample prescription photo there to test this module.")
