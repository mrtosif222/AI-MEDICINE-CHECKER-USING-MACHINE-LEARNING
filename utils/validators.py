"""
validators.py
Input validation helpers used by app.py before passing user input to the
OCR, lookup, or interaction-checking modules.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import ALLOWED_IMAGE_EXTENSIONS, MAX_UPLOAD_SIZE_MB, SUPPORTED_LANGUAGES


def is_valid_medicine_name(name: str) -> bool:
    """Rejects empty, whitespace-only, or suspiciously short input."""
    if not name or not isinstance(name, str):
        return False
    return len(name.strip()) >= 2


def is_valid_image_file(filename: str) -> bool:
    """Checks the file extension is one of the allowed image types."""
    if not filename or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_IMAGE_EXTENSIONS


def is_valid_file_size(file_size_bytes: int) -> bool:
    """Checks the uploaded file doesn't exceed the configured size limit."""
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    return 0 < file_size_bytes <= max_bytes


def is_supported_language(lang_code: str) -> bool:
    """Checks a language code is one the app actually supports."""
    return lang_code in SUPPORTED_LANGUAGES


def validate_medicine_pair(medicine_a: str, medicine_b: str) -> tuple[bool, str]:
    """Validates a pair of medicine names for the /check endpoint.
    Returns (is_valid, error_message). error_message is empty if valid."""
    if not is_valid_medicine_name(medicine_a) or not is_valid_medicine_name(medicine_b):
        return False, "Please provide two valid medicine names (at least 2 characters each)."

    if medicine_a.strip().lower() == medicine_b.strip().lower():
        return False, "Please provide two different medicines to compare."

    return True, ""
