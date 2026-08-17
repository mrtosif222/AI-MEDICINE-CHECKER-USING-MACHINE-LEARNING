"""
app.py
Main Flask server. Connects the OCR, medicine lookup, and interaction
checker modules together and exposes them as web routes.

Routes:
    GET  /              - health check / home page
    POST /check          - check interaction between two medicine names (JSON)
    POST /upload          - upload a prescription photo, run OCR, return
                            candidate medicine names found in it
    POST /medicine-info    - look up details for a single medicine name (JSON)
"""

import os
import sys

# Make sure the modules/ folder is importable regardless of where this
# script is launched from (its imports use plain names like
# "from medicine_lookup import find_medicine").
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from interaction_checker import check_interaction
from medicine_lookup import find_medicine
from ocr_reader import extract_medicine_candidates

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "AI Medicine Interaction Checker server is running"})


@app.route("/check", methods=["POST"])
def check():
    """Expects JSON: {"medicine_a": "...", "medicine_b": "..."}"""
    data = request.get_json(silent=True)
    if not data or "medicine_a" not in data or "medicine_b" not in data:
        return jsonify({"error": "Please provide medicine_a and medicine_b"}), 400

    result = check_interaction(data["medicine_a"], data["medicine_b"])
    return jsonify(result)


@app.route("/medicine-info", methods=["POST"])
def medicine_info():
    """Expects JSON: {"name": "..."}"""
    data = request.get_json(silent=True)
    if not data or "name" not in data:
        return jsonify({"error": "Please provide a medicine name"}), 400

    result = find_medicine(data["name"])
    if result is None:
        return jsonify({"found": False, "message": "Medicine not found in our database"}), 404

    return jsonify({"found": True, **result})


@app.route("/upload", methods=["POST"])
def upload():
    """Expects a multipart form with a file field named 'prescription'.
    Returns the list of candidate medicine names OCR found in the photo -
    the frontend should let the user confirm/edit these before checking
    interactions (OCR is not always accurate, especially on handwriting)."""
    if "prescription" not in request.files:
        return jsonify({"error": "No file uploaded (expected field name 'prescription')"}), 400

    file = request.files["prescription"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": "Only png/jpg/jpeg files are supported"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    candidates = extract_medicine_candidates(save_path)

    return jsonify({
        "candidates": candidates,
        "note": "OCR results may not be fully accurate, especially for "
                "handwritten prescriptions. Please confirm or edit these "
                "names before checking interactions."
    })


if __name__ == "__main__":
    app.run(debug=True)
