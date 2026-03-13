"""
app.py
Flask entry point for the Smart Retail Checkout system.

Routes:
  GET  /                  → main checkout UI
  POST /upload            → classify image, return top-5 predictions + bill
  POST /confirm_product   → user confirms / overrides the product choice
  GET  /payment           → payment page
  GET  /payment/success   → success page
  POST /download_bill     → generate & return PDF receipt
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file

from bill_generator import generate_pdf
from classifier import ZeroShotClassifier
from product_db import ProductDatabase

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ---------------------------------------------------------------------------
# Shared services (loaded once at startup)
# ---------------------------------------------------------------------------
logger.info("Initialising classifier …")
classifier = ZeroShotClassifier(
    quantize=os.environ.get("QUANTIZE", "false").lower() == "true",
    ov_device=os.environ.get("OV_DEVICE", "GPU"),
)
UNKNOWN_THRESHOLD = float(os.environ.get("UNKNOWN_THRESHOLD", 40.0))

logger.info("Loading product database …")
product_db = ProductDatabase(BASE_DIR / "product_prices.csv")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return send_file(BASE_DIR / "templates" / "webpage_design.html")


@app.route("/upload", methods=["POST"])
def upload_image():
    """
    Accepts a multipart image, runs classification, and returns:
    {
      "top5": [
        {"label": "…", "short_name": "…", "confidence": 91.2},
        …
      ],
      "bill_item": {"Product": "…", "Quantity": 1, "Unit_Price": …, "Total": …} | null
    }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Save & decode image
        save_path = UPLOAD_FOLDER / file.filename
        file.save(save_path)
        img = cv2.imread(str(save_path))
        if img is None:
            return jsonify({"error": "Could not decode image"}), 422

        # Run zero-shot classification
        predictions = classifier.classify(img, threshold=UNKNOWN_THRESHOLD)

        is_unknown = predictions[0].is_unknown
        top5 = [
            {
                "label": p.label,
                "short_name": p.short_name,
                "confidence": p.confidence,
            }
            for p in predictions
        ]

        # Suppress bill item when confidence is below threshold
        if is_unknown:
            bill_item = None
        else:
            best = predictions[0]
            bill_item = product_db.build_bill(best.short_name)

        return jsonify({
            "top5": top5,
            "bill_item": bill_item,
            "is_unknown": is_unknown,
            "threshold": UNKNOWN_THRESHOLD,
        })

    except Exception as exc:
        logger.exception("Error in /upload")
        return jsonify({"error": str(exc)}), 500


@app.route("/confirm_product", methods=["POST"])
def confirm_product():
    """
    Called when the user overrides the top-1 prediction with one of the
    other top-5 candidates.

    Request JSON: { "short_name": "Granny-Smith" }
    Response:     { "bill_item": { … } }
    """
    data = request.get_json(force=True)
    short_name = (data or {}).get("short_name", "")
    if not short_name:
        return jsonify({"error": "short_name is required"}), 400

    bill_item = product_db.build_bill(short_name)
    if bill_item is None:
        return jsonify({"error": f"Product '{short_name}' not found in database"}), 404

    return jsonify({"bill_item": bill_item})


@app.route("/payment")
def payment_page():
    return send_file(BASE_DIR / "templates" / "payment.html")


@app.route("/payment/success")
def payment_success():
    return send_file(BASE_DIR / "templates" / "success.html")


@app.route("/download_bill", methods=["POST"])
def download_bill():
    """
    Request JSON: { "bill": [...], "total_price": float, "total_items": int }
    Returns: PDF file attachment
    """
    data = request.get_json(force=True) or {}
    bill = data.get("bill", [])
    total_price = float(data.get("total_price", 0))
    total_items = int(data.get("total_items", 0))
    subtotal = data.get("subtotal")
    sst      = data.get("sst")
    if subtotal is not None:
        subtotal = float(subtotal)
    if sst is not None:
        sst = float(sst)

    if not bill:
        return jsonify({"error": "No bill items provided"}), 400

    pdf_buffer = generate_pdf(bill, total_price, total_items, subtotal=subtotal, sst=sst)
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name="receipt.pdf",
        mimetype="application/pdf",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Automatic Vision-based Checkout server on port %d …", port)
    app.run(host="0.0.0.0", port=port, debug=False)
