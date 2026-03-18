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

import json
import logging
import os
import queue
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file

from bill_generator import generate_pdf
from classifier import ZeroShotClassifier, load_labels, save_labels
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
# Label management routes
# ---------------------------------------------------------------------------

@app.route("/labels", methods=["GET"])
def get_labels():
    """Return the current product label list."""
    return jsonify({"labels": load_labels()})


@app.route("/labels", methods=["POST"])
def add_label():
    """
    Add a new label. Request JSON: { "label": "Category/SubCat/Name" }
    Optionally set "regenerate": true to rebuild weights immediately.
    """
    data = request.get_json(force=True) or {}
    label = (data.get("label") or "").strip()
    if not label:
        return jsonify({"error": "label is required"}), 400

    labels = load_labels()
    if label in labels:
        return jsonify({"error": "Label already exists"}), 409

    labels.append(label)
    save_labels(labels)

    if data.get("regenerate"):
        classifier.rebuild_weights()

    return jsonify({"labels": labels, "added": label})


@app.route("/labels", methods=["PUT"])
def modify_label():
    """
    Modify an existing label.
    Request JSON: { "old_label": "…", "new_label": "…" }
    Optionally set "regenerate": true to rebuild weights immediately.
    """
    data = request.get_json(force=True) or {}
    old_label = (data.get("old_label") or "").strip()
    new_label = (data.get("new_label") or "").strip()
    if not old_label or not new_label:
        return jsonify({"error": "old_label and new_label are required"}), 400

    labels = load_labels()
    if old_label not in labels:
        return jsonify({"error": f"Label '{old_label}' not found"}), 404
    if new_label in labels:
        return jsonify({"error": f"Label '{new_label}' already exists"}), 409

    idx = labels.index(old_label)
    labels[idx] = new_label
    save_labels(labels)

    if data.get("regenerate"):
        classifier.rebuild_weights()

    return jsonify({"labels": labels, "modified": {"old": old_label, "new": new_label}})


@app.route("/labels", methods=["DELETE"])
def delete_label():
    """
    Delete a label. Request JSON: { "label": "…" }
    Optionally set "regenerate": true to rebuild weights immediately.
    """
    data = request.get_json(force=True) or {}
    label = (data.get("label") or "").strip()
    if not label:
        return jsonify({"error": "label is required"}), 400

    labels = load_labels()
    if label not in labels:
        return jsonify({"error": f"Label '{label}' not found"}), 404

    labels.remove(label)
    save_labels(labels)

    if data.get("regenerate"):
        classifier.rebuild_weights()

    return jsonify({"labels": labels, "deleted": label})


@app.route("/labels/regenerate", methods=["POST"])
def regenerate_weights():
    """Force regeneration of zero-shot classifier weights from current labels."""
    try:
        classifier.rebuild_weights()
        return jsonify({"status": "ok", "message": "Weights regenerated successfully"})
    except Exception as exc:
        logger.exception("Error regenerating weights")
        return jsonify({"error": str(exc)}), 500


@app.route("/labels/regenerate_stream")
def regenerate_weights_stream():
    """SSE endpoint that streams progress while regenerating weights."""
    progress_queue: queue.Queue = queue.Queue()

    def progress_cb(current: int, total: int, label: str) -> None:
        progress_queue.put({"current": current, "total": total, "label": label})

    def generate():
        import threading
        error_holder: list = []

        def do_rebuild():
            try:
                classifier.rebuild_weights(progress_cb=progress_cb)
            except Exception as exc:
                error_holder.append(str(exc))
            finally:
                progress_queue.put(None)  # sentinel

        t = threading.Thread(target=do_rebuild, daemon=True)
        t.start()

        while True:
            item = progress_queue.get()
            if item is None:
                if error_holder:
                    yield f"data: {json.dumps({'error': error_holder[0]})}\n\n"
                else:
                    yield f"data: {json.dumps({'done': True})}\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Automatic Vision-based Checkout server on port %d …", port)
    app.run(host="0.0.0.0", port=port, debug=False)
