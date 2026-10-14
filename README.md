# Vision Checkout (OpenCLIP + OpenVINO)

## Changelog

| Version | Changes |
|---|---|
| **0.0** | Baseline — Flask app with zero-shot CLIP classification, OpenVINO export, label management UI, cart & PDF receipt |
| **0.1** | Cached OV visual model + CLIP text model (no per-call reloads); smart label rebuild (only encodes new/changed labels); GPU auto-detection with CPU fallback; FP16 / INT8 precision toggle in UI; incremental add/remove/update label weights without full rebuild; bug fixes (merge conflict, camera not restarting after Settings) |
| **0.2** | Backend selector in the UI; selectable OpenVINO backends (GPU default, plus CPU and NPU); NPU static-shape export path for FP16 / INT8 inference |
| **0.3** | Optional **Intel DL Streamer** vision backend — the CLIP visual encoder runs through the `gvaclassify` GStreamer element; label (product) embeddings now cached in **`.safetensors`** instead of `.pth` (legacy `.pth` auto-migrated on first load) |

Automatic vision-based retail checkout demo: upload a product image, run **zero-shot** classification (CLIP/OpenCLIP accelerated with **OpenVINO**), show the top predictions, and generate a **PDF receipt**.

## What's in this repo

A single Flask web app with three main parts:

1. **Web server + UI** (`app.py` + `templates/`)
2. **Zero-shot vision classifier** (`classifier.py`)
3. **Billing / pricing + PDF receipt** (`product_db.py`, `bill_generator.py`)

## Architecture

### Request flow

1. **User opens the UI** (`GET /`) -- serves `templates/webpage_design.html`.

2. **User uploads an image** (`POST /upload`)
   - Flask receives a multipart file upload (`image`).
   - The image is saved into `uploads/` and decoded with OpenCV.
   - The classifier runs zero-shot inference and returns **top-5 predictions**.
   - If the top-1 confidence is below a configurable threshold, the result is treated as **Unknown** and no bill item is created.
   - Otherwise, the predicted class's `short_name` (last path segment) is used to look up a unit price in the CSV-backed product DB and a bill line item is returned.

3. **User confirms or overrides the product** (`POST /confirm_product`) -- builds a bill item from the selected `short_name`.

4. **Payment pages** (`GET /payment`, `GET /payment/success`) -- styled dark-themed HTML matching the main UI.

5. **Receipt download** (`POST /download_bill`) -- generates a **PDF** receipt via ReportLab and returns it as a file download.

### Key files

| File | Purpose |
|---|---|
| `app.py` | Flask entry point. Initializes `ZeroShotClassifier` and `ProductDatabase`, exposes all REST endpoints. |
| `classifier.py` | OpenCLIP/OpenVINO zero-shot classifier. Loads labels from `labels.json`, supports multiple CLIP models via `MODEL_REGISTRY`, caches weights to `clip_zeroshot_cls.safetensors`. Can run the visual encoder via the OpenVINO wrapper or via DL Streamer `gvaclassify`. |
| `dlstreamer_encoder.py` | DL Streamer vision backend. Runs the CLIP visual encoder by invoking the native `gst-launch-1.0` CLI (`filesrc ! decodebin ! videoconvert ! gvaclassify ! gvametaconvert ! gvametapublish`), then reads the image-feature tensor back from the published JSON. Uses **no** `gi`/PyGObject bindings, so it works on any Python under the MSVC GStreamer / DL Streamer runtime. |
| `model_proc/clip_image_encoder.json` | `gvaclassify` model-proc describing CLIP preprocessing (resize, RGB, mean/std) and raw embedding output. |
| `labels.json` | Product label taxonomy as a JSON array (e.g. `Fruit/Apple/Granny-Smith`). Editable at runtime via the Settings UI. |
| `model_config.json` | Persists which CLIP model is currently active (auto-created on first model switch). |
| `product_db.py` | CSV price lookup. Loads `product_prices.csv` (columns: Product, Price). |
| `bill_generator.py` | Generates PDF receipts with SST 6% (currency RM). |
| `templates/webpage_design.html` | Main checkout UI (dark theme, single-page app). |
| `templates/payment.html` | Payment form page. |
| `templates/success.html` | Payment success page. |

## Features

### Settings page

Accessible via the gear icon in the top bar. Contains:

- **OpenVINO backend** -- Select GPU, CPU, or NPU. GPU remains the default on startup.
- **Model selection** -- Switch between available CLIP models:
  - MetaCLIP2 ViT-bigG-14 Worldwide (default)
  - Apple DFN5B CLIP ViT-H-14-378
  - Switching automatically regenerates label embeddings with a live progress bar.
- **Label embeddings** -- An always-visible "Regenerate" button to rebuild `clip_zeroshot_cls.safetensors` at any time. Progress is streamed via Server-Sent Events (SSE).
- **Product label management** -- Add, edit, and delete product labels with:
  - Real-time search/filter
  - Input validation enforcing the `Category/Sub/Name` format (alphanumeric + hyphens, at least 2 `/`-separated segments)
  - Format tooltip and hint text
  - A warning banner when labels have changed and embeddings need regeneration

### Vision Scan

- Upload an image or use the device camera to capture a product photo.
- Top-5 predictions with confidence bars and "Unknown" detection when confidence is below threshold.
- Click any prediction to override the selection and update the cart.

### Checkout

- Cart with quantity tracking, per-unit pricing, SST (6%) calculation.
- PDF receipt download and payment flow.

## Technology stack

- **Backend**: Python + Flask
- **Computer Vision**: OpenCV, NumPy
- **ML / Zero-shot**:
  - PyTorch
  - `open_clip`
  - Hugging Face Transformers (image processor)
  - `optimum[openvino]` (OpenVINO runtime + model export wrappers)
- **Data**: Pandas (CSV price table)
- **PDF**: ReportLab
- **(Optional)**: `pyngrok` for exposing localhost

See `requirements.txt` for the full dependency list.

## Setup

### Prerequisites

- Python 3.10+ recommended
- A working environment for PyTorch + OpenVINO (CPU works; GPU depends on your OpenVINO setup)
- `product_prices.csv` present at the repo root

### Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Running the app

```bash
python app.py
```

Then open http://localhost:5000

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---:|---|
| `PORT` | `5000` | Flask port |
| `OV_DEVICE` | `GPU` | OpenVINO backend string (e.g. `GPU`, `CPU`, `NPU`) |
| `VISION_BACKEND` | `openvino` | Visual encoder path: `openvino` (optimum-intel wrapper) or `dlstreamer` (`gvaclassify`) |
| `DLSTREAMER_DEVICE` | `GPU` | Device string passed to `gvaclassify` when `VISION_BACKEND=dlstreamer` |
| `UNKNOWN_THRESHOLD` | `40.0` | If top-1 confidence (%) is below this, treat as Unknown |
| `QUANTIZE` | `false` | If `true`, export/use an INT8 OpenVINO model (otherwise FP16) |

Example (CPU + higher threshold):

```bash
export OV_DEVICE=CPU
export UNKNOWN_THRESHOLD=50
python app.py
```

If you want to use NPU in INT8 mode, install `nncf` as well so the static-shape export can compress the exported OpenVINO graphs.

## DL Streamer vision backend

By default the CLIP **visual** encoder runs through the optimum-intel OpenVINO
wrapper. Setting `VISION_BACKEND=dlstreamer` instead runs it through Intel DL
Streamer's `gvaclassify` GStreamer element:

```
appsrc ! videoconvert ! gvaclassify model=image_encoder.xml \
    model-proc=model_proc/clip_image_encoder.json \
    device=GPU inference-region=full-frame ! appsink
```

- On first use the visual encoder is exported to a static-shape OpenVINO IR
  (`<model>-openclip-dlstreamer/<precision>/image_encoder.xml`) with a
  `pixel_values` input and `image_features` output.
- `gvaclassify` performs resize + normalization per
  `model_proc/clip_image_encoder.json`, runs inference, and attaches the
  embedding tensor to the frame; it is read back via the `gstgva` bindings.

```bash
export VISION_BACKEND=dlstreamer
export DLSTREAMER_DEVICE=GPU
python app.py
```

On Windows (PowerShell):

```powershell
$env:VISION_BACKEND = "dlstreamer"
$env:DLSTREAMER_DEVICE = "GPU"
python app.py
```

Requirements (not pip-installable): the Intel DL Streamer runtime, GStreamer 1.0
with the DL Streamer plugins (`gvaclassify`), the `gi` bindings (`Gst`,
`GstApp`), and the `gstgva` Python package. DL Streamer runs on both **Linux**
and **Windows** (as well as the `intel/dlstreamer` container). On Windows,
install the DL Streamer / GStreamer runtime and make sure its `bin` folder is on
`PATH` and the PyGObject (`gi`) bindings are importable; model and model-proc
paths are passed as element properties, so native Windows paths work.

The rest of the app (Flask, OpenVINO via optimum-intel, OpenCLIP, safetensors)
runs on Windows as-is — only the optional `dlstreamer` vision backend needs the
DL Streamer/GStreamer runtime.

> **Why only the visual encoder?** `gvaclassify` is an **image** inference
> element: it takes video frames in and runs a model on them. The product
> (label) embeddings are produced by the CLIP **text** encoder from label
> prompts, which has no image input and therefore cannot be driven by
> `gvaclassify`. Regenerating product embeddings continues to use the OpenVINO /
> OpenCLIP text encoder directly (see below).

### Regenerating product embeddings

The product embeddings are the text-derived zero-shot class weights cached in
`clip_zeroshot_cls.safetensors`. They are (re)built from the label prompts by
the CLIP text encoder and are independent of the chosen **vision** backend.
`gvaclassify` cannot generate them, so regeneration always runs the text encoder
directly via OpenCLIP / OpenVINO (`POST /labels/regenerate` or the Settings UI).

## Notes / troubleshooting

- **First run can be slow**: the app may export an OpenVINO model and/or build zero-shot weights (cached to `clip_zeroshot_cls.safetensors`).
- If you see errors about OpenVINO GPU, try `OV_DEVICE=CPU`.
- If NPU INT8 export fails, install `nncf` or switch to FP16.
- If `product_prices.csv` is missing or a predicted `short_name` doesn't exist in the CSV, `/upload` may return `bill_item: null` (or `/confirm_product` can return a 404).
- **Model switching** exports a new OV model directory if it doesn't already exist, which can also take time on first use.

## API summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main UI |
| `POST` | `/upload` | Classify image, return top-5 + bill item |
| `POST` | `/confirm_product` | Override product selection |
| `GET` | `/payment` | Payment page |
| `GET` | `/payment/success` | Success page |
| `POST` | `/download_bill` | PDF receipt download |
| `GET` | `/labels` | Get current label list |
| `POST` | `/labels` | Add a label |
| `PUT` | `/labels` | Modify a label |
| `DELETE` | `/labels` | Delete a label |
| `POST` | `/labels/regenerate` | Rebuild `clip_zeroshot_cls.safetensors` |
| `GET` | `/labels/regenerate_stream` | SSE stream of regeneration progress |
| `GET` | `/model` | Available models + active selection |
| `GET` | `/model/switch_stream?model=<key>` | SSE stream: switch model + regenerate |
