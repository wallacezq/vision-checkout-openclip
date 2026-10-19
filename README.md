# Vision Checkout (OpenCLIP + OpenVINO)

## Changelog

| Version | Changes |
|---|---|
| **0.0** | Baseline — Flask app with zero-shot CLIP classification, OpenVINO export, label management UI, cart & PDF receipt |
| **0.1** | Cached OV visual model + CLIP text model (no per-call reloads); smart label rebuild (only encodes new/changed labels); GPU auto-detection with CPU fallback; FP16 / INT8 precision toggle in UI; incremental add/remove/update label weights without full rebuild; bug fixes (merge conflict, camera not restarting after Settings) |
| **0.2** | Backend selector in the UI; selectable OpenVINO backends (GPU default, plus CPU and NPU); NPU static-shape export path for FP16 / INT8 inference |
| **0.3** | Optional **Intel DL Streamer** vision backend — the CLIP visual encoder runs through the `gvaclassify` GStreamer element; label (product) embeddings now cached in **`.safetensors`** instead of `.pth` (legacy `.pth` auto-migrated on first load) |
| **0.4** | DL Streamer backend rewritten to drive `gvaclassify` via the native **`gst-launch-1.0` CLI** (subprocess + JSON readback) — **no `gi`/PyGObject or `gstgva`** needed, works on any Python; **vision backend switchable in the UI** (OpenVINO ↔ DL Streamer) with an install/availability check; **exported-model availability indicator** + one-click **Export / Re-export**; **adjustable Unknown threshold** (slider + number field) in Settings |
| **0.5** | **Unified backend-device selector** — the single **Backend Device** (GPU/CPU/NPU) setting now drives the inference device for **both** the OpenVINO and DL Streamer vision backends, so the device is settable while DL Streamer is active (previously fixed to `DLSTREAMER_DEVICE`) |

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

- **Vision backend** -- Switch the CLIP **visual** encoder between **OpenVINO** (optimum-intel) and **DL Streamer** (`gvaclassify`) at runtime. Switching to DL Streamer first verifies the GStreamer / DL Streamer runtime is installed and reports a clear message if it is not.
- **Backend device** -- Select GPU, CPU, or NPU. This single selector sets the inference device for **both** vision backends (OpenVINO and DL Streamer). GPU remains the default on startup.
- **Precision** -- Toggle between FP16 and INT8; the matching OpenVINO model is exported on demand.
- **Model selection** -- Switch between available CLIP models:
  - MetaCLIP2 ViT-bigG-14 Worldwide (default)
  - Apple DFN5B CLIP ViT-H-14-378
  - Switching automatically regenerates label embeddings with a live progress bar.
- **Exported model** -- Shows whether the exported model artifacts exist for the current model/device/precision/backend, with a one-click **Export / Re-export** button (streamed progress).
- **Detection threshold** -- Adjust the "Unknown" confidence threshold with a slider **and** a direct number input (0–100%).
- **Label embeddings** -- An always-visible "Regenerate" button to rebuild `clip_zeroshot_cls.safetensors` at any time. Progress is streamed via Server-Sent Events (SSE).
- **Product label management** -- Add, edit, and delete product labels with:
  - Real-time search/filter
  - Input validation enforcing the `Category/Sub/Name` format (alphanumeric + hyphens, at least 2 `/`-separated segments)
  - Format tooltip and hint text
  - A warning banner when labels have changed and embeddings need regeneration

### Vision Scan

- Upload an image or use the device camera to capture a product photo.
- Top-5 predictions with confidence bars and "Unknown" detection when confidence is below the threshold (adjustable in **Settings → Detection**).
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
| `OV_DEVICE` | `GPU` | Startup default inference device (e.g. `GPU`, `CPU`, `NPU`). The in-app **Backend Device** selector overrides this at runtime for both vision backends. |
| `VISION_BACKEND` | `openvino` | Startup default for the visual encoder: `openvino` (optimum-intel wrapper) or `dlstreamer` (`gvaclassify`). Switchable at runtime in **Settings → Vision Backend**. |
| `DLSTREAMER_DEVICE` | `GPU` | Startup default device when the app boots with `VISION_BACKEND=dlstreamer`. At runtime the **Backend Device** selector controls the device for both backends. |
| `UNKNOWN_THRESHOLD` | `40.0` | Startup default for the Unknown threshold (%); adjustable at runtime in **Settings → Detection** |
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
wrapper. The alternative **DL Streamer** backend runs it through Intel DL
Streamer's `gvaclassify` element. You can switch between the two **at runtime**
from **Settings → Vision Backend**, or set `VISION_BACKEND=dlstreamer` as the
startup default.

Instead of the Python `gi` / PyGObject bindings, the backend drives DL Streamer
through the **native `gst-launch-1.0` CLI** as a subprocess. For each frame it
writes the image to a temporary PNG and runs:

```
filesrc location=<frame.png> ! decodebin ! videoconvert !
gvaclassify model=<image_encoder.xml> model-proc=<clip_image_encoder.json>
            device=<GPU|CPU|NPU> inference-region=full-frame !
gvametaconvert add-tensor-data=true format=json !
gvametapublish method=file file-path=<result.json> file-format=json !
fakesink
```

The raw `image_features` embedding is serialized to the JSON file and read back
with Python's standard-library `json` module. This means **no `gi`/PyGObject and
no `gstgva` package are required**, so the backend works under any modern Python.

- On first use the visual encoder is exported to a static-shape OpenVINO IR
  (`<model>-openclip-dlstreamer/<precision>/image_encoder.xml`) with a
  `pixel_values` input and `image_features` output. You can also trigger this
  from **Settings → Exported Model → Export / Re-export**.
- `gvaclassify` performs resize + normalization per
  `model_proc/clip_image_encoder.json`, runs inference, and attaches the
  embedding tensor to the frame.
- Switching to the DL Streamer backend first checks that the runtime is
  installed and that `gvaclassify` is discoverable, and reports a clear message
  in the UI if it is not.

### Requirements & environment

Only the native runtime is needed (not pip-installable): the Intel DL Streamer
runtime and GStreamer 1.0 with the DL Streamer plugins — providing
`gst-launch-1.0`, `gvaclassify`, `gvametaconvert` and `gvametapublish`. DL
Streamer runs on both **Linux** and **Windows** (and the `intel/dlstreamer`
container).

The backend discovers the runtime from environment variables set by the
installers and augments `PATH` / `GST_PLUGIN_PATH` for the subprocess:

| Variable | Purpose |
|---|---|
| `GSTREAMER_1_0_ROOT_MSVC_X86_64` / `GSTREAMER_ROOT` | GStreamer install root (locates `gst-launch-1.0` and plugins) |
| `DLSTREAMER_DIR` | Intel DL Streamer install root (plugins + dependencies) |
| `GST_LAUNCH` | Explicit full path to `gst-launch-1.0` (overrides discovery) |
| `DLSTREAMER_DEVICE` | Startup default device for `gvaclassify` (`GPU` / `CPU` / `NPU`); overridden at runtime by the **Backend Device** selector |
| `DLSTREAMER_TIMEOUT_S` | Per-frame subprocess timeout in seconds (default `120`) |

Paths are passed to `gst-launch` using forward slashes (Windows backslashes are
escape characters in the pipeline syntax), so native Windows paths work.

The rest of the app (Flask, OpenVINO via optimum-intel, OpenCLIP, safetensors)
runs on Windows as-is — only the optional `dlstreamer` vision backend needs the
DL Streamer / GStreamer runtime.

> **Note on PyGObject (Windows):** the historical `gi` / `gstgva` approach does
> not work reliably on Windows because the pip `PyGObject` wheels target the
> MSYS2 GLib stack while the DL Streamer runtime ships the MSVC GLib stack, and
> GStreamer's bundled `gi` is built only for an old Python. The subprocess
> approach above sidesteps PyGObject entirely.

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
| `GET` | `/model` | Available models + active selection, device, precision, vision backend |
| `GET` | `/model/switch_stream?model=<key>` | SSE stream: switch model + regenerate |
| `GET` | `/model/backend_stream?backend=<GPU\|CPU\|NPU>` | SSE: switch OpenVINO device |
| `GET` | `/model/precision_stream?precision=<FP16\|INT8>` | SSE: switch inference precision |
| `GET` | `/model/vision_backend_stream?vision_backend=<openvino\|dlstreamer>` | SSE: switch vision backend (checks DL Streamer availability first) |
| `GET` | `/model/export_status` | Whether exported model artifacts exist for the current config |
| `GET` | `/model/export_stream?force=<true\|false>` | SSE: export / re-export the OpenVINO model |
| `GET` | `/threshold` | Get the Unknown-detection threshold |
| `POST` | `/threshold` | Set the Unknown-detection threshold (0–100) |
| `GET` | `/prices` | List product prices |
| `POST` | `/prices` | Add a product price |
| `PUT` | `/prices` | Update a product price |
