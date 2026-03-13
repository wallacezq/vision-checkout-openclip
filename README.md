# Vision Checkout (OpenCLIP + OpenVINO)

Automatic vision-based retail checkout demo: upload a product image, run **zero-shot** classification (CLIP/OpenCLIP accelerated with **OpenVINO**), show the top predictions, and generate a **PDF receipt**.

## What’s in this repo

At a high level, this is a single Flask web app with three main parts:

1. **Web server + UI** (`app.py` + `templates/`)
2. **Zero-shot vision classifier** (`classifier.py`)
3. **Billing / pricing + PDF receipt** (`product_db.py`, `bill_generator.py`)

## Architecture (high level)

### Request flow

1. **User opens the UI** (`GET /`)  
   The app serves a static HTML page (`templates/webpage_design.html`).

2. **User uploads an image** (`POST /upload`)  
   - Flask receives a multipart file upload (`image`).
   - The image is saved into `uploads/` and decoded with OpenCV.
   - The classifier runs zero-shot inference and returns **top-5 predictions**.
   - If the top-1 confidence is below a configurable threshold, the result is treated as **Unknown** and no bill item is created.
   - Otherwise, the predicted class’ `short_name` (last path segment) is used to look up a unit price in a CSV-backed product DB and a bill line item is returned.

3. **User confirms or overrides the product** (`POST /confirm_product`)  
   If the user selects one of the other top predictions, the server builds a bill item from the selected `short_name`.

4. **Payment pages** (`GET /payment`, `GET /payment/success`)  
   Served as static HTML from `templates/`.

5. **Receipt download** (`POST /download_bill`)  
   The frontend posts the full bill + totals; the server generates a **PDF** receipt using ReportLab and returns it as a file download.

### Components

#### `app.py` (Flask entry point)
- Initializes shared services once at startup:
  - `ZeroShotClassifier(...)`
  - `ProductDatabase(product_prices.csv)`
- Exposes REST-ish endpoints for upload/classify, confirm product, and PDF receipt generation.

#### `classifier.py` (OpenCLIP/OpenVINO zero-shot classifier)
- Maintains a fixed label taxonomy (`LABELS`) like `Fruit/Apple/Granny-Smith`.
- Builds a *zero-shot weight matrix* from text prompts (`CLASS_TEMPLATES`) and caches it to `clip_zeroshot_cls.pth`.
- Uses **Optimum Intel OpenVINO** wrappers to export/load an OpenVINO OpenCLIP model and run visual inference.
- Produces `Prediction` objects: `{label, short_name, confidence, is_unknown}`.

#### `product_db.py` (CSV price lookup)
- Loads `product_prices.csv` (expected columns: Product, Price).
- Builds bill line items with `Unit_Price` and `Total`.

#### `bill_generator.py` (PDF)
- Generates a formatted receipt (currency RM) with **SST 6%**.

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
- `product_prices.csv` present at the repo root (used by `ProductDatabase`)

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

Then open:

- http://localhost:5000

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---:|---|
| `PORT` | `5000` | Flask port |
| `OV_DEVICE` | `GPU` | OpenVINO device string (e.g. `CPU`, `GPU`) |
| `UNKNOWN_THRESHOLD` | `40.0` | If top-1 confidence (%) is below this, treat as Unknown |
| `QUANTIZE` | `false` | If `true`, export/use an INT8 OpenVINO model (otherwise FP16) |

Example (CPU + higher threshold):

```bash
export OV_DEVICE=CPU
export UNKNOWN_THRESHOLD=50
python app.py
```

## Notes / troubleshooting

- **First run can be slow**: the app may export an OpenVINO model and/or build zero-shot weights (cached to `clip_zeroshot_cls.pth`).
- If you see errors about OpenVINO GPU, try `OV_DEVICE=CPU`.
- If `product_prices.csv` is missing or a predicted `short_name` doesn’t exist in the CSV, `/upload` may return `bill_item: null` (or `/confirm_product` can return a 404).

## API summary

- `GET /` → main UI
- `POST /upload` → `{ top5: [...], bill_item, is_unknown, threshold }`
- `POST /confirm_product` → `{ bill_item }`
- `GET /payment` → payment page
- `GET /payment/success` → success page
- `POST /download_bill` → PDF receipt download
