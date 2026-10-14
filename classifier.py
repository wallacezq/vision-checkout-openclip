"""
classifier.py
Zero-shot image classifier using OpenCLIP / Apple DFN5B-CLIP.
Responsible for loading model weights and running inference.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as _safetensors_load, save_file as _safetensors_save
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, AutoModel, AutoImageProcessor
import open_clip

from optimum.intel.openvino import (
    OVModelOpenCLIPForZeroShotImageClassification,
    OVModelOpenCLIPVisual,
    OVWeightQuantizationConfig,
)

# ---------------------------------------------------------------------------
# Monkey-patch: fix "multiple values for argument 'allow_new'" on Python 3.14+
# Python 3.14's functools.partial (vectorcall) validates keyword arguments at
# the C level before dispatching to __init__, so patching __init__ alone does
# not help. Instead, replace every functools.partial-based NORMALIZED_CONFIG_CLASS
# with a plain callable, and patch with_args to stop creating partials.
#
# NOTE: The replacement must NOT be a plain function/lambda, because Python
# functions are descriptors — when stored as a class attribute and accessed
# via an instance (self.NORMALIZED_CONFIG_CLASS), they would be bound as a
# method, injecting `self` as an extra first argument.  functools.partial
# objects are *not* descriptors, so the originals never had that problem.
# We therefore use a simple callable class (_ConfigFactory) which also is
# not a descriptor.
# ---------------------------------------------------------------------------
import functools as _functools
from optimum.utils.normalized_config import NormalizedConfig as _NormalizedConfig


class _ConfigFactory:
    """Non-descriptor callable that replaces functools.partial for NormalizedConfig."""
    __slots__ = ("_func", "_kw")

    def __init__(self, func, **kw):
        self._func = func
        self._kw = kw

    def __call__(self, config):
        return self._func(config, **self._kw)


# 1) Patch with_args so future calls produce _ConfigFactory instead of partial.
@classmethod
def _safe_with_args(cls, allow_new=False, **kwargs):
    return _ConfigFactory(cls, allow_new=allow_new, **kwargs)

_NormalizedConfig.with_args = _safe_with_args

# 2) Replace already-created functools.partial NORMALIZED_CONFIG_CLASS attrs.
for _mod_path in (
    "optimum.exporters.onnx.model_configs",
    "optimum.exporters.openvino.model_configs",
):
    try:
        import importlib
        _mod = importlib.import_module(_mod_path)
    except ImportError:
        continue
    for _name in dir(_mod):
        _obj = getattr(_mod, _name, None)
        if isinstance(_obj, type):
            _ncc = _obj.__dict__.get("NORMALIZED_CONFIG_CLASS")
            if isinstance(_ncc, _functools.partial):
                setattr(_obj, "NORMALIZED_CONFIG_CLASS",
                        _ConfigFactory(_ncc.func, **_ncc.keywords))
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Product label taxonomy — loaded from labels.json
# ---------------------------------------------------------------------------
LABELS_PATH = Path(__file__).parent / "labels.json"


def load_labels() -> list[str]:
    """Load product labels from the JSON file."""
    with open(LABELS_PATH, "r") as f:
        return json.load(f)


def save_labels(labels: list[str]) -> None:
    """Save product labels to the JSON file."""
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=4)


LABELS: list[str] = load_labels()

# Text prompts used to build zero-shot classifier weights
CLASS_TEMPLATES: list[str] = [
    "a photo of a {label} in bounding box.",
    "a product photo of a {label} in bounding box.",
    "a retail image of a {label} in bounding box.",
    "a picture of a {label} in bounding box.",
    "an image of a {label} in bounding box.",
    #"{label} for sale.",
    "a photo of {label} in red bounding box."
]

# ---------------------------------------------------------------------------
# Available model configurations
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {
    "metaclip2-vit-bigG-14": {
        "display_name": "MetaCLIP2 ViT-bigG-14 Worldwide",
        "openclip_model_id": "ViT-bigG-14-worldwide",
        "model_id": "timm/vit_gigantic_patch14_clip_378.metaclip2_worldwide",
        "pretrained": "metaclip2_worldwide",
    },
    "dfn5b-vit-H-14": {
        "display_name": "Apple DFN5B CLIP ViT-H-14-378",
        "openclip_model_id": "ViT-H-14-378-quickgelu",
        "model_id": "apple/DFN5B-CLIP-ViT-H-14-378",
        "pretrained": "dfn5b",
    },
}

MODEL_CONFIG_PATH = Path(__file__).parent / "model_config.json"


def load_model_config() -> str:
    """Return the currently selected model key."""
    if MODEL_CONFIG_PATH.exists():
        with open(MODEL_CONFIG_PATH, "r") as f:
            data = json.load(f)
            key = data.get("model")
            if key in MODEL_REGISTRY:
                return key
    return "dfn5b-vit-H-14"  # default

def save_model_config(model_key: str) -> None:
    """Persist the selected model key."""
    with open(MODEL_CONFIG_PATH, "w") as f:
        json.dump({"model": model_key}, f, indent=4)


# Resolve active model from config
_active_model_key = load_model_config()
_active_model = MODEL_REGISTRY[_active_model_key]

OPENCLIP_MODEL_ID = _active_model["openclip_model_id"]
MODEL_ID = _active_model["model_id"]
PRETRAINED = _active_model["pretrained"]

ZEROSHOT_WEIGHTS_PATH = Path("clip_zeroshot_cls.safetensors")
LEGACY_ZEROSHOT_WEIGHTS_PATH = Path("clip_zeroshot_cls.pth")
ZEROSHOT_LABELS_PATH  = Path("clip_zeroshot_cls_labels.json")
_ZEROSHOT_TENSOR_KEY = "zeroshot_weights"
OV_DEVICE = "GPU"
SUPPORTED_OV_DEVICES = ("GPU", "CPU", "NPU")
NPU_IMAGE_SIZE = 378
NPU_CONTEXT_LENGTH = 77

# ---------------------------------------------------------------------------
# Vision-encoder backend selection
#   "openvino"   → run the CLIP visual encoder via the optimum-intel OV wrapper
#   "dlstreamer" → run the CLIP visual encoder through Intel DL Streamer's
#                  gvaclassify GStreamer element
# ---------------------------------------------------------------------------
VISION_BACKEND = os.environ.get("VISION_BACKEND", "openvino").strip().lower()
DLSTREAMER_DEVICE = os.environ.get("DLSTREAMER_DEVICE", "GPU").strip().upper()
MODEL_PROC_PATH = Path(__file__).parent / "model_proc" / "clip_image_encoder.json"


def _save_weights(weights: torch.Tensor) -> None:
    """Persist the zero-shot weight matrix in safetensors format."""
    _safetensors_save(
        {_ZEROSHOT_TENSOR_KEY: weights.detach().cpu().contiguous()},
        str(ZEROSHOT_WEIGHTS_PATH),
    )


def _load_weights() -> torch.Tensor:
    """Load the zero-shot weight matrix from the safetensors file."""
    return _safetensors_load(str(ZEROSHOT_WEIGHTS_PATH))[_ZEROSHOT_TENSOR_KEY]


def _maybe_migrate_legacy_weights() -> None:
    """One-time migration of a legacy .pth weight cache to safetensors."""
    if ZEROSHOT_WEIGHTS_PATH.exists() or not LEGACY_ZEROSHOT_WEIGHTS_PATH.exists():
        return
    try:
        legacy = torch.load(LEGACY_ZEROSHOT_WEIGHTS_PATH, map_location="cpu")
        _save_weights(legacy)
        logger.info(
            "Migrated legacy weights %s -> %s",
            LEGACY_ZEROSHOT_WEIGHTS_PATH, ZEROSHOT_WEIGHTS_PATH,
        )
    except Exception as exc:
        logger.warning("Could not migrate legacy weights (%s).", exc)


def _ov_runtime_config(device: str) -> dict[str, str]:
    """Return a conservative OpenVINO runtime config for the target device."""
    config: dict[str, str] = {"PERFORMANCE_HINT": "LATENCY"}
    if device.upper() == "GPU":
        cache_dir = Path(__file__).parent / ".openvino_cache" / "GPU"
        cache_dir.mkdir(parents=True, exist_ok=True)
        config.update({"NUM_STREAMS": "1", "CACHE_DIR": str(cache_dir.resolve())})
    return config


def _resolve_ov_device(requested: str) -> str:
    """Return *requested* if available, otherwise fall back to CPU."""
    normalized = requested.upper()
    if normalized == "CPU":
        return "CPU"
    try:
        import openvino as ov
        available = ov.Core().available_devices
        if normalized in [d.upper() for d in available]:
            return normalized
        logger.warning(
            "OpenVINO device '%s' not available (available: %s). Falling back to CPU.",
            requested, available,
        )
    except Exception as exc:
        logger.warning("Could not query OpenVINO devices (%s). Falling back to CPU.", exc)
    return "CPU"
TOP_K = 5

# If the top-1 softmax probability (0-100) is below this value the result is
# treated as "Unknown".  Override via the UNKNOWN_THRESHOLD env var.
DEFAULT_UNKNOWN_THRESHOLD: float = float(os.environ.get("UNKNOWN_THRESHOLD", 40.0))


class Prediction(NamedTuple):
    label: str          # full path label e.g. "Fruit/Apple/Granny-Smith"
    short_name: str     # last segment e.g. "Granny-Smith"
    confidence: float   # 0-100 softmax probability
    is_unknown: bool = False  # True when confidence < threshold


class ZeroShotClassifier:
    """Wraps OpenVINO-accelerated CLIP for zero-shot product classification."""

    def __init__(self, quantize: bool = False, ov_device: str = OV_DEVICE,
                 vision_backend: str = VISION_BACKEND) -> None:
        self.ov_device = _resolve_ov_device(ov_device)
        self.backend = self.ov_device
        self.quantize = quantize
        self.vision_backend = vision_backend.strip().lower()
        
        clip_model, _, preprocess = open_clip.create_model_and_transforms(OPENCLIP_MODEL_ID, pretrained=PRETRAINED)
        tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL_ID)
        self._clip_model = clip_model
        self.tokenizer = tokenizer
        #self.processor = preprocess
        #self.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)

        self.model_dir = self._resolve_model_dir()
        self._ensure_model_artifacts()

        self.zeroshot_weights = self._load_or_build_weights()
        self._ov_vision = self._load_ov_vision()

    # ------------------------------------------------------------------
    # Model switching
    # ------------------------------------------------------------------

    @property
    def active_model_key(self) -> str:
        return load_model_config()

    def _base_model_dir(self) -> Path:
        return Path(f"{MODEL_ID.split('/')[-1]}-openclip")

    def _npu_static_model_dir(self) -> Path:
        return Path(f"{MODEL_ID.split('/')[-1]}-openclip-npu-static")

    def _resolve_model_dir(self) -> Path:
        if self.ov_device == "NPU":
            return self._npu_static_model_dir() / ("INT8" if self.quantize else "FP16")
        return self._base_model_dir() / ("INT8" if self.quantize else "FP16")

    def _ensure_standard_model_artifacts(self) -> None:
        if self.quantize:
            if not self.model_dir.exists():
                logger.info("Exporting INT8 quantised OV model …")
                OVModelOpenCLIPForZeroShotImageClassification.from_pretrained(
                    MODEL_ID,
                    quantization_config=OVWeightQuantizationConfig(bits=8),
                ).save_pretrained(self.model_dir)
        else:
            if not self.model_dir.exists():
                logger.info("Exporting FP16 OV model …")
                OVModelOpenCLIPForZeroShotImageClassification.from_pretrained(
                    MODEL_ID
                ).save_pretrained(self.model_dir)

    def _ensure_npu_static_artifacts(self) -> None:
        import openvino as ov

        try:
            import nncf
        except Exception:
            nncf = None

        image_xml = self.model_dir / "image_encoder.xml"
        text_xml = self.model_dir / "text_encoder.xml"
        if image_xml.exists() and text_xml.exists():
            return

        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Exporting NPU static OV models to %s …", self.model_dir)

        class ImageEncoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.visual = model.visual

            def forward(self, pixel_values):
                return self.visual(pixel_values)

        class TextEncoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids):
                return self.model.encode_text(input_ids)

        clip_model = self._clip_model
        clip_model.eval()

        image_encoder = ImageEncoder(clip_model).eval()
        text_encoder = TextEncoder(clip_model).eval()

        dummy_image = torch.randn(1, 3, NPU_IMAGE_SIZE, NPU_IMAGE_SIZE, dtype=torch.float32)
        dummy_text = torch.zeros(1, NPU_CONTEXT_LENGTH, dtype=torch.long)

        with torch.no_grad():
            ov_image_model = ov.convert_model(
                image_encoder,
                example_input=dummy_image,
                input=[ov.PartialShape([1, 3, NPU_IMAGE_SIZE, NPU_IMAGE_SIZE])],
            )
            ov_text_model = ov.convert_model(
                text_encoder,
                example_input=dummy_text,
                input=[ov.PartialShape([1, NPU_CONTEXT_LENGTH])],
            )

        ov_image_model.inputs[0].get_tensor().set_names({"pixel_values"})
        ov_image_model.outputs[0].get_tensor().set_names({"image_features"})
        ov_image_model.reshape({"pixel_values": [1, 3, NPU_IMAGE_SIZE, NPU_IMAGE_SIZE]})
        ov_image_model.validate_nodes_and_infer_types()

        ov_text_model.inputs[0].get_tensor().set_names({"input_ids"})
        ov_text_model.outputs[0].get_tensor().set_names({"text_features"})
        ov_text_model.reshape({"input_ids": [1, NPU_CONTEXT_LENGTH]})
        ov_text_model.validate_nodes_and_infer_types()

        if self.quantize:
            if nncf is None:
                raise RuntimeError(
                    "INT8 NPU export requires nncf. Install it and retry, or select FP16."
                )
            ov_image_model = nncf.compress_weights(
                ov_image_model, mode=nncf.CompressWeightsMode.INT8_SYM
            )
            ov_text_model = nncf.compress_weights(
                ov_text_model, mode=nncf.CompressWeightsMode.INT8_SYM
            )
            ov.save_model(ov_image_model, str(image_xml), compress_to_fp16=False)
            ov.save_model(ov_text_model, str(text_xml), compress_to_fp16=False)
        else:
            ov.save_model(ov_image_model, str(image_xml), compress_to_fp16=True)
            ov.save_model(ov_text_model, str(text_xml), compress_to_fp16=True)

    def _ensure_model_artifacts(self) -> None:
        self.model_dir = self._resolve_model_dir()
        if self.ov_device == "NPU":
            self._ensure_npu_static_artifacts()
        else:
            self._ensure_standard_model_artifacts()

    def _reload_runtime_model(self) -> None:
        self.model_dir = self._resolve_model_dir()
        self._ensure_model_artifacts()
        self._ov_vision = self._load_ov_vision()

    # ------------------------------------------------------------------
    # Model export management
    # ------------------------------------------------------------------

    def _current_export_dir(self) -> Path:
        """Directory holding the exported artifacts for the current config."""
        precision = "INT8" if self.quantize else "FP16"
        if self.vision_backend == "dlstreamer":
            return self._dlstreamer_model_dir()
        if self.ov_device == "NPU":
            return self._npu_static_model_dir() / precision
        return self._base_model_dir() / precision

    def _expected_artifacts(self) -> list[Path]:
        """Exported model files that must exist for the current backend/precision."""
        d = self._current_export_dir()
        if self.vision_backend == "dlstreamer":
            return [d / "image_encoder.xml", d / "image_encoder.bin"]
        if self.ov_device == "NPU":
            return [d / "image_encoder.xml", d / "text_encoder.xml"]
        return [d / "openvino_model_vision.xml", d / "openvino_model_text.xml"]

    def export_status(self) -> dict:
        """Report whether the exported model artifacts for the current config exist."""
        artifacts = self._expected_artifacts()
        items = [{"path": str(p), "exists": p.exists()} for p in artifacts]
        return {
            "model": self.active_model_key,
            "model_name": MODEL_REGISTRY.get(self.active_model_key, {}).get(
                "display_name", self.active_model_key
            ),
            "device": self.ov_device,
            "precision": "INT8" if self.quantize else "FP16",
            "vision_backend": self.vision_backend,
            "export_dir": str(self._current_export_dir()),
            "artifacts": items,
            "available": bool(items) and all(i["exists"] for i in items),
        }

    def export_models(self, force: bool = False, progress_cb=None) -> dict:
        """Export (or re-export) the OpenCLIP model for the current config.

        When *force* is True, any existing exported artifacts are removed first
        so the model is regenerated from scratch.
        """
        def emit(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)

        if force:
            emit("Removing existing exported artifacts \u2026")
            target = self._current_export_dir()
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)

        emit("Exporting model \u2014 this can take a while \u2026")
        if self.vision_backend == "dlstreamer":
            self._ensure_dlstreamer_image_encoder()
        else:
            self.model_dir = self._resolve_model_dir()
            self._ensure_model_artifacts()

        emit("Reloading inference runtime \u2026")
        self._ov_vision = self._load_ov_vision()
        emit("Export complete")
        return self.export_status()

    def switch_model(self, model_key: str, progress_cb=None) -> None:
        """Switch to a different CLIP model and regenerate everything."""
        global OPENCLIP_MODEL_ID, MODEL_ID, PRETRAINED

        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key: {model_key}")

        cfg = MODEL_REGISTRY[model_key]
        OPENCLIP_MODEL_ID = cfg["openclip_model_id"]
        MODEL_ID = cfg["model_id"]
        PRETRAINED = cfg["pretrained"]
        save_model_config(model_key)

        # Rebuild tokenizer & processor for the new model
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            OPENCLIP_MODEL_ID, pretrained=PRETRAINED
        )
        self._clip_model = clip_model
        self.tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL_ID)
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)

        self._reload_runtime_model()

        # Regenerate label embeddings
        self.rebuild_weights(progress_cb=progress_cb)

    def switch_backend(self, backend: str) -> None:
        """Switch between OpenVINO backends such as GPU, CPU, and NPU."""
        resolved = _resolve_ov_device(backend)
        if self.ov_device == resolved:
            logger.info("Already using %s — nothing to do.", resolved)
            return

        self.ov_device = resolved
        self.backend = resolved
        self._reload_runtime_model()
        logger.info("Switched OpenVINO backend to %s.", self.ov_device)

    def switch_vision_backend(self, vision_backend: str) -> None:
        """Switch the vision encoder between 'openvino' and 'dlstreamer'."""
        normalized = vision_backend.strip().lower()
        if normalized not in ("openvino", "dlstreamer"):
            raise ValueError("vision_backend must be 'openvino' or 'dlstreamer'")
        if self.vision_backend == normalized:
            logger.info("Already using %s vision backend — nothing to do.", normalized)
            return

        # Verify the runtime is installed before the (potentially slow) reload so
        # the user gets a clear message instead of a late, cryptic failure.
        if normalized == "dlstreamer":
            from dlstreamer_encoder import check_dlstreamer_available

            available, detail = check_dlstreamer_available()
            logger.info("DL Streamer availability: %s (%s)", available, detail)
            if not available:
                raise RuntimeError(detail)

        self.vision_backend = normalized
        self._reload_runtime_model()
        logger.info("Switched vision backend to %s.", self.vision_backend)

    def switch_precision(self, quantize: bool) -> None:
        """Switch between FP16 (quantize=False) and INT8 (quantize=True).

        Exports the OV model for the target precision if it does not already
        exist, then hot-swaps the compiled visual model.  Label embeddings are
        not affected and do not need to be regenerated.
        """
        if self.quantize == quantize:
            logger.info("Already using %s — nothing to do.", "INT8" if quantize else "FP16")
            return

        self.quantize = quantize
        self._reload_runtime_model()
        logger.info("Switched OV precision to %s.", "INT8" if quantize else "FP16")

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    def rebuild_weights(self, progress_cb=None) -> None:
        """Reload labels from disk and regenerate only missing embeddings.

        Labels that already have a cached embedding are reused as-is.
        Only new labels (not present in the saved manifest) are encoded.
        Pass *progress_cb(current, total, label)* to stream progress.
        """
        global LABELS
        LABELS = load_labels()
        _maybe_migrate_legacy_weights()

        # Load existing weights + manifest if both are present
        if ZEROSHOT_WEIGHTS_PATH.exists() and ZEROSHOT_LABELS_PATH.exists():
            try:
                with open(ZEROSHOT_LABELS_PATH) as f:
                    saved_labels = json.load(f)
                existing = _load_weights()
                saved_idx = {lbl: i for i, lbl in enumerate(saved_labels)}

                new_labels = [lbl for lbl in LABELS if lbl not in saved_idx]
                if not new_labels and [lbl for lbl in saved_labels if lbl not in set(LABELS)] == []:
                    # Every current label already has an embedding — just reorder if needed
                    if LABELS == saved_labels:
                        logger.info("All %d embeddings are up to date — nothing to regenerate.", len(LABELS))
                        if progress_cb:
                            for i, lbl in enumerate(LABELS):
                                progress_cb(i + 1, len(LABELS), lbl)
                        return

                logger.info("%d new label(s) to encode, %d reused from cache.",
                            len(new_labels), len(LABELS) - len(new_labels))
                total = len(LABELS)
                cols = []
                for i, label in enumerate(LABELS):
                    if label in saved_idx:
                        cols.append(existing[:, saved_idx[label]])
                    else:
                        cols.append(self._encode_label(label))
                    if progress_cb:
                        progress_cb(i + 1, total, label)

                self.zeroshot_weights = torch.stack(cols, dim=1)
                _save_weights(self.zeroshot_weights)
                with open(ZEROSHOT_LABELS_PATH, "w") as f:
                    json.dump(LABELS, f, indent=2)
                logger.info("Saved updated zero-shot weights to %s", ZEROSHOT_WEIGHTS_PATH)
                return
            except Exception as exc:
                logger.warning("Could not load cached weights (%s) — doing full rebuild.", exc)

        # No usable cache — full rebuild
        self.zeroshot_weights = self._build_weights(progress_cb=progress_cb)

    def _dlstreamer_model_dir(self) -> Path:
        return Path(
            f"{MODEL_ID.split('/')[-1]}-openclip-dlstreamer"
        ) / ("INT8" if self.quantize else "FP16")

    def _ensure_dlstreamer_image_encoder(self) -> Path:
        """Export the CLIP visual encoder to an OpenVINO IR for gvaclassify.

        Reuses the static-shape export path (named ``pixel_values`` input and
        ``image_features`` output) so DL Streamer's gvaclassify can load it and
        the model-proc can bind to those layer names.
        """
        model_dir = self._dlstreamer_model_dir()
        image_xml = model_dir / "image_encoder.xml"
        if not image_xml.exists():
            saved_dir = self.model_dir
            self.model_dir = model_dir
            try:
                self._ensure_npu_static_artifacts()
            finally:
                self.model_dir = saved_dir
        return image_xml

    def _load_dlstreamer_vision(self):
        """Create the gvaclassify-backed visual encoder."""
        from dlstreamer_encoder import DLStreamerVisionEncoder

        image_xml = self._ensure_dlstreamer_image_encoder()
        model_proc = MODEL_PROC_PATH if MODEL_PROC_PATH.exists() else None
        logger.info(
            "Loading DL Streamer gvaclassify visual encoder on %s (model=%s)",
            DLSTREAMER_DEVICE, image_xml,
        )
        return DLStreamerVisionEncoder(
            model_xml=image_xml,
            model_proc=model_proc,
            device=DLSTREAMER_DEVICE,
        )

    def _load_ov_vision(self):
        """Load and compile the visual encoder for the current backend."""
        if self.vision_backend == "dlstreamer":
            return self._load_dlstreamer_vision()

        logger.info("Loading OV visual model on %s …", self.ov_device)

        if self.ov_device == "NPU":
            import openvino as ov

            image_xml = self.model_dir / "image_encoder.xml"
            if not image_xml.exists():
                raise FileNotFoundError(
                    f"NPU static model not found at {image_xml}. Re-run with NPU selected."
                )

            core = ov.Core()
            ov_config = _ov_runtime_config(self.ov_device)
            try:
                image_model = core.read_model(str(image_xml))
                return core.compile_model(image_model, self.ov_device, ov_config)
            except RuntimeError as exc:
                if self.ov_device == "CPU":
                    raise
                logger.warning(
                    "Failed to compile NPU model on %s (%s). Falling back to CPU.",
                    self.ov_device, exc,
                )
                image_model = core.read_model(str(image_xml))
                return core.compile_model(image_model, "CPU", _ov_runtime_config("CPU"))

        try:
            return OVModelOpenCLIPVisual.from_pretrained(
                self.model_dir,
                device=self.ov_device,
                ov_config=_ov_runtime_config(self.ov_device),
            )
        except RuntimeError as exc:
            if self.ov_device != "CPU":
                logger.warning(
                    "Failed to load OV model on %s (%s). Falling back to CPU.",
                    self.ov_device, exc,
                )
                self.ov_device = "CPU"
                self.backend = "CPU"
                return OVModelOpenCLIPVisual.from_pretrained(
                    self.model_dir,
                    device="CPU",
                    ov_config=_ov_runtime_config("CPU"),
                )
            raise

    def _build_weights(self, progress_cb=None) -> torch.Tensor:
        """Build zero-shot classifier weight matrix from text prompts."""
        logger.info("Building zero-shot weights (one-time, may take a few minutes) …")
        clip_model = self._clip_model
        
        total = len(LABELS)
        weights = []
        for i, label in enumerate(tqdm(LABELS, desc="Encoding labels")):
            texts = [t.format(label=label) for t in CLASS_TEMPLATES]
            #inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            with torch.no_grad():
                #embeddings = clip_model.get_text_features(**inputs)
                embeddings = clip_model.encode_text(self.tokenizer(texts))
            embedding = F.normalize(embeddings, dim=-1).mean(dim=0)
            embedding /= embedding.norm()
            weights.append(embedding)
            if progress_cb:
                progress_cb(i + 1, total, label)
        weight_matrix = torch.stack(weights, dim=1)
        _save_weights(weight_matrix)
        with open(ZEROSHOT_LABELS_PATH, "w") as f:
            json.dump(LABELS, f, indent=2)
        logger.info("Saved zero-shot weights to %s", ZEROSHOT_WEIGHTS_PATH)
        return weight_matrix

    def _load_or_build_weights(self) -> torch.Tensor:
        _maybe_migrate_legacy_weights()
        if ZEROSHOT_WEIGHTS_PATH.exists():
            logger.info("Loading cached zero-shot weights from %s", ZEROSHOT_WEIGHTS_PATH)
            weights = _load_weights()
            logger.info("Weights shape: %s", weights.shape)
            return weights
        return self._build_weights()

    def _encode_label(self, label: str) -> torch.Tensor:
        """Compute a normalised embedding vector for a single label."""
        texts = [t.format(label=label) for t in CLASS_TEMPLATES]
        with torch.no_grad():
            embeddings = self._clip_model.encode_text(self.tokenizer(texts))
        embedding = F.normalize(embeddings, dim=-1).mean(dim=0)
        embedding /= embedding.norm()
        return embedding

    def add_label_weight(self, label: str) -> None:
        """Append an embedding for *label* without recomputing existing labels."""
        global LABELS
        old_len = len(LABELS)
        if self.zeroshot_weights.shape[1] != old_len:
            logger.warning("Weight matrix out of sync — falling back to full rebuild.")
            LABELS = load_labels()
            self.zeroshot_weights = self._build_weights()
            return
        new_vec = self._encode_label(label)
        self.zeroshot_weights = torch.cat(
            [self.zeroshot_weights, new_vec.unsqueeze(1)], dim=1
        )
        LABELS = load_labels()
        _save_weights(self.zeroshot_weights)
        with open(ZEROSHOT_LABELS_PATH, "w") as f:
            json.dump(LABELS, f, indent=2)
        logger.info("Incrementally added embedding for '%s'", label)

    def remove_label_weight(self, label: str) -> None:
        """Remove the embedding column for *label* without recomputing others."""
        global LABELS
        old_labels = list(LABELS)
        if self.zeroshot_weights.shape[1] != len(old_labels):
            logger.warning("Weight matrix out of sync — falling back to full rebuild.")
            LABELS = load_labels()
            self.zeroshot_weights = self._build_weights()
            return
        try:
            idx = old_labels.index(label)
        except ValueError:
            logger.warning("Label '%s' not in cached list — falling back to full rebuild.", label)
            LABELS = load_labels()
            self.zeroshot_weights = self._build_weights()
            return
        cols = [i for i in range(self.zeroshot_weights.shape[1]) if i != idx]
        self.zeroshot_weights = self.zeroshot_weights[:, cols]
        LABELS = load_labels()
        _save_weights(self.zeroshot_weights)
        with open(ZEROSHOT_LABELS_PATH, "w") as f:
            json.dump(LABELS, f, indent=2)
        logger.info("Incrementally removed embedding for '%s'", label)

    def update_label_weight(self, old_label: str, new_label: str) -> None:
        """Replace the embedding for *old_label* with one for *new_label*."""
        global LABELS
        old_labels = list(LABELS)
        if self.zeroshot_weights.shape[1] != len(old_labels):
            logger.warning("Weight matrix out of sync — falling back to full rebuild.")
            LABELS = load_labels()
            self.zeroshot_weights = self._build_weights()
            return
        try:
            idx = old_labels.index(old_label)
        except ValueError:
            logger.warning("Label '%s' not in cached list — falling back to full rebuild.", old_label)
            LABELS = load_labels()
            self.zeroshot_weights = self._build_weights()
            return
        new_vec = self._encode_label(new_label)
        self.zeroshot_weights[:, idx] = new_vec
        LABELS = load_labels()
        _save_weights(self.zeroshot_weights)
        with open(ZEROSHOT_LABELS_PATH, "w") as f:
            json.dump(LABELS, f, indent=2)
        logger.info("Incrementally updated embedding '%s' -> '%s'", old_label, new_label)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(
        self,
        img_array,
        threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
    ) -> list[Prediction]:
        """
        Run zero-shot classification on a single image (numpy array, BGR or RGB).

        Returns a ranked list of top-K Prediction objects.
        When the top-1 softmax confidence is below *threshold* (0-100), every
        prediction in the list is flagged with ``is_unknown=True`` so callers
        can decide how to present the ambiguous result.
        """
        t0 = time.perf_counter()

        if self.vision_backend == "dlstreamer":
            # gvaclassify performs resize + normalization internally per the
            # model-proc, so it receives the raw (BGR) frame directly.
            feats = self._ov_vision.encode(img_array)
            image_features = torch.from_numpy(
                np.ascontiguousarray(feats)
            ).float().reshape(1, -1)
            image_features = F.normalize(image_features, dim=-1)
        else:
            img_inputs = self.processor(images=[img_array], return_tensors="pt")

            if self.ov_device == "NPU":
                img_np = img_inputs["pixel_values"].numpy().astype("float32")
                visual_out = self._ov_vision({"pixel_values": img_np})
                image_features = torch.from_numpy(visual_out[self._ov_vision.outputs[0]])
            else:
                visual_out = self._ov_vision(**img_inputs)
                image_features = visual_out["image_features"]  # (1, D)

        logits = 100.0 * image_features @ self.zeroshot_weights  # (1, N)
        probs = torch.softmax(logits, dim=-1).squeeze()           # (N,)

        top_values, top_indices = probs.topk(TOP_K)

        top_confidence = round(top_values[0].item() * 100, 2)
        is_unknown = top_confidence < threshold

        predictions = [
            Prediction(
                label=LABELS[idx.item()],
                short_name=LABELS[idx.item()].split("/")[-1],
                confidence=round(val.item() * 100, 2),
                is_unknown=is_unknown,
            )
            for val, idx in zip(top_values, top_indices)
        ]

        elapsed = time.perf_counter() - t0
        logger.info(
            "Inference complete in %.2fs — top: %s (%.1f%%) [%s] threshold=%.1f%%",
            elapsed,
            predictions[0].label,
            predictions[0].confidence,
            "UNKNOWN" if is_unknown else "OK",
            threshold,
        )
        return predictions
