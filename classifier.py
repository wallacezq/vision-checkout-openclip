"""
classifier.py
Zero-shot image classifier using OpenCLIP / Apple DFN5B-CLIP.
Responsible for loading model weights and running inference.
"""
from __future__ import annotations

import json
import os
import time
import logging
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
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
    #return "metaclip2-vit-bigG-14"  # default
    return "dfn5b-vit-H-14"

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

ZEROSHOT_WEIGHTS_PATH = Path("clip_zeroshot_cls.pth")
OV_DEVICE = "GPU"
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

    def __init__(self, quantize: bool = False, ov_device: str = OV_DEVICE) -> None:
        self.ov_device = ov_device
        
        clip_model, _, preprocess = open_clip.create_model_and_transforms(OPENCLIP_MODEL_ID, pretrained=PRETRAINED)
        tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL_ID)
        self.tokenizer = tokenizer
        #self.processor = preprocess
        #self.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)

        # Resolve / build OV model directory
        base = Path(f"{MODEL_ID.split('/')[-1]}-openclip")
        if quantize:
            self.model_dir = base / "INT8"
            if not self.model_dir.exists():
                logger.info("Exporting INT8 quantised OV model …")
                OVModelOpenCLIPForZeroShotImageClassification.from_pretrained(
                    MODEL_ID,
                    quantization_config=OVWeightQuantizationConfig(bits=8),
                ).save_pretrained(self.model_dir)
        else:
            self.model_dir = base / "FP16"
            if not self.model_dir.exists():
                logger.info("Exporting FP16 OV model …")
                OVModelOpenCLIPForZeroShotImageClassification.from_pretrained(
                    MODEL_ID
                ).save_pretrained(self.model_dir)

        self.zeroshot_weights = self._load_or_build_weights()

    # ------------------------------------------------------------------
    # Model switching
    # ------------------------------------------------------------------

    @property
    def active_model_key(self) -> str:
        return load_model_config()

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
        self.tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL_ID)
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)

        # Rebuild OV model directory
        base = Path(f"{MODEL_ID.split('/')[-1]}-openclip")
        self.model_dir = base / "FP16"
        if not self.model_dir.exists():
            logger.info("Exporting FP16 OV model for %s …", MODEL_ID)
            OVModelOpenCLIPForZeroShotImageClassification.from_pretrained(
                MODEL_ID
            ).save_pretrained(self.model_dir)

        # Regenerate label embeddings
        self.rebuild_weights(progress_cb=progress_cb)

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    def rebuild_weights(self, progress_cb=None) -> None:
        """Reload labels from disk and regenerate zero-shot classifier weights.

        *progress_cb*, if provided, is called as ``progress_cb(current, total, label)``
        after each label is encoded.
        """
        global LABELS
        LABELS = load_labels()
        if ZEROSHOT_WEIGHTS_PATH.exists():
            ZEROSHOT_WEIGHTS_PATH.unlink()
        self.zeroshot_weights = self._build_weights(progress_cb=progress_cb)

    def _build_weights(self, progress_cb=None) -> torch.Tensor:
        """Build zero-shot classifier weight matrix from text prompts."""
        logger.info("Building zero-shot weights (one-time, may take a few minutes) …")
        #clip_model = CLIPModel.from_pretrained(MODEL_ID)
        #clip_model = AutoModel.from_pretrained(MODEL_ID)
        clip_model, _, preprocess = open_clip.create_model_and_transforms(OPENCLIP_MODEL_ID, pretrained=PRETRAINED)
        #tokenizer = open_clip.get_tokenizer("ViT-bigG-14-worldwide")
        
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
        torch.save(weight_matrix, ZEROSHOT_WEIGHTS_PATH)
        logger.info("Saved zero-shot weights to %s", ZEROSHOT_WEIGHTS_PATH)
        return weight_matrix

    def _load_or_build_weights(self) -> torch.Tensor:
        if ZEROSHOT_WEIGHTS_PATH.exists():
            logger.info("Loading cached zero-shot weights from %s", ZEROSHOT_WEIGHTS_PATH)
            weights = torch.load(ZEROSHOT_WEIGHTS_PATH, map_location="cpu")
            logger.info("Weights shape: %s", weights.shape)
            return weights
        return self._build_weights()

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

        img_inputs = self.processor(
            images=[img_array], return_tensors="pt" #, padding=True
        )
        
        ov_vision = OVModelOpenCLIPVisual.from_pretrained(
            self.model_dir, device=self.ov_device
        )
        visual_out = ov_vision(**img_inputs)
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
