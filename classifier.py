"""
classifier.py
Zero-shot image classifier using OpenCLIP / Apple DFN5B-CLIP.
Responsible for loading model weights and running inference.
"""
from __future__ import annotations

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Product label taxonomy
# ---------------------------------------------------------------------------
LABELS: list[str] = [
    "Fruit/Apple/Golden-Delicious",
    "Fruit/Apple/Granny-Smith",
    "Fruit/Apple/Pink-Lady",
    "Fruit/Apple/Red-Delicious",
    "Fruit/Apple/Royal-Gala",
    "Fruit/Avocado",
    "Fruit/Banana",
    "Fruit/Kiwi",
    "Fruit/Lemon",
    "Fruit/Lime",
    "Fruit/Mango",
    "Fruit/Melon/Cantaloupe",
    "Fruit/Melon/Galia-Melon",
    "Fruit/Melon/Honeydew-Melon",
    "Fruit/Melon/Watermelon",
    "Fruit/Nectarine",
    "Fruit/Orange",
    "Fruit/Papaya",
    "Fruit/Passion-Fruit",
    "Fruit/Peach",
    "Fruit/Pear/Anjou",
    "Fruit/Pear/Conference",
    "Fruit/Pear/Kaiser",
    "Fruit/Pineapple",
    "Fruit/Plum",
    "Fruit/Pomegranate",
    "Fruit/Red-Grapefruit",
    "Fruit/Satsumas",
    "Packages/Juice/Bravo-Apple-Juice",
    "Packages/Juice/Bravo-Orange-Juice",
    "Packages/Juice/God-Morgon-Apple-Juice",
    "Packages/Juice/God-Morgon-Orange-Juice",
    "Packages/Juice/God-Morgon-Orange-Red-Grapefruit-Juice",
    "Packages/Juice/God-Morgon-Red-Grapefruit-Juice",
    "Packages/Juice/Tropicana-Apple-Juice",
    "Packages/Juice/Tropicana-Golden-Grapefruit",
    "Packages/Juice/Tropicana-Juice-Smooth",
    "Packages/Juice/Tropicana-Mandarin-Morning",
    "Packages/Milk/Arla-Ecological-Medium-Fat-Milk",
    "Packages/Milk/Arla-Lactose-Medium-Fat-Milk",
    "Packages/Milk/Arla-Medium-Fat-Milk",
    "Packages/Milk/Arla-Standard-Milk",
    "Packages/Milk/Garant-Ecological-Medium-Fat-Milk",
    "Packages/Milk/Garant-Ecological-Standard-Milk",
    "Packages/Oat-Milk/Oatly-Oat-Milk",
    "Packages/Oatghurt/Oatly-Natural-Oatghurt",
    "Packages/Sour-Cream/Arla-Ecological-Sour-Cream",
    "Packages/Sour-Cream/Arla-Sour-Cream",
    "Packages/Sour-Milk/Arla-Sour-Milk",
    "Packages/Soy-Milk/Alpro-Fresh-Soy-Milk",
    "Packages/Soy-Milk/Alpro-Shelf-Soy-Milk",
    "Packages/Soyghurt/Alpro-Blueberry-Soyghurt",
    "Packages/Soyghurt/Alpro-Vanilla-Soyghurt",
    "Packages/Yoghurt/Arla-Mild-Vanilla-Yoghurt",
    "Packages/Yoghurt/Arla-Natural-Mild-Low-Fat-Yoghurt",
    "Packages/Yoghurt/Arla-Natural-Yoghurt",
    "Packages/Yoghurt/Valio-Vanilla-Yoghurt",
    "Packages/Yoghurt/Yoggi-Strawberry-Yoghurt",
    "Packages/Yoghurt/Yoggi-Vanilla-Yoghurt",
    "Packages/Instant-Noodles/Nissin-Premium-Ramen",
    "Packages/Instant-Noodles/Shin-Ramyun",
    "Packages/Chips/Snek-Mi-Mi",
    "Vegetables/Asparagus",
    "Vegetables/Aubergine",
    "Vegetables/Brown-Cap-Mushroom",
    "Vegetables/Cabbage",
    "Vegetables/Carrots",
    "Vegetables/Cucumber",
    "Vegetables/Garlic",
    "Vegetables/Ginger",
    "Vegetables/Leek",
    "Vegetables/Onion/Yellow-Onion",
    "Vegetables/Pepper/Green-Bell-Pepper",
    "Vegetables/Pepper/Orange-Bell-Pepper",
    "Vegetables/Pepper/Red-Bell-Pepper",
    "Vegetables/Pepper/Yellow-Bell-Pepper",
    "Vegetables/Potato/Floury-Potato",
    "Vegetables/Potato/Solid-Potato",
    "Vegetables/Potato/Sweet-Potato",
    "Vegetables/Red-Beet",
    "Vegetables/Tomato/Beef-Tomato",
    "Vegetables/Tomato/Regular-Tomato",
    "Vegetables/Tomato/Vine-Tomato",
    "Vegetables/Zucchini",
    "Beverages/Can/Milo",
    "Beverages/Can/Coke",
]

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

OPENCLIP_MODEL_ID = "ViT-bigG-14-worldwide"
MODEL_ID = "timm/vit_gigantic_patch14_clip_378.metaclip2_worldwide"
PRETRAINED = "metaclip2_worldwide"

#OPENCLIP_MODEL_ID = "ViT-H-14-378-quickgelu"
#MODEL_ID = "apple/DFN5B-CLIP-ViT-H-14-378"
#PRETRAINED = "dfn5b"
#MODEL_ID = "timm/vit_gigantic_patch14_clip_378.metaclip2_worldwide"
#MODEL_ID = "facebook/metaclip-2-worldwide-giant-378"

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
    # Weight helpers
    # ------------------------------------------------------------------

    def _build_weights(self) -> torch.Tensor:
        """Build zero-shot classifier weight matrix from text prompts."""
        logger.info("Building zero-shot weights (one-time, may take a few minutes) …")
        #clip_model = CLIPModel.from_pretrained(MODEL_ID)
        #clip_model = AutoModel.from_pretrained(MODEL_ID)
        clip_model, _, preprocess = open_clip.create_model_and_transforms(OPENCLIP_MODEL_ID, pretrained=PRETRAINED)
        #tokenizer = open_clip.get_tokenizer("ViT-bigG-14-worldwide")
        
        weights = []
        for label in tqdm(LABELS, desc="Encoding labels"):
            texts = [t.format(label=label) for t in CLASS_TEMPLATES]
            #inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            with torch.no_grad():
                #embeddings = clip_model.get_text_features(**inputs)
                embeddings = clip_model.encode_text(self.tokenizer(texts))
            embedding = F.normalize(embeddings, dim=-1).mean(dim=0)
            embedding /= embedding.norm()
            weights.append(embedding)
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
