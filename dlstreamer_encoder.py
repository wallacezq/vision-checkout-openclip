"""
dlstreamer_encoder.py
Run the CLIP visual (image) encoder through Intel DL Streamer's ``gvaclassify``
GStreamer element and return the resulting image-feature tensor.

This backend drives DL Streamer through the **native** ``gst-launch-1.0``
command-line tool (a subprocess) instead of the Python ``gi`` / PyGObject
bindings. On Windows the pip ``PyGObject`` wheels are built against the MSYS2
GLib stack (``libglib-2.0-0.dll`` / ``libgirepository-2.0-0.dll``) while the
Intel DL Streamer runtime ships the MSVC GLib stack (``glib-2.0-0.dll`` /
``girepository-1.0``), and the only ``gi`` bundled with GStreamer is compiled
for Python 3.9 -- so ``import gi`` cannot be made to work under a modern Python.
Shelling out to ``gst-launch-1.0`` sidesteps PyGObject entirely.

Pipeline (run once per image)::

    filesrc location=<img> ! decodebin ! videoconvert !
    gvaclassify model=<xml> model-proc=<json> device=<dev>
                inference-region=full-frame !
    gvametaconvert add-tensor-data=true format=json !
    gvametapublish method=file file-path=<out.json> file-format=json !
    fakesink sync=false

``gvaclassify`` runs the OpenVINO visual encoder on the full frame; the raw
``image_features`` embedding is serialized to a JSON file by ``gvametaconvert``
+ ``gvametapublish`` and read back here with the standard-library ``json``
module.

Requirements (installed with the Intel DL Streamer runtime, not via pip):
  * GStreamer 1.0 + the DL Streamer plugins (provides ``gvaclassify``,
    ``gvametaconvert``, ``gvametapublish`` and ``gst-launch-1.0``)

The GStreamer / DL Streamer locations are discovered from the
``GSTREAMER_1_0_ROOT_MSVC_X86_64`` and ``DLSTREAMER_DIR`` environment variables
(set by their installers) and can be overridden with ``GSTREAMER_ROOT`` /
``DLSTREAMER_DIR`` / ``GST_LAUNCH``. No Python GObject bindings are required.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Wall-clock timeout for a single-frame inference subprocess.
_SUBPROCESS_TIMEOUT_S = float(os.environ.get("DLSTREAMER_TIMEOUT_S", "120"))

# Layer name of the CLIP embedding (matches model_proc/clip_image_encoder.json).
_FEATURES_LAYER = "image_features"


def _first_existing(*paths: str | os.PathLike) -> str | None:
    for p in paths:
        if p and Path(p).exists():
            return str(p)
    return None


def _resolve_gst_launch() -> str:
    """Locate the native ``gst-launch-1.0`` executable."""
    explicit = os.environ.get("GST_LAUNCH")
    if explicit and Path(explicit).exists():
        return explicit

    gst_root = os.environ.get("GSTREAMER_ROOT") or os.environ.get(
        "GSTREAMER_1_0_ROOT_MSVC_X86_64"
    )
    candidates: list[str] = []
    if gst_root:
        candidates += [
            str(Path(gst_root) / "bin" / "gst-launch-1.0.exe"),
            str(Path(gst_root) / "bin" / "gst-launch-1.0"),
        ]
    found = _first_existing(*candidates)
    if found:
        return found

    on_path = shutil.which("gst-launch-1.0") or shutil.which("gst-launch-1.0.exe")
    if on_path:
        return on_path

    raise RuntimeError(
        "Could not find 'gst-launch-1.0'. Install the GStreamer / DL Streamer "
        "runtime and set GSTREAMER_ROOT (or GSTREAMER_1_0_ROOT_MSVC_X86_64), or "
        "set GST_LAUNCH to the full path of the executable."
    )


def _build_subprocess_env() -> dict[str, str]:
    """Augment the environment so gst-launch finds the DL Streamer plugins/DLLs."""
    env = dict(os.environ)

    gst_root = env.get("GSTREAMER_ROOT") or env.get("GSTREAMER_1_0_ROOT_MSVC_X86_64")
    dls_root = env.get("DLSTREAMER_DIR")

    path_parts: list[str] = []
    plugin_parts: list[str] = []

    if gst_root:
        path_parts.append(str(Path(gst_root) / "bin"))
        plugin_parts.append(str(Path(gst_root) / "lib" / "gstreamer-1.0"))
    if dls_root:
        path_parts += [
            str(Path(dls_root) / "bin"),
            str(Path(dls_root) / "lib"),
            str(Path(dls_root) / "dependencies"),
        ]
        plugin_parts += [
            str(Path(dls_root) / "lib" / "gstreamer-1.0"),
            str(Path(dls_root) / "bin"),
        ]

    if path_parts:
        env["PATH"] = os.pathsep.join(path_parts + [env.get("PATH", "")])

    existing_plugins = env.get("GST_PLUGIN_PATH", "")
    plugin_parts = [p for p in plugin_parts if p]
    if existing_plugins:
        plugin_parts.insert(0, existing_plugins)
    if plugin_parts:
        env["GST_PLUGIN_PATH"] = os.pathsep.join(plugin_parts)

    return env


def _resolve_gst_inspect() -> str | None:
    """Locate the native ``gst-inspect-1.0`` executable (None if not found)."""
    explicit = os.environ.get("GST_INSPECT")
    if explicit and Path(explicit).exists():
        return explicit

    gst_root = os.environ.get("GSTREAMER_ROOT") or os.environ.get(
        "GSTREAMER_1_0_ROOT_MSVC_X86_64"
    )
    candidates: list[str] = []
    if gst_root:
        candidates += [
            str(Path(gst_root) / "bin" / "gst-inspect-1.0.exe"),
            str(Path(gst_root) / "bin" / "gst-inspect-1.0"),
        ]
    found = _first_existing(*candidates)
    if found:
        return found

    return shutil.which("gst-inspect-1.0") or shutil.which("gst-inspect-1.0.exe")


def check_dlstreamer_available() -> tuple[bool, str]:
    """Verify the GStreamer / DL Streamer runtime and ``gvaclassify`` are usable.

    Returns ``(available, detail)`` where *detail* is a human-readable message
    suitable for showing to the user.
    """
    try:
        _resolve_gst_launch()
    except RuntimeError as exc:
        return False, str(exc)

    inspect = _resolve_gst_inspect()
    if inspect is None:
        return False, (
            "GStreamer was found but 'gst-inspect-1.0' is missing — the "
            "GStreamer / DL Streamer runtime install looks incomplete."
        )

    env = _build_subprocess_env()
    try:
        proc = subprocess.run(
            [inspect, "gvaclassify"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f"Could not run 'gst-inspect-1.0': {exc}"

    if proc.returncode != 0:
        return False, (
            "GStreamer is installed but the Intel DL Streamer 'gvaclassify' "
            "element was not found. Install the DL Streamer runtime and set "
            "DLSTREAMER_DIR (or add its plugins to GST_PLUGIN_PATH)."
        )
    return True, "GStreamer and Intel DL Streamer (gvaclassify) are installed."


class DLStreamerVisionEncoder:
    """Single-image CLIP encoder backed by a ``gst-launch-1.0`` subprocess."""

    def __init__(
        self,
        model_xml: str | Path,
        model_proc: str | Path | None = None,
        device: str = "GPU",
    ) -> None:
        # Absolute native paths are passed as element properties, so Windows
        # paths (drive letters, backslashes, spaces) are handled safely.
        self.model_xml = str(Path(model_xml).resolve())
        self.model_proc = str(Path(model_proc).resolve()) if model_proc else None
        self.device = device

        if not Path(self.model_xml).exists():
            raise FileNotFoundError(f"gvaclassify model not found: {self.model_xml}")

        self._gst_launch = _resolve_gst_launch()
        self._env = _build_subprocess_env()

    # ------------------------------------------------------------------

    def _pipeline_args(self, img_path: str, out_json: str) -> list[str]:
        # gst-launch parses its args as a pipeline description and treats '\'
        # as an escape character, so all paths must use forward slashes (which
        # Windows GStreamer accepts).
        model_xml = self.model_xml.replace("\\", "/")
        img_path = img_path.replace("\\", "/")
        out_json = out_json.replace("\\", "/")

        gvaclassify = [
            "gvaclassify",
            f"model={model_xml}",
            "inference-region=full-frame",
            f"device={self.device}",
        ]
        if self.model_proc:
            model_proc = self.model_proc.replace("\\", "/")
            gvaclassify.insert(2, f"model-proc={model_proc}")

        return [
            self._gst_launch,
            "-q",
            "filesrc",
            f"location={img_path}",
            "!",
            "decodebin",
            "!",
            "videoconvert",
            "!",
            *gvaclassify,
            "!",
            "gvametaconvert",
            "add-tensor-data=true",
            "add-empty-results=true",
            "format=json",
            "!",
            "gvametapublish",
            "method=file",
            f"file-path={out_json}",
            "file-format=json",
            "!",
            "fakesink",
            "sync=false",
        ]

    def encode(self, img_bgr: np.ndarray) -> np.ndarray:
        """Run the visual encoder on a single BGR image and return (1, D) features."""
        if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError("Expected a 3-channel (BGR) image array")

        tmp_dir = tempfile.mkdtemp(prefix="dlstreamer_")
        img_path = os.path.join(tmp_dir, "frame.png")
        out_json = os.path.join(tmp_dir, "result.json")
        try:
            self._write_png(img_path, img_bgr)

            args = self._pipeline_args(img_path, out_json)
            logger.debug("gst-launch pipeline: %s", " ".join(args))
            try:
                proc = subprocess.run(
                    args,
                    env=self._env,
                    capture_output=True,
                    text=True,
                    timeout=_SUBPROCESS_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "gvaclassify pipeline timed out after "
                    f"{_SUBPROCESS_TIMEOUT_S:g}s. Check the model, model-proc, and "
                    f"device '{self.device}'."
                ) from exc

            if proc.returncode != 0:
                raise RuntimeError(
                    "gvaclassify pipeline failed (exit "
                    f"{proc.returncode}). Check the model, model-proc, and device "
                    f"'{self.device}'.\n--- gst-launch stderr ---\n{proc.stderr.strip()}"
                )

            if not os.path.exists(out_json):
                raise RuntimeError(
                    "gvaclassify produced no output JSON. "
                    f"gst-launch stderr:\n{proc.stderr.strip()}"
                )
            return self._extract_features(out_json)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------

    @staticmethod
    def _write_png(path: str, img_bgr: np.ndarray) -> None:
        """Write the BGR frame to a lossless PNG for the filesrc pipeline."""
        try:
            import cv2
        except ImportError as exc:  # cv2 is a core project dependency
            raise RuntimeError(
                "OpenCV (cv2) is required to hand a frame to the DL Streamer "
                "subprocess backend."
            ) from exc
        if not cv2.imwrite(path, np.ascontiguousarray(img_bgr)):
            raise RuntimeError(f"Failed to write temporary frame to {path}")

    @classmethod
    def _extract_features(cls, out_json: str) -> np.ndarray:
        """Pull the raw image-feature embedding out of the published JSON."""
        with open(out_json, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
        if not text:
            raise RuntimeError("gvametapublish wrote an empty JSON file.")

        # file-format=json is a JSON array; tolerate json-lines output too.
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [json.loads(line) for line in text.splitlines() if line.strip()]

        best: list[float] | None = None
        best_named = False
        for data, named in cls._iter_tensor_data(parsed):
            if named:
                best = data
                best_named = True
                break
            if best is None:
                best = data

        if best is None:
            raise RuntimeError(
                "No tensor data found in gvaclassify JSON output. Ensure the "
                "model-proc uses the 'raw_data_copy' converter and gvametaconvert "
                "has add-tensor-data=true."
            )
        logger.debug(
            "Extracted %d-D embedding (named=%s) from gvaclassify JSON",
            len(best), best_named,
        )
        return np.asarray(best, dtype=np.float32).reshape(1, -1)

    @classmethod
    def _iter_tensor_data(cls, node):
        """Yield (float_list, is_named_image_features) for every tensor data array."""
        if isinstance(node, dict):
            data = node.get("data")
            if (
                isinstance(data, list)
                and data
                and all(isinstance(v, (int, float)) for v in data)
            ):
                name = str(node.get("layer_name") or node.get("name") or "")
                yield data, (name == _FEATURES_LAYER)
            for value in node.values():
                yield from cls._iter_tensor_data(value)
        elif isinstance(node, list):
            for item in node:
                yield from cls._iter_tensor_data(item)
