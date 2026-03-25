"""Model deployment utilities for AddaxAI.

Subprocess management, YOLOv5 version switching, and synthetic detection
generation. The main deploy_model() and classify_detections() orchestrators
remain in AddaxAI_GUI.py for now (heavily UI-coupled).

Future: an InferenceBackend interface will be defined here to support
both local and cloud-based inference.
"""

import datetime
import json
import logging
import os
import signal
import sys
from subprocess import Popen
from typing import Any, Dict

logger = logging.getLogger(__name__)


def cancel_subprocess(process: Popen) -> None:
    """Kill a running subprocess in an OS-appropriate way.

    Args:
        process: A subprocess.Popen instance to terminate.
    """
    if os.name == 'nt':
        Popen(f"TASKKILL /F /PID {process.pid} /T")
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # type: ignore[attr-defined]


def switch_yolov5_version(model_type: str, base_path: str) -> None:
    """Switch YOLOv5 version by modifying Python import paths.

    Args:
        model_type: "old models" or "new models".
        base_path: Root AddaxAI directory containing yolov5_versions/.

    Raises:
        ValueError: If model_type is not recognized.
    """
    versions_base = os.path.join(base_path, "yolov5_versions")
    if model_type == "old models":
        version_path = os.path.join(versions_base, "yolov5_old", "yolov5")
    elif model_type == "new models":
        version_path = os.path.join(versions_base, "yolov5_new", "yolov5")
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    if version_path not in sys.path:
        sys.path.insert(0, version_path)

    separator = ";" if os.name == "nt" else ":"
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    prefix = version_path + separator
    if not current_pythonpath.startswith(prefix):
        os.environ["PYTHONPATH"] = prefix + current_pythonpath


def imitate_object_detection_for_full_image_classifier(chosen_folder: str) -> None:
    """Create a synthetic detection JSON for full-image classifiers.

    Generates an image_recognition_file.json where every image has a
    single detection covering the entire frame (bbox [0,0,1,1]).
    This allows full-image classifiers to use the same pipeline as
    crop-based classifiers.

    Args:
        chosen_folder: Path to the folder containing images.
    """
    image_files = [
        f for f in os.listdir(chosen_folder)
        if f.lower().endswith(('jpg', 'jpeg', 'png'))
    ]

    result: Dict[str, Any] = {
        "images": [],
        "detection_categories": {
            "1": "animal",
            "2": "person",
            "3": "vehicle",
        },
        "info": {
            "detection_completion_time": datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S'),
            "format_version": "",
            "detector": None,
            "detector_metadata": {},
        },
    }

    for image_file in image_files:
        image_data = {
            "file": image_file,
            "detections": [{
                "category": "1",
                "conf": 1.0,
                "bbox": [0.0, 0.0, 1.0, 1.0],
            }],
        }
        result["images"].append(image_data)

    json_filename = os.path.join(chosen_folder, "image_recognition_file.json")
    with open(json_filename, "w") as f:
        json.dump(result, f, indent=4)


def extract_label_map_from_model(model_file: str) -> Dict[Any, str]:
    """Load a detection model and extract its class label map.

    Imports PTDetector, loads the model on CPU, reads model.names,
    and returns the label map dict. The detector is deleted after
    extraction to free memory.

    Args:
        model_file: Path to the model weights file (.pt).

    Returns:
        Dictionary mapping class IDs to class name strings.

    Raises:
        Exception: If the model cannot be loaded or class names cannot
            be read. The caller is responsible for showing any error UI.
    """
    from cameratraps.megadetector.detection.pytorch_detector import PTDetector

    detector = PTDetector(model_file, force_cpu=True)

    try:
        label_map = {}  # type: Dict[Any, str]
        for id in detector.model.names:
            label_map[id] = detector.model.names[id]
    except Exception as error:
        logger.error("Failed to extract label map: %s", error, exc_info=True)
        del detector
        raise

    del detector
    logger.debug("Label map: %s", label_map)
    return label_map
