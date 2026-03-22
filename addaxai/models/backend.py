"""Inference backend protocol — the contract all model adapters must fulfill.

This module defines the InferenceBackend protocol that standardizes how detection
and classification models are called. All inference backends (local subprocess,
cloud API, ONNX runtime, etc.) must implement this interface.

Current implementations:
- LocalSubprocessBackend (subprocess in isolated conda env) — each model adapter
  in classification_utils/model_types/

Future implementations:
- CloudBackend (HuggingFace Endpoints / Replicate API)
- ONNXBackend (local ONNX runtime, no conda env needed)
"""

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class InferenceBackend(Protocol):
    """Protocol that all inference backends must implement.

    An inference backend encapsulates how detection and classification models
    are executed. Each backend can have different requirements (conda envs,
    API keys, hardware) but must expose the same interface.
    """

    def detect(
        self,
        image_paths: List[str],
        model_path: str,
        confidence_threshold: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run object detection on a list of images.

        Args:
            image_paths: List of image file paths to process.
            model_path: Path to the model checkpoint or weights.
            confidence_threshold: Minimum confidence to report detections.
            **kwargs: Backend-specific options (e.g., batch_size, device).

        Returns:
            Recognition output JSON with keys:
            - "images": List of {file, detections} dicts
            - "detection_categories": {id: name} mapping
            - "info": Optional metadata dict
        """
        ...

    def classify(
        self,
        crops: List[Dict[str, Any]],
        model_path: str,
        confidence_threshold: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run species classification on detection crops.

        Args:
            crops: List of crop dicts with keys:
                - "image_path": Path to the original image
                - "bbox": [x_min, y_min, x_max, y_max] or cropped image array
                - "category": Original detection category
            model_path: Path to the model checkpoint.
            confidence_threshold: Minimum confidence to report classifications.
            **kwargs: Backend-specific options (e.g., batch_size, device).

        Returns:
            Updated crops dict with "classifications" field added to each crop:
            [["species_name", confidence], [...], ...]
        """
        ...

    def is_available(self) -> bool:
        """Check if this backend is ready to use.

        Returns True if:
        - Model weights are downloaded and valid
        - Environment/dependencies are installed
        - Hardware (GPU) is accessible (if required)

        Returns:
            True if backend can run inference, False otherwise.
        """
        ...
