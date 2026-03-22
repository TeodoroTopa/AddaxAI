# ruff: noqa: F841 — This template has intentionally unused variables that users will fill in
"""Template classification adapter for AddaxAI.

**DO NOT MODIFY THIS FILE.** Copy this directory to create a new model adapter.

This script is invoked as a subprocess by AddaxAI:

    python classify_detections.py <AddaxAI_files> <model_path> <det_thresh> \\
        <cls_thresh> <smooth> <json_path> <temp_frame_folder> \\
        <tax_fallback> <tax_levels_idx>

Arguments (positional, via sys.argv):
    1. AddaxAI_files       - Root path to AddaxAI installation (base_path)
    2. cls_model_fpath    - Path to model checkpoint/weights file
    3. cls_detec_thresh   - Detection confidence threshold (float, 0.0-1.0)
    4. cls_class_thresh   - Classification confidence threshold (float, 0.0-1.0)
    5. smooth_bool        - "True"/"False" — smooth predictions across image series
    6. json_path          - Path to input/output recognition JSON file
    7. temp_frame_folder  - Path to temp video frames or "None" if not processing video
    8. cls_tax_fallback   - "True"/"False" — use taxonomic fallback for low-conf predictions
    9. cls_tax_levels_idx - Taxonomy level index (int, 0 or higher)

Exit behavior:
    - Exit 0 if classification succeeds
    - Exit 1 if fatal error (stops the run)
    - Print warnings/info to stdout (captured in log)

Output:
    - Modifies json_path in-place: adds "classifications" array to each detection
    - Format per detection: "classifications": [["species_name", confidence], ...]
"""

import json
import sys
from pathlib import Path
from PIL import ImageFile

# ============================================================================
# ESSENTIAL: Fix truncated image handling (required by AddaxAI)
# ============================================================================
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main() -> int:
    """Main entry point — parse args, load model, classify, write output."""
    try:
        # ====================================================================
        # Parse CLI arguments (FIXED for all adapters — do not change)
        # ====================================================================
        if len(sys.argv) < 10:
            print("ERROR: Missing arguments", file=sys.stderr)
            return 1

        AddaxAI_files = str(sys.argv[1])
        cls_model_fpath = str(sys.argv[2])
        cls_detec_thresh = float(sys.argv[3])
        cls_class_thresh = float(sys.argv[4])
        smooth_bool = True if sys.argv[5] == "True" else False
        json_path = str(sys.argv[6])
        temp_frame_folder = None if sys.argv[7] == "None" else str(sys.argv[7])
        cls_tax_fallback = True if sys.argv[8] == "True" else False
        cls_tax_levels_idx = int(sys.argv[9])

        # ====================================================================
        # TODO: Import your model framework
        # ====================================================================
        # Uncomment and adapt for your model:
        # import torch
        # import tensorflow as tf
        # from your_model_lib import load_model
        # etc.

        # ====================================================================
        # TODO: Load the model
        # ====================================================================
        # Example:
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # model = torch.load(cls_model_fpath, map_location=device)
        # model.eval()
        #
        # OR for TensorFlow:
        # model = tf.keras.models.load_model(cls_model_fpath)
        #
        # OR for ONNX:
        # import onnxruntime as ort
        # session = ort.InferenceSession(cls_model_fpath)

        # For now, just demonstrate the structure (this is a no-op template)
        model = None  # Replace with actual model loading

        # ====================================================================
        # Read the input recognition JSON
        # ====================================================================
        if not Path(json_path).exists():
            print(f"ERROR: Recognition JSON not found: {json_path}", file=sys.stderr)
            return 1

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        images = data.get("images", [])
        if not images:
            print("No images in recognition JSON, nothing to classify", file=sys.stderr)
            return 0

        # ====================================================================
        # TODO: Classify detections
        # ====================================================================
        # Pseudo-code — adapt to your model:
        #
        # from PIL import Image
        # import os
        #
        # for image_entry in images:
        #     image_path = image_entry["file"]
        #     if not os.path.isabs(image_path):
        #         image_path = os.path.join(AddaxAI_files, image_path)
        #
        #     try:
        #         img = Image.open(image_path)
        #     except Exception as e:
        #         print(f"WARNING: Could not load image {image_path}: {e}")
        #         continue
        #
        #     for detection in image_entry.get("detections", []):
        #         # Skip low-confidence detections
        #         if detection.get("conf", 0) < cls_detec_thresh:
        #             continue
        #
        #         # Extract bounding box and crop image
        #         bbox = detection.get("bbox", [0, 0, 1, 1])
        #         x_min, y_min, x_max, y_max = bbox
        #         w, h = img.size
        #         crop_box = (int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h))
        #         crop = img.crop(crop_box)
        #
        #         # Run model on crop
        #         # predictions = model.predict(crop)  # adapt to your API
        #         #
        #         # Filter by confidence threshold
        #         # top_predictions = [
        #         #     [label, conf] for label, conf in predictions
        #         #     if conf >= cls_class_thresh
        #         # ]
        #         #
        #         # Add to detection
        #         # if top_predictions:
        #         #     detection["classifications"] = top_predictions

        # ====================================================================
        # Write the output JSON with classifications added
        # ====================================================================
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=True)

        return 0

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
