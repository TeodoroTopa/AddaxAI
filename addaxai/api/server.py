"""Local REST API server for AddaxAI.

Start with: uvicorn addaxai.api.server:app --port 6189
"""

import json
import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="AddaxAI API",
    description="Local API for camera trap image classification",
    version="0.1.0",
)


def _get_base_path() -> str:
    """Resolve AddaxAI_files path."""
    # Check environment variable first, then default locations
    base = os.environ.get("ADDAXAI_FILES")
    if base and os.path.isdir(base):
        return base
    raise HTTPException(
        status_code=500,
        detail="ADDAXAI_FILES environment variable not set",
    )


@app.get("/models")
def list_models() -> Dict[str, List[str]]:
    """List available detection and classification models."""
    from addaxai.models.registry import fetch_known_models
    from addaxai.core.paths import get_det_dir, get_cls_dir

    base = _get_base_path()
    det_dir = get_det_dir(base)
    cls_dir = get_cls_dir(base)
    return {
        "detection": fetch_known_models(det_dir),
        "classification": fetch_known_models(cls_dir),
    }


@app.get("/results/{folder_name}")
def get_results(folder_name: str) -> Dict[str, Any]:
    """Get recognition results for a processed folder."""
    base = _get_base_path()
    json_path = os.path.join(base, folder_name, "image_recognition_file.json")
    if not os.path.isfile(json_path):
        raise HTTPException(status_code=404, detail="Results not found")
    with open(json_path, "r") as f:
        return json.load(f)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
