"""Tests for addaxai.api.server — REST API endpoints."""

import json
import os
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from addaxai.api.server import app


def test_health_endpoint() -> None:
    """GET /health should return status ok."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models_with_env_var() -> None:
    """GET /models should list available models when ADDAXAI_FILES is set."""
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create model directories
        det_dir = Path(tmp_dir) / "models" / "det"
        cls_dir = Path(tmp_dir) / "models" / "cls"
        det_dir.mkdir(parents=True)
        cls_dir.mkdir(parents=True)

        # Set environment variable
        old_val = os.environ.get("ADDAXAI_FILES")
        os.environ["ADDAXAI_FILES"] = tmp_dir

        try:
            response = client.get("/models")
            assert response.status_code == 200
            data = response.json()
            assert "detection" in data
            assert "classification" in data
            assert isinstance(data["detection"], list)
            assert isinstance(data["classification"], list)
        finally:
            if old_val is None:
                os.environ.pop("ADDAXAI_FILES", None)
            else:
                os.environ["ADDAXAI_FILES"] = old_val


def test_list_models_without_env_var() -> None:
    """GET /models should return 500 when ADDAXAI_FILES is not set."""
    client = TestClient(app)
    old_val = os.environ.pop("ADDAXAI_FILES", None)

    try:
        response = client.get("/models")
        assert response.status_code == 500
        assert "ADDAXAI_FILES environment variable not set" in response.json()[
            "detail"
        ]
    finally:
        if old_val is not None:
            os.environ["ADDAXAI_FILES"] = old_val


def test_get_results_with_fixture() -> None:
    """GET /results/{folder} should return recognition JSON when it exists."""
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create results file
        results_dir = Path(tmp_dir) / "test_folder"
        results_dir.mkdir(parents=True)
        results_file = results_dir / "image_recognition_file.json"
        test_data = {
            "images": [{"file": "test.jpg", "detections": []}],
            "detection_categories": {"1": "animal"},
        }
        with open(results_file, "w") as f:
            json.dump(test_data, f)

        # Set environment variable
        old_val = os.environ.get("ADDAXAI_FILES")
        os.environ["ADDAXAI_FILES"] = tmp_dir

        try:
            response = client.get("/results/test_folder")
            assert response.status_code == 200
            assert response.json() == test_data
        finally:
            if old_val is None:
                os.environ.pop("ADDAXAI_FILES", None)
            else:
                os.environ["ADDAXAI_FILES"] = old_val


def test_get_results_missing_folder() -> None:
    """GET /results/{missing} should return 404 when results don't exist."""
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Set environment variable
        old_val = os.environ.get("ADDAXAI_FILES")
        os.environ["ADDAXAI_FILES"] = tmp_dir

        try:
            response = client.get("/results/nonexistent")
            assert response.status_code == 404
            assert "Results not found" in response.json()["detail"]
        finally:
            if old_val is None:
                os.environ.pop("ADDAXAI_FILES", None)
            else:
                os.environ["ADDAXAI_FILES"] = old_val


def test_get_results_without_env_var() -> None:
    """GET /results/{folder} should return 500 when ADDAXAI_FILES is not set."""
    client = TestClient(app)
    old_val = os.environ.pop("ADDAXAI_FILES", None)

    try:
        response = client.get("/results/some_folder")
        assert response.status_code == 500
        assert "ADDAXAI_FILES environment variable not set" in response.json()[
            "detail"
        ]
    finally:
        if old_val is not None:
            os.environ["ADDAXAI_FILES"] = old_val


def test_api_server_instantiation() -> None:
    """FastAPI app should be properly instantiated."""
    assert app.title == "AddaxAI API"
    assert app.version == "0.1.0"
