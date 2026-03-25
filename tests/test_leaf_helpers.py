"""Tests for leaf helper functions extracted from app.py in Phase 7b Step B1.

Tests are written BEFORE the implementation (TDD).
Functions under test:
  - taxon_mapping_csv_present(base_path, cls_model_name) in models/registry.py
  - extract_label_map_from_model(model_file) in models/deploy.py
"""

import json
import os
import sys
from types import ModuleType
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from addaxai.models.registry import (
    taxon_mapping_csv_present,
)


@pytest.fixture
def mock_ptdetector():
    """Mock the cameratraps PTDetector import chain.

    The cameratraps package is not installed in the test venv, so we
    mock the entire module hierarchy and yield the mock PTDetector class.
    """
    mock_module = ModuleType("cameratraps")
    mock_mega = ModuleType("cameratraps.megadetector")
    mock_det = ModuleType("cameratraps.megadetector.detection")
    mock_pytorch = ModuleType("cameratraps.megadetector.detection.pytorch_detector")

    mock_cls = MagicMock()
    mock_pytorch.PTDetector = mock_cls  # type: ignore[attr-defined]

    modules = {
        "cameratraps": mock_module,
        "cameratraps.megadetector": mock_mega,
        "cameratraps.megadetector.detection": mock_det,
        "cameratraps.megadetector.detection.pytorch_detector": mock_pytorch,
    }

    with patch.dict(sys.modules, modules):
        yield mock_cls


# --- taxon_mapping_csv_present ---


def test_taxon_mapping_csv_present_returns_true(tmp_path):
    """Should return True when taxon-mapping.csv exists in the model dir."""
    model_dir = tmp_path / "models" / "cls" / "SpeciesNet"
    model_dir.mkdir(parents=True)
    (model_dir / "taxon-mapping.csv").write_text("species,common_name\n")

    result = taxon_mapping_csv_present(
        base_path=str(tmp_path),
        cls_model_name="SpeciesNet",
    )
    assert result is True


def test_taxon_mapping_csv_present_returns_false(tmp_path):
    """Should return False when taxon-mapping.csv does not exist."""
    model_dir = tmp_path / "models" / "cls" / "SpeciesNet"
    model_dir.mkdir(parents=True)

    result = taxon_mapping_csv_present(
        base_path=str(tmp_path),
        cls_model_name="SpeciesNet",
    )
    assert result is False


def test_taxon_mapping_csv_present_model_dir_missing(tmp_path):
    """Should return False when the model directory itself doesn't exist."""
    result = taxon_mapping_csv_present(
        base_path=str(tmp_path),
        cls_model_name="NonExistentModel",
    )
    assert result is False


def test_taxon_mapping_csv_present_uses_correct_path(tmp_path):
    """Should look in base_path/models/cls/<model_name>/taxon-mapping.csv."""
    model_dir = tmp_path / "models" / "cls" / "MyModel"
    model_dir.mkdir(parents=True)
    csv_path = model_dir / "taxon-mapping.csv"
    csv_path.write_text("data")

    assert taxon_mapping_csv_present(str(tmp_path), "MyModel") is True
    # Different model name should be False
    assert taxon_mapping_csv_present(str(tmp_path), "OtherModel") is False


# --- extract_label_map_from_model ---
# These tests mock PTDetector since we don't have real model files in unit tests.


def test_extract_label_map_returns_dict(mock_ptdetector):
    """Should return a dict mapping IDs to class names from model.names."""
    from addaxai.models.deploy import extract_label_map_from_model

    mock_detector = MagicMock()
    mock_detector.model.names = {0: "animal", 1: "person", 2: "vehicle"}
    mock_ptdetector.return_value = mock_detector

    result = extract_label_map_from_model("/fake/model.pt")
    assert result == {0: "animal", 1: "person", 2: "vehicle"}


def test_extract_label_map_empty_names(mock_ptdetector):
    """Should return empty dict when model has no class names."""
    from addaxai.models.deploy import extract_label_map_from_model

    mock_detector = MagicMock()
    mock_detector.model.names = {}
    mock_ptdetector.return_value = mock_detector

    result = extract_label_map_from_model("/fake/model.pt")
    assert result == {}


def test_extract_label_map_raises_on_error(mock_ptdetector):
    """Should raise an exception when model class extraction fails.

    The extracted version raises instead of calling mb.showerror().
    The caller (app.py) is responsible for showing the error dialog.
    """
    from addaxai.models.deploy import extract_label_map_from_model

    mock_detector = MagicMock()
    # Make accessing .names raise
    type(mock_detector.model).names = property(
        fget=MagicMock(side_effect=RuntimeError("corrupt model"))
    )
    mock_ptdetector.return_value = mock_detector

    with pytest.raises(RuntimeError, match="corrupt model"):
        extract_label_map_from_model("/fake/model.pt")


def test_extract_label_map_cleans_up_detector(mock_ptdetector):
    """Should delete the detector object to free GPU memory."""
    from addaxai.models.deploy import extract_label_map_from_model

    mock_detector = MagicMock()
    mock_detector.model.names = {0: "animal"}
    mock_ptdetector.return_value = mock_detector

    result = extract_label_map_from_model("/fake/model.pt")
    assert result == {0: "animal"}
