"""Tests for JSON schema validation (addaxai/schemas/)."""

import json
import os
from typing import Any, Dict

import pytest

from addaxai.schemas.validate import (
    validate_global_vars,
    validate_model_vars,
    validate_recognition_output,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


class TestGlobalVarsValidation:
    """Tests for global_vars.json validation."""

    def test_valid_global_vars(self) -> None:
        """Valid global_vars fixture should pass."""
        with open(os.path.join(FIXTURES_DIR, "global_vars_valid.json")) as f:
            data = json.load(f)
        is_valid, errors = validate_global_vars(data)
        assert is_valid, f"Valid global_vars failed validation: {errors}"
        assert errors == []

    def test_invalid_global_vars_wrong_type(self) -> None:
        """global_vars with wrong types should fail."""
        with open(os.path.join(FIXTURES_DIR, "global_vars_invalid.json")) as f:
            data = json.load(f)
        is_valid, errors = validate_global_vars(data)
        assert not is_valid
        # Should have errors for lang_idx (string instead of int)
        # and var_thresh (1.5 out of range)
        assert any("lang_idx" in err for err in errors)
        assert any("var_thresh" in err for err in errors)

    def test_invalid_global_vars_unknown_key(self) -> None:
        """global_vars with unknown key should fail."""
        with open(os.path.join(FIXTURES_DIR, "global_vars_invalid.json")) as f:
            data = json.load(f)
        is_valid, errors = validate_global_vars(data)
        assert not is_valid
        assert any("unknown_key" in err for err in errors)

    def test_global_vars_missing_required_key(self) -> None:
        """global_vars missing required keys should fail."""
        data = {"lang_idx": 0}  # Missing almost all required keys
        is_valid, errors = validate_global_vars(data)
        assert not is_valid
        # Should complain about multiple missing required keys
        assert any("Required" in err for err in errors)

    def test_global_vars_all_keys_present(self) -> None:
        """Check that valid fixture includes all required keys."""
        with open(os.path.join(FIXTURES_DIR, "global_vars_valid.json")) as f:
            data = json.load(f)
        # Just verify it loads and has expected structure
        assert "lang_idx" in data
        assert "var_thresh" in data
        assert isinstance(data["lang_idx"], int)
        assert isinstance(data["var_thresh"], float)

    def test_global_vars_threshold_constraints(self) -> None:
        """Test that threshold values respect 0.0-1.0 range."""
        data_valid = {"lang_idx": 0, "var_thresh": 0.5, "var_thresh_default": 0.3}
        data_invalid_low = {"lang_idx": 0, "var_thresh": -0.1, "var_thresh_default": 0.3}
        data_invalid_high = {"lang_idx": 0, "var_thresh": 1.5, "var_thresh_default": 0.3}

        # Complete the data with all required fields for validation
        with open(os.path.join(FIXTURES_DIR, "global_vars_valid.json")) as f:
            complete_data = json.load(f)

        # Test low threshold
        complete_data["var_thresh"] = -0.1
        is_valid, errors = validate_global_vars(complete_data)
        assert not is_valid
        assert any("var_thresh" in err for err in errors)

        # Test high threshold
        complete_data["var_thresh"] = 1.5
        is_valid, errors = validate_global_vars(complete_data)
        assert not is_valid
        assert any("var_thresh" in err for err in errors)


class TestModelVarsValidation:
    """Tests for model variables.json validation."""

    def test_valid_model_vars_minimal(self) -> None:
        """Minimal valid model_vars should pass."""
        data: Dict[str, Any] = {
            "model_fname": "weights.pth",
            "all_classes": ["lion", "zebra"],
        }
        is_valid, errors = validate_model_vars(data)
        assert is_valid, f"Valid model_vars failed: {errors}"
        assert errors == []

    def test_valid_model_vars_comprehensive(self) -> None:
        """Comprehensive model_vars with all fields should pass."""
        data = {
            "model_type": "SpeciesNet",
            "framework": "pytorch",
            "model_fname": "model.pth",
            "all_classes": ["zebra", "lion", "giraffe"],
            "var_cls_class_thresh": 0.5,
            "var_cls_class_thresh_default": 0.5,
            "developer": "Example Developer",
            "description": "Example model",
            "info_url": "https://example.com",
        }
        is_valid, errors = validate_model_vars(data)
        assert is_valid, f"Valid comprehensive model_vars failed: {errors}"

    def test_model_vars_unknown_key(self) -> None:
        """model_vars with unknown key should fail."""
        data = {
            "model_fname": "weights.pth",
            "all_classes": ["lion"],
            "unknown_model_field": "should_fail",
        }
        is_valid, errors = validate_model_vars(data)
        assert not is_valid
        assert any("unknown_model_field" in err for err in errors)

    def test_model_vars_invalid_framework(self) -> None:
        """model_vars with invalid framework should fail."""
        data = {
            "model_fname": "weights.pth",
            "framework": "caffe",  # Invalid, not in enum
            "all_classes": ["lion"],
        }
        is_valid, errors = validate_model_vars(data)
        assert not is_valid
        assert any("framework" in err for err in errors)

    def test_model_vars_threshold_out_of_range(self) -> None:
        """model_vars with out-of-range threshold should fail."""
        data = {
            "model_fname": "weights.pth",
            "all_classes": ["lion"],
            "var_cls_class_thresh": 1.5,  # Out of range
        }
        is_valid, errors = validate_model_vars(data)
        assert not is_valid
        assert any("var_cls_class_thresh" in err for err in errors)

    def test_model_vars_classes_wrong_type(self) -> None:
        """model_vars with wrong type for all_classes should fail."""
        data = {
            "model_fname": "weights.pth",
            "all_classes": "not_an_array",  # Should be array
        }
        is_valid, errors = validate_model_vars(data)
        assert not is_valid
        assert any("all_classes" in err for err in errors)


class TestRecognitionOutputValidation:
    """Tests for recognition output JSON validation."""

    def test_valid_recognition_output(self) -> None:
        """Valid recognition output fixture should pass."""
        with open(os.path.join(FIXTURES_DIR, "recognition_output_valid.json")) as f:
            data = json.load(f)
        is_valid, errors = validate_recognition_output(data)
        assert is_valid, f"Valid recognition output failed: {errors}"
        assert errors == []

    def test_invalid_recognition_output(self) -> None:
        """Invalid recognition output should fail with appropriate errors."""
        with open(os.path.join(FIXTURES_DIR, "recognition_output_invalid.json")) as f:
            data = json.load(f)
        is_valid, errors = validate_recognition_output(data)
        assert not is_valid
        # Should have errors for wrong types
        assert len(errors) > 0

    def test_recognition_output_missing_required_fields(self) -> None:
        """Recognition output missing required fields should fail."""
        data = {"images": []}  # Missing detection_categories
        is_valid, errors = validate_recognition_output(data)
        assert not is_valid
        assert any("detection_categories" in err for err in errors)

    def test_recognition_output_images_not_array(self) -> None:
        """Recognition output with non-array images should fail."""
        data = {
            "images": "not_an_array",
            "detection_categories": {"1": "animal"},
        }
        is_valid, errors = validate_recognition_output(data)
        assert not is_valid
        assert any("images" in err for err in errors)

    def test_recognition_output_detection_categories_not_object(self) -> None:
        """Recognition output with non-object detection_categories should fail."""
        data = {
            "images": [],
            "detection_categories": ["animal", "person"],  # Should be object
        }
        is_valid, errors = validate_recognition_output(data)
        assert not is_valid
        assert any("detection_categories" in err for err in errors)

    def test_recognition_output_category_value_not_string(self) -> None:
        """Recognition output with non-string category values should fail."""
        data = {
            "images": [],
            "detection_categories": {"1": 123},  # Should be string
        }
        is_valid, errors = validate_recognition_output(data)
        assert not is_valid
        assert any("detection_categories" in err for err in errors)

    def test_recognition_output_with_classifications(self) -> None:
        """Recognition output with valid classifications should pass."""
        data = {
            "images": [
                {
                    "file": "test.jpg",
                    "detections": [
                        {
                            "category": "1",
                            "conf": 0.9,
                            "bbox": [0.0, 0.0, 1.0, 1.0],
                            "classifications": [
                                ["lion", 0.95],
                                ["zebra", 0.03],
                            ],
                        }
                    ],
                }
            ],
            "detection_categories": {"1": "animal"},
        }
        is_valid, errors = validate_recognition_output(data)
        assert is_valid, f"Valid output with classifications failed: {errors}"

    def test_recognition_output_empty_detections(self) -> None:
        """Recognition output with empty detections array should pass."""
        data = {
            "images": [
                {
                    "file": "empty.jpg",
                    "detections": [],
                }
            ],
            "detection_categories": {"1": "animal"},
        }
        is_valid, errors = validate_recognition_output(data)
        assert is_valid, f"Valid output with empty detections failed: {errors}"

    def test_recognition_output_multiple_images(self) -> None:
        """Recognition output with multiple images should validate each."""
        data = {
            "images": [
                {
                    "file": "img1.jpg",
                    "detections": [
                        {"category": "1", "conf": 0.9, "bbox": [0, 0, 1, 1]},
                    ],
                },
                {
                    "file": "img2.jpg",
                    "detections": [],
                },
            ],
            "detection_categories": {"1": "animal"},
        }
        is_valid, errors = validate_recognition_output(data)
        assert is_valid, f"Multi-image output failed: {errors}"
