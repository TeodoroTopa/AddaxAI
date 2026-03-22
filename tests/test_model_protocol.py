"""Tests for model adapter protocol (InferenceBackend)."""

import os
from typing import Any, Dict

import pytest

from addaxai.models.backend import InferenceBackend


class MockInferenceBackend:
    """Concrete implementation of InferenceBackend for testing."""

    def detect(
        self,
        image_paths: list,
        model_path: str,
        confidence_threshold: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Mock detection implementation."""
        return {
            "images": [{"file": path, "detections": []} for path in image_paths],
            "detection_categories": {"1": "animal"},
        }

    def classify(
        self,
        crops: list,
        model_path: str,
        confidence_threshold: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Mock classification implementation."""
        return {"crops": crops}

    def is_available(self) -> bool:
        """Mock availability check."""
        return True


class TestInferenceBackendProtocol:
    """Tests for InferenceBackend protocol definition."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """InferenceBackend should be a runtime_checkable Protocol."""
        mock = MockInferenceBackend()
        assert isinstance(mock, InferenceBackend)

    def test_protocol_requires_detect_method(self) -> None:
        """InferenceBackend must have detect() method."""
        mock = MockInferenceBackend()
        assert hasattr(mock, "detect")
        assert callable(mock.detect)

    def test_protocol_requires_classify_method(self) -> None:
        """InferenceBackend must have classify() method."""
        mock = MockInferenceBackend()
        assert hasattr(mock, "classify")
        assert callable(mock.classify)

    def test_protocol_requires_is_available_method(self) -> None:
        """InferenceBackend must have is_available() method."""
        mock = MockInferenceBackend()
        assert hasattr(mock, "is_available")
        assert callable(mock.is_available)

    def test_mock_detect_returns_recognition_json(self) -> None:
        """Mock detect() should return valid recognition JSON structure."""
        mock = MockInferenceBackend()
        result = mock.detect(["image1.jpg", "image2.jpg"], "/path/to/model", 0.5)

        assert isinstance(result, dict)
        assert "images" in result
        assert "detection_categories" in result
        assert isinstance(result["images"], list)
        assert isinstance(result["detection_categories"], dict)

    def test_mock_classify_returns_crops(self) -> None:
        """Mock classify() should return crops with classifications."""
        mock = MockInferenceBackend()
        crops = [{"bbox": [0, 0, 1, 1]}]
        result = mock.classify(crops, "/path/to/model", 0.5)

        assert isinstance(result, dict)
        assert "crops" in result

    def test_mock_is_available_returns_bool(self) -> None:
        """Mock is_available() should return boolean."""
        mock = MockInferenceBackend()
        result = mock.is_available()

        assert isinstance(result, bool)

    def test_incomplete_implementation_rejected(self) -> None:
        """Class missing methods should not satisfy protocol."""

        class IncompleteBackend:
            """Missing classify() and is_available()."""

            def detect(self, image_paths, model_path, confidence_threshold, **kwargs):
                return {}

        incomplete = IncompleteBackend()
        # Should not be an instance of InferenceBackend
        assert not isinstance(incomplete, InferenceBackend)

    def test_multiple_backends_can_implement_protocol(self) -> None:
        """Multiple independent classes can implement the protocol."""

        class LocalBackend:
            def detect(self, image_paths, model_path, confidence_threshold, **kwargs):
                return {"images": []}

            def classify(self, crops, model_path, confidence_threshold, **kwargs):
                return {"crops": []}

            def is_available(self):
                return True

        class CloudBackend:
            def detect(self, image_paths, model_path, confidence_threshold, **kwargs):
                # Calls cloud API instead
                return {"images": []}

            def classify(self, crops, model_path, confidence_threshold, **kwargs):
                return {"crops": []}

            def is_available(self):
                return True

        local = LocalBackend()
        cloud = CloudBackend()

        assert isinstance(local, InferenceBackend)
        assert isinstance(cloud, InferenceBackend)


class TestTemplateAdapterStructure:
    """Tests for the template classifier adapter file."""

    def test_template_adapter_exists(self) -> None:
        """Template adapter file should exist."""
        template_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "classification_utils",
            "model_types",
            "_template",
            "classify_detections.py",
        )
        assert os.path.exists(template_path)

    def test_template_adapter_is_valid_python(self) -> None:
        """Template adapter should be valid Python."""
        template_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "classification_utils",
            "model_types",
            "_template",
            "classify_detections.py",
        )
        try:
            with open(template_path, "r") as f:
                compile(f.read(), template_path, "exec")
        except SyntaxError as e:
            pytest.fail(f"Template adapter has syntax error: {e}")

    def test_template_readme_exists(self) -> None:
        """Template README should exist."""
        readme_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "classification_utils",
            "model_types",
            "_template",
            "README.md",
        )
        assert os.path.exists(readme_path)

    def test_template_readme_is_not_empty(self) -> None:
        """Template README should have content."""
        readme_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "classification_utils",
            "model_types",
            "_template",
            "README.md",
        )
        with open(readme_path, "r") as f:
            content = f.read()
        assert len(content) > 500  # Should have meaningful documentation


class TestTemplateAdapterParsing:
    """Tests for argument parsing in template adapter."""

    def test_template_handles_argv_correctly(self) -> None:
        """Template should have correct sys.argv[1:9] parsing."""
        template_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "classification_utils",
            "model_types",
            "_template",
            "classify_detections.py",
        )
        with open(template_path, "r") as f:
            content = f.read()

        # Check that template mentions all 9 arguments
        assert "sys.argv[1]" in content  # AddaxAI_files
        assert "sys.argv[2]" in content  # cls_model_fpath
        assert "sys.argv[3]" in content  # cls_detec_thresh
        assert "sys.argv[4]" in content  # cls_class_thresh
        assert "sys.argv[5]" in content  # smooth_bool
        assert "sys.argv[6]" in content  # json_path
        assert "sys.argv[7]" in content  # temp_frame_folder
        assert "sys.argv[8]" in content  # cls_tax_fallback
        assert "sys.argv[9]" in content  # cls_tax_levels_idx

    def test_template_has_todo_markers(self) -> None:
        """Template should have TODO markers for customization."""
        template_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "classification_utils",
            "model_types",
            "_template",
            "classify_detections.py",
        )
        with open(template_path, "r") as f:
            content = f.read()

        # Should have TODOs for customization
        assert "TODO: Import" in content
        assert "TODO: Load" in content
        assert "TODO: Classify" in content

    def test_template_has_json_io_structure(self) -> None:
        """Template should read and write JSON correctly."""
        template_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "classification_utils",
            "model_types",
            "_template",
            "classify_detections.py",
        )
        with open(template_path, "r") as f:
            content = f.read()

        # Should load JSON
        assert "json.load" in content or "json.loads" in content
        # Should dump JSON
        assert "json.dump" in content
        # Should reference the recognition JSON structure
        assert "images" in content
        assert "detections" in content


class TestProtocolDocumentation:
    """Tests that protocol has proper documentation."""

    def test_backend_protocol_has_docstring(self) -> None:
        """InferenceBackend protocol should be documented."""
        assert InferenceBackend.__doc__ is not None
        assert len(InferenceBackend.__doc__) > 100

    def test_detect_method_has_docstring(self) -> None:
        """detect() method should be documented."""
        assert InferenceBackend.detect.__doc__ is not None

    def test_classify_method_has_docstring(self) -> None:
        """classify() method should be documented."""
        assert InferenceBackend.classify.__doc__ is not None

    def test_is_available_method_has_docstring(self) -> None:
        """is_available() method should be documented."""
        assert InferenceBackend.is_available.__doc__ is not None
