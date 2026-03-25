"""Tests for run_detection() pipeline function (Step B6, TDD red phase).

Written BEFORE the implementation.

Tests verify:
  - DetectionResult and run_detection are importable
  - DetectionResult is a dataclass with success, json_path, error_code, error_message
  - run_detection emits DEPLOY_STARTED event
  - Empty stdout → success=True, DEPLOY_FINISHED emitted, on_error not called
  - 'No image files found' → error_code='no_images', on_error called
  - 'No videos found' → error_code='no_videos'
  - 'No frames extracted' → error_code='no_frames'
  - 'UnicodeEncodeError:' → error_code='unicode_error'
  - cancel_check() returns True → error_code='cancelled', DEPLOY_CANCELLED emitted
  - cancel_func_factory called with the Popen process instance
"""

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from addaxai.i18n import init as i18n_init

# Initialise i18n once for all tests in this module (English).
i18n_init(0)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_popen_mock(stdout_lines):
    """Return a Popen mock whose stdout is an iterable of the given lines."""
    mock_proc = MagicMock()
    mock_proc.stdout = iter(stdout_lines)
    return mock_proc


@pytest.fixture
def basic_config():
    from addaxai.orchestration.context import DeployConfig
    return DeployConfig(
        base_path="/tmp/addaxai",
        det_model_dir="/tmp/addaxai/models/det",
        det_model_name="MegaDetector 5a",
        det_model_path="",
        cls_model_name="None",       # English "None" == t('none') after i18n_init(0)
        disable_gpu=False,
        use_abs_paths=False,
        source_folder="/tmp/nonexistent_images",
        # "Custom model" must be the last entry so "MegaDetector 5a" is not custom
        dpd_options_model=[["MegaDetector 5a", "Custom model"],
                           ["MegaDetector 5a", "Modelo personalizado"]],
        lang_idx=0,
    )


@pytest.fixture
def headless_callbacks():
    from addaxai.orchestration.callbacks import OrchestratorCallbacks
    return OrchestratorCallbacks(
        on_error=MagicMock(),
        on_warning=MagicMock(),
        on_info=MagicMock(),
        on_confirm=MagicMock(return_value=True),
        update_ui=MagicMock(),
        cancel_check=MagicMock(return_value=False),
    )


def _load_vars_side_effect(base_path, model_type, model_dir):
    """Return minimal model vars; cls always returns {} (no full_image_cls)."""
    if model_type == "det":
        return {"model_fname": "md_v5a.0.0.pt"}
    return {}


def _run(config, callbacks, stdout_lines, data_type="img",
         simple_mode=False, cancel_check_return=False, **extra_kwargs):
    """Convenience wrapper around run_detection() with standard mocks applied."""
    from addaxai.orchestration.pipeline import run_detection
    callbacks.cancel_check.return_value = cancel_check_return
    mock_proc = _make_popen_mock(stdout_lines)

    with patch("addaxai.orchestration.pipeline.Popen", return_value=mock_proc), \
         patch("addaxai.orchestration.pipeline.load_model_vars_for",
               side_effect=_load_vars_side_effect), \
         patch("addaxai.orchestration.pipeline.switch_yolov5_version"), \
         patch("addaxai.orchestration.pipeline.get_python_interpreter",
               return_value="/path/to/python"):
        return run_detection(
            config=config,
            callbacks=callbacks,
            data_type=data_type,
            selected_options=[],
            simple_mode=simple_mode,
            cancel_func_factory=lambda p: (lambda: None),
            error_log_path="/tmp/errors.log",
            warning_log_path="/tmp/warnings.log",
            current_version="5.0",
            **extra_kwargs,
        )


# ─── importability ───────────────────────────────────────────────────────────

def test_run_detection_importable():
    from addaxai.orchestration.pipeline import run_detection  # noqa: F401


def test_detection_result_importable():
    from addaxai.orchestration.pipeline import DetectionResult  # noqa: F401


# ─── DetectionResult dataclass ──────────────────────────────────────────────

class TestDetectionResult:
    def test_is_dataclass(self):
        from addaxai.orchestration.pipeline import DetectionResult
        assert dataclasses.is_dataclass(DetectionResult)

    def test_has_required_fields(self):
        from addaxai.orchestration.pipeline import DetectionResult
        fields = {f.name for f in dataclasses.fields(DetectionResult)}
        assert {"success", "json_path", "error_code", "error_message"}.issubset(fields)

    def test_can_instantiate_success(self):
        from addaxai.orchestration.pipeline import DetectionResult
        r = DetectionResult(
            success=True, json_path="/path/image_recognition_file.json",
            error_code=None, error_message=None,
        )
        assert r.success is True
        assert r.error_code is None

    def test_can_instantiate_failure(self):
        from addaxai.orchestration.pipeline import DetectionResult
        r = DetectionResult(
            success=False, json_path=None,
            error_code="no_images", error_message="No image files found",
        )
        assert r.success is False
        assert r.error_code == "no_images"


# ─── success path ────────────────────────────────────────────────────────────

class TestRunDetectionSuccess:
    def test_empty_stdout_returns_success(self, basic_config, headless_callbacks):
        result = _run(basic_config, headless_callbacks, stdout_lines=[])
        assert result.success is True

    def test_empty_stdout_error_code_is_none(self, basic_config, headless_callbacks):
        result = _run(basic_config, headless_callbacks, stdout_lines=[])
        assert result.error_code is None

    def test_empty_stdout_error_message_is_none(self, basic_config, headless_callbacks):
        result = _run(basic_config, headless_callbacks, stdout_lines=[])
        assert result.error_message is None

    def test_on_error_not_called_on_success(self, basic_config, headless_callbacks):
        _run(basic_config, headless_callbacks, stdout_lines=[])
        assert not headless_callbacks.on_error.called

    def test_json_path_set_on_success_img(self, basic_config, headless_callbacks):
        """json_path should be the image_recognition_file path on img success."""
        result = _run(basic_config, headless_callbacks, stdout_lines=[], data_type="img")
        assert result.json_path is not None
        assert "image_recognition_file.json" in result.json_path

    def test_json_path_set_on_success_vid(self, basic_config, headless_callbacks):
        result = _run(basic_config, headless_callbacks, stdout_lines=[], data_type="vid")
        assert result.json_path is not None
        assert "video_recognition_file.json" in result.json_path


# ─── error paths ─────────────────────────────────────────────────────────────

class TestRunDetectionErrors:
    def test_no_images_found_returns_failure(self, basic_config, headless_callbacks):
        result = _run(basic_config, headless_callbacks,
                      stdout_lines=["No image files found in /folder\n"])
        assert result.success is False
        assert result.error_code == "no_images"

    def test_no_images_found_calls_on_error(self, basic_config, headless_callbacks):
        _run(basic_config, headless_callbacks,
             stdout_lines=["No image files found in /folder\n"])
        assert headless_callbacks.on_error.called

    def test_no_videos_found_returns_failure(self, basic_config, headless_callbacks):
        result = _run(basic_config, headless_callbacks,
                      stdout_lines=["No videos found in /folder\n"],
                      data_type="vid")
        assert result.success is False
        assert result.error_code == "no_videos"

    def test_no_videos_calls_on_error(self, basic_config, headless_callbacks):
        _run(basic_config, headless_callbacks,
             stdout_lines=["No videos found in /folder\n"], data_type="vid")
        assert headless_callbacks.on_error.called

    def test_no_frames_extracted_returns_failure(self, basic_config, headless_callbacks):
        result = _run(basic_config, headless_callbacks,
                      stdout_lines=["No frames extracted from /video.mp4\n"],
                      data_type="vid")
        assert result.success is False
        assert result.error_code == "no_frames"

    def test_no_frames_calls_on_error(self, basic_config, headless_callbacks):
        _run(basic_config, headless_callbacks,
             stdout_lines=["No frames extracted from /video.mp4\n"], data_type="vid")
        assert headless_callbacks.on_error.called

    def test_unicode_error_returns_failure(self, basic_config, headless_callbacks):
        result = _run(basic_config, headless_callbacks,
                      stdout_lines=["UnicodeEncodeError: 'charmap' codec\n"])
        assert result.success is False
        assert result.error_code == "unicode_error"

    def test_unicode_error_calls_on_error(self, basic_config, headless_callbacks):
        _run(basic_config, headless_callbacks,
             stdout_lines=["UnicodeEncodeError: 'charmap' codec\n"])
        assert headless_callbacks.on_error.called


# ─── cancellation ─────────────────────────────────────────────────────────────

class TestRunDetectionCancel:
    def test_cancel_check_true_returns_failure(self, basic_config, headless_callbacks):
        result = _run(basic_config, headless_callbacks,
                      stdout_lines=[], cancel_check_return=True)
        assert result.success is False
        assert result.error_code == "cancelled"

    def test_cancel_check_true_no_on_error_called(self, basic_config, headless_callbacks):
        """Cancellation should not trigger on_error — it's not an error."""
        _run(basic_config, headless_callbacks,
             stdout_lines=[], cancel_check_return=True)
        assert not headless_callbacks.on_error.called


# ─── events ──────────────────────────────────────────────────────────────────

class TestRunDetectionEvents:
    def test_deploy_started_emitted(self, basic_config, headless_callbacks):
        from addaxai.core.events import event_bus
        from addaxai.core.event_types import DEPLOY_STARTED
        with patch.object(event_bus, "emit") as mock_emit:
            _run(basic_config, headless_callbacks, stdout_lines=[])
        started = [c for c in mock_emit.call_args_list
                   if c.args and c.args[0] == DEPLOY_STARTED]
        assert started, "DEPLOY_STARTED was not emitted"

    def test_deploy_finished_emitted_on_success(self, basic_config, headless_callbacks):
        from addaxai.core.events import event_bus
        from addaxai.core.event_types import DEPLOY_FINISHED
        with patch.object(event_bus, "emit") as mock_emit:
            _run(basic_config, headless_callbacks, stdout_lines=[])
        finished = [c for c in mock_emit.call_args_list
                    if c.args and c.args[0] == DEPLOY_FINISHED]
        assert finished, "DEPLOY_FINISHED was not emitted on success"

    def test_deploy_cancelled_emitted_when_cancelled(self, basic_config, headless_callbacks):
        from addaxai.core.events import event_bus
        from addaxai.core.event_types import DEPLOY_CANCELLED
        with patch.object(event_bus, "emit") as mock_emit:
            _run(basic_config, headless_callbacks,
                 stdout_lines=[], cancel_check_return=True)
        cancelled = [c for c in mock_emit.call_args_list
                     if c.args and c.args[0] == DEPLOY_CANCELLED]
        assert cancelled, "DEPLOY_CANCELLED was not emitted when cancelled"

    def test_deploy_started_carries_process_kwarg(self, basic_config, headless_callbacks):
        from addaxai.core.events import event_bus
        from addaxai.core.event_types import DEPLOY_STARTED
        with patch.object(event_bus, "emit") as mock_emit:
            _run(basic_config, headless_callbacks, stdout_lines=[], data_type="img")
        started = [c for c in mock_emit.call_args_list
                   if c.args and c.args[0] == DEPLOY_STARTED]
        assert started[0].kwargs.get("process") == "img_det"


# ─── cancel_func_factory ─────────────────────────────────────────────────────

class TestRunDetectionCancelFuncFactory:
    def test_cancel_func_factory_called_with_popen_process(
            self, basic_config, headless_callbacks):
        """cancel_func_factory must be invoked with the Popen instance."""
        factory_calls = []

        def _factory(p):
            factory_calls.append(p)
            return lambda: None

        mock_proc = _make_popen_mock([])

        from addaxai.orchestration.pipeline import run_detection
        with patch("addaxai.orchestration.pipeline.Popen", return_value=mock_proc), \
             patch("addaxai.orchestration.pipeline.load_model_vars_for",
                   side_effect=_load_vars_side_effect), \
             patch("addaxai.orchestration.pipeline.switch_yolov5_version"), \
             patch("addaxai.orchestration.pipeline.get_python_interpreter",
                   return_value="/path/to/python"):
            run_detection(
                config=basic_config,
                callbacks=headless_callbacks,
                data_type="img",
                selected_options=[],
                simple_mode=False,
                cancel_func_factory=_factory,
                error_log_path="/tmp/errors.log",
                warning_log_path="/tmp/warnings.log",
                current_version="5.0",
            )

        assert factory_calls, "cancel_func_factory was never called"
        assert factory_calls[0] is mock_proc, (
            f"cancel_func_factory was called with {factory_calls[0]!r}, "
            f"expected the mock Popen process {mock_proc!r}"
        )


# =============================================================================
# run_classification tests (Step B7)
# =============================================================================

# ─── cls helpers ──────────────────────────────────────────────────────────────

_CLS_MODEL_VARS = {
    "model_fname": "model.pt",
    "type": "EfficientDet",
    "env": "env-cls",
    "var_cls_detec_thresh_default": 0.5,
    "var_cls_class_thresh_default": 0.5,
    "var_tax_fallback_default": False,
    "var_tax_levels_idx": 0,
}


def _load_cls_vars_side_effect(base_path, model_type, model_dir):
    if model_type == "cls":
        return _CLS_MODEL_VARS
    return {}


@pytest.fixture
def cls_config():
    from addaxai.orchestration.context import ClassifyConfig
    return ClassifyConfig(
        base_path="/tmp/addaxai",
        cls_model_name="EuroMammals",
        disable_gpu=False,
        cls_detec_thresh=0.6,
        cls_class_thresh=0.6,
        smooth_cls_animal=True,
        tax_fallback=False,
        temp_frame_folder="",
        lang_idx=0,
    )


def _run_cls(config, callbacks, stdout_lines, data_type="img",
             simple_mode=False, **extra_kwargs):
    """Convenience wrapper around run_classification() with standard mocks."""
    from addaxai.orchestration.pipeline import run_classification
    mock_proc = _make_popen_mock(stdout_lines)

    with patch("addaxai.orchestration.pipeline.Popen", return_value=mock_proc), \
         patch("addaxai.orchestration.pipeline.load_model_vars_for",
               side_effect=_load_cls_vars_side_effect), \
         patch("addaxai.orchestration.pipeline.taxon_mapping_csv_present",
               return_value=False), \
         patch("addaxai.orchestration.pipeline.get_python_interpreter",
               return_value="/path/to/python"):
        return run_classification(
            config=config,
            callbacks=callbacks,
            json_fpath="/tmp/addaxai/image_recognition_file.json",
            data_type=data_type,
            cancel_func_factory=lambda p: (lambda: None),
            simple_mode=simple_mode,
            **extra_kwargs,
        )


# ─── importability ────────────────────────────────────────────────────────────

def test_run_classification_importable():
    from addaxai.orchestration.pipeline import run_classification  # noqa: F401


def test_classification_result_importable():
    from addaxai.orchestration.pipeline import ClassificationResult  # noqa: F401


# ─── ClassificationResult dataclass ──────────────────────────────────────────

class TestClassificationResult:
    def test_is_dataclass(self):
        import dataclasses
        from addaxai.orchestration.pipeline import ClassificationResult
        assert dataclasses.is_dataclass(ClassificationResult)

    def test_has_required_fields(self):
        import dataclasses
        from addaxai.orchestration.pipeline import ClassificationResult
        fields = {f.name for f in dataclasses.fields(ClassificationResult)}
        assert {"success", "json_path", "error_code", "error_message"}.issubset(fields)

    def test_can_instantiate_success(self):
        from addaxai.orchestration.pipeline import ClassificationResult
        r = ClassificationResult(
            success=True, json_path="/path/image_recognition_file.json",
            error_code=None, error_message=None,
        )
        assert r.success is True
        assert r.error_code is None

    def test_can_instantiate_failure(self):
        from addaxai.orchestration.pipeline import ClassificationResult
        r = ClassificationResult(
            success=False, json_path=None,
            error_code="no_crops", error_message="No crops to classify",
        )
        assert r.success is False
        assert r.error_code == "no_crops"


# ─── success path ─────────────────────────────────────────────────────────────

class TestRunClassificationSuccess:
    def test_empty_stdout_returns_success(self, cls_config, headless_callbacks):
        result = _run_cls(cls_config, headless_callbacks, stdout_lines=[])
        assert result.success is True

    def test_empty_stdout_error_code_is_none(self, cls_config, headless_callbacks):
        result = _run_cls(cls_config, headless_callbacks, stdout_lines=[])
        assert result.error_code is None

    def test_empty_stdout_json_path_equals_input(self, cls_config, headless_callbacks):
        result = _run_cls(cls_config, headless_callbacks, stdout_lines=[])
        assert result.json_path == "/tmp/addaxai/image_recognition_file.json"

    def test_on_error_not_called_on_success(self, cls_config, headless_callbacks):
        _run_cls(cls_config, headless_callbacks, stdout_lines=[])
        assert not headless_callbacks.on_error.called


# ─── no_crops error path ──────────────────────────────────────────────────────

class TestRunClassificationNoCrops:
    def test_no_crops_returns_failure(self, cls_config, headless_callbacks):
        result = _run_cls(
            cls_config, headless_callbacks,
            stdout_lines=["n_crops_to_classify is zero. Nothing to classify.\n"],
        )
        assert result.success is False
        assert result.error_code == "no_crops"

    def test_no_crops_calls_on_info(self, cls_config, headless_callbacks):
        _run_cls(
            cls_config, headless_callbacks,
            stdout_lines=["n_crops_to_classify is zero. Nothing to classify.\n"],
        )
        assert headless_callbacks.on_info.called

    def test_no_crops_does_not_call_on_error(self, cls_config, headless_callbacks):
        _run_cls(
            cls_config, headless_callbacks,
            stdout_lines=["n_crops_to_classify is zero. Nothing to classify.\n"],
        )
        assert not headless_callbacks.on_error.called


# ─── events ───────────────────────────────────────────────────────────────────

class TestRunClassificationEvents:
    def test_classify_started_emitted(self, cls_config, headless_callbacks):
        from addaxai.core.events import event_bus
        from addaxai.core.event_types import CLASSIFY_STARTED
        with patch.object(event_bus, "emit") as mock_emit:
            _run_cls(cls_config, headless_callbacks, stdout_lines=[])
        started = [c for c in mock_emit.call_args_list
                   if c.args and c.args[0] == CLASSIFY_STARTED]
        assert started, "CLASSIFY_STARTED was not emitted"

    def test_classify_finished_emitted_on_success(self, cls_config, headless_callbacks):
        from addaxai.core.events import event_bus
        from addaxai.core.event_types import CLASSIFY_FINISHED
        with patch.object(event_bus, "emit") as mock_emit:
            _run_cls(cls_config, headless_callbacks, stdout_lines=[])
        finished = [c for c in mock_emit.call_args_list
                    if c.args and c.args[0] == CLASSIFY_FINISHED]
        assert finished, "CLASSIFY_FINISHED was not emitted on success"

    def test_classify_started_carries_process_kwarg(self, cls_config, headless_callbacks):
        from addaxai.core.events import event_bus
        from addaxai.core.event_types import CLASSIFY_STARTED
        with patch.object(event_bus, "emit") as mock_emit:
            _run_cls(cls_config, headless_callbacks, stdout_lines=[], data_type="img")
        started = [c for c in mock_emit.call_args_list
                   if c.args and c.args[0] == CLASSIFY_STARTED]
        assert started[0].kwargs.get("process") == "img_cls"


# ─── cancel_func_factory ──────────────────────────────────────────────────────

class TestRunClassificationCancelFuncFactory:
    def test_cancel_func_factory_called_with_popen_process(
            self, cls_config, headless_callbacks):
        """cancel_func_factory must be invoked with the Popen instance."""
        factory_calls = []

        def _factory(p):
            factory_calls.append(p)
            return lambda: None

        mock_proc = _make_popen_mock([])

        from addaxai.orchestration.pipeline import run_classification
        with patch("addaxai.orchestration.pipeline.Popen", return_value=mock_proc), \
             patch("addaxai.orchestration.pipeline.load_model_vars_for",
                   side_effect=_load_cls_vars_side_effect), \
             patch("addaxai.orchestration.pipeline.taxon_mapping_csv_present",
                   return_value=False), \
             patch("addaxai.orchestration.pipeline.get_python_interpreter",
                   return_value="/path/to/python"):
            run_classification(
                config=cls_config,
                callbacks=headless_callbacks,
                json_fpath="/tmp/addaxai/image_recognition_file.json",
                data_type="img",
                cancel_func_factory=_factory,
                simple_mode=False,
            )

        assert factory_calls, "cancel_func_factory was never called"
        assert factory_calls[0] is mock_proc, (
            f"cancel_func_factory was called with {factory_calls[0]!r}, "
            f"expected the mock Popen process {mock_proc!r}"
        )
