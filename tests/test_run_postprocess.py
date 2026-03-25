"""Tests for run_postprocess() pipeline function (Step B8, TDD red phase).

Written BEFORE the implementation.

Tests verify:
  - PostprocessResult and run_postprocess are importable
  - PostprocessResult is a dataclass with success, error_code, error_message fields
  - No recognition JSON found -> on_error called, success=False, error_code="no_json"
  - Invalid dest dir -> on_error called, success=False, error_code="invalid_dest"
  - img_json present -> _postprocess_inner called with data_type="img"
  - vid_json present -> _postprocess_inner called with data_type="vid"
  - Both jsons present -> _postprocess_inner called twice
  - POSTPROCESS_STARTED emitted
  - POSTPROCESS_FINISHED emitted on success
  - POSTPROCESS_ERROR emitted on exception
  - Warning shown if error_log file exists after processing
  - cancel_func forwarded to _postprocess_inner
"""

import dataclasses
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from addaxai.i18n import init as i18n_init

# Initialise i18n once for all tests in this module (English).
i18n_init(0)


# ─── helpers ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dirs():
    """Create real tmp src and dst directories for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "source")
        dst = os.path.join(tmpdir, "dest")
        os.makedirs(src)
        os.makedirs(dst)
        yield src, dst


@pytest.fixture
def ppcfg(tmp_dirs):
    """A minimal PostprocessConfig pointing at real tmp dirs."""
    from addaxai.orchestration.context import PostprocessConfig
    src, dst = tmp_dirs
    return PostprocessConfig(
        source_folder=src,
        dest_folder=dst,
        thresh=0.2,
        separate_files=False,
        file_placement=2,
        sep_conf=False,
        vis=False,
        crp=False,
        exp=False,
        plt=False,
        exp_format="CSV",
        data_type="img",
        vis_blur=False,
        vis_bbox=True,
        vis_size_idx=0,
        keep_series=False,
        keep_series_seconds=5.0,
        keep_series_species=[],
        current_version="5.0",
        lang_idx=0,
    )


@pytest.fixture
def headless_cb():
    from addaxai.orchestration.callbacks import OrchestratorCallbacks
    return OrchestratorCallbacks(
        on_error=MagicMock(),
        on_warning=MagicMock(),
        on_info=MagicMock(),
        on_confirm=MagicMock(return_value=True),
        update_ui=MagicMock(),
        cancel_check=MagicMock(return_value=False),
    )


def _cancel_func():
    pass


def _no_plots(results_dir):
    pass


def _run_pp(ppcfg, headless_cb, img=False, vid=False,
            inner_side_effect=None, **extra_kwargs):
    """Call run_postprocess() with standard mocks.

    img/vid: create the respective recognition JSON file in src.
    inner_side_effect: if set, used as side_effect for _postprocess_inner mock.
    """
    from addaxai.orchestration.pipeline import run_postprocess

    if img:
        Path(os.path.join(ppcfg.source_folder, "image_recognition_file.json")).write_text(
            '{"images": [], "detection_categories": {}, "addaxai_metadata": {}}')
    if vid:
        Path(os.path.join(ppcfg.source_folder, "video_recognition_file.json")).write_text(
            '{"images": [], "detection_categories": {}, "addaxai_metadata": {}}')

    inner_mock = MagicMock(return_value=None)
    if inner_side_effect is not None:
        inner_mock.side_effect = inner_side_effect

    with patch("addaxai.orchestration.pipeline._postprocess_inner",
               inner_mock) as mock_inner:
        result = run_postprocess(
            config=ppcfg,
            callbacks=headless_cb,
            cancel_func=_cancel_func,
            produce_plots_func=_no_plots,
            **extra_kwargs,
        )

    return result, mock_inner


# ─── importability ────────────────────────────────────────────────────────────

def test_run_postprocess_importable():
    from addaxai.orchestration.pipeline import run_postprocess  # noqa: F401


def test_postprocess_result_importable():
    from addaxai.orchestration.pipeline import PostprocessResult  # noqa: F401


# ─── PostprocessResult dataclass ─────────────────────────────────────────────

class TestPostprocessResult:
    def test_is_dataclass(self):
        from addaxai.orchestration.pipeline import PostprocessResult
        assert dataclasses.is_dataclass(PostprocessResult)

    def test_has_required_fields(self):
        from addaxai.orchestration.pipeline import PostprocessResult
        fields = {f.name for f in dataclasses.fields(PostprocessResult)}
        assert {"success", "error_code", "error_message"}.issubset(fields)

    def test_can_instantiate_success(self):
        from addaxai.orchestration.pipeline import PostprocessResult
        r = PostprocessResult(success=True, error_code=None, error_message=None)
        assert r.success is True

    def test_can_instantiate_failure(self):
        from addaxai.orchestration.pipeline import PostprocessResult
        r = PostprocessResult(success=False, error_code="no_json",
                              error_message="No recognition files found")
        assert r.success is False
        assert r.error_code == "no_json"


# ─── validation: no JSON file ─────────────────────────────────────────────────

class TestRunPostprocessNoJson:
    def test_no_json_returns_failure(self, ppcfg, headless_cb):
        result, _ = _run_pp(ppcfg, headless_cb, img=False, vid=False)
        assert result.success is False
        assert result.error_code == "no_json"

    def test_no_json_calls_on_error(self, ppcfg, headless_cb):
        _run_pp(ppcfg, headless_cb, img=False, vid=False)
        assert headless_cb.on_error.called

    def test_no_json_inner_not_called(self, ppcfg, headless_cb):
        _, mock_inner = _run_pp(ppcfg, headless_cb, img=False, vid=False)
        assert not mock_inner.called


# ─── validation: invalid dest dir ────────────────────────────────────────────

class TestRunPostprocessInvalidDest:
    def test_invalid_dest_returns_failure(self, ppcfg, headless_cb):
        ppcfg.dest_folder = ""
        result, _ = _run_pp(ppcfg, headless_cb, img=True)
        assert result.success is False
        assert result.error_code == "invalid_dest"

    def test_invalid_dest_calls_on_error(self, ppcfg, headless_cb):
        ppcfg.dest_folder = ""
        _run_pp(ppcfg, headless_cb, img=True)
        assert headless_cb.on_error.called

    def test_nonexistent_dest_returns_failure(self, ppcfg, headless_cb):
        ppcfg.dest_folder = "/nonexistent/path/that/does/not/exist"
        result, _ = _run_pp(ppcfg, headless_cb, img=True)
        assert result.success is False
        assert result.error_code == "invalid_dest"


# ─── success path ─────────────────────────────────────────────────────────────

class TestRunPostprocessSuccess:
    def test_img_json_calls_inner_with_img(self, ppcfg, headless_cb):
        _, mock_inner = _run_pp(ppcfg, headless_cb, img=True)
        data_types = [c.kwargs.get("data_type") or
                      (c.args[13] if len(c.args) > 13 else None)
                      for c in mock_inner.call_args_list]
        assert "img" in data_types

    def test_vid_json_calls_inner_with_vid(self, ppcfg, headless_cb):
        _, mock_inner = _run_pp(ppcfg, headless_cb, vid=True)
        data_types = [c.kwargs.get("data_type") or
                      (c.args[13] if len(c.args) > 13 else None)
                      for c in mock_inner.call_args_list]
        assert "vid" in data_types

    def test_both_jsons_inner_called_twice(self, ppcfg, headless_cb):
        _, mock_inner = _run_pp(ppcfg, headless_cb, img=True, vid=True)
        assert mock_inner.call_count == 2

    def test_success_returns_true(self, ppcfg, headless_cb):
        result, _ = _run_pp(ppcfg, headless_cb, img=True)
        assert result.success is True

    def test_on_error_not_called_on_success(self, ppcfg, headless_cb):
        _run_pp(ppcfg, headless_cb, img=True)
        assert not headless_cb.on_error.called


# ─── events ───────────────────────────────────────────────────────────────────

class TestRunPostprocessEvents:
    def test_postprocess_started_emitted(self, ppcfg, headless_cb):
        from addaxai.core.events import event_bus
        from addaxai.core.event_types import POSTPROCESS_STARTED
        with patch.object(event_bus, "emit") as mock_emit, \
             patch("addaxai.orchestration.pipeline._postprocess_inner"):
            # create img json
            Path(os.path.join(ppcfg.source_folder,
                              "image_recognition_file.json")).write_text(
                '{"images": [], "detection_categories": {}, "addaxai_metadata": {}}')
            run_postprocess = __import__(
                "addaxai.orchestration.pipeline", fromlist=["run_postprocess"]
            ).run_postprocess
            run_postprocess(config=ppcfg, callbacks=headless_cb,
                            cancel_func=_cancel_func)
        started = [c for c in mock_emit.call_args_list
                   if c.args and c.args[0] == POSTPROCESS_STARTED]
        assert started, "POSTPROCESS_STARTED was not emitted"

    def test_postprocess_finished_emitted_on_success(self, ppcfg, headless_cb):
        from addaxai.core.events import event_bus
        from addaxai.core.event_types import POSTPROCESS_FINISHED
        from addaxai.orchestration.pipeline import run_postprocess
        Path(os.path.join(ppcfg.source_folder,
                          "image_recognition_file.json")).write_text(
            '{"images": [], "detection_categories": {}, "addaxai_metadata": {}}')
        with patch.object(event_bus, "emit") as mock_emit, \
             patch("addaxai.orchestration.pipeline._postprocess_inner"):
            run_postprocess(config=ppcfg, callbacks=headless_cb,
                            cancel_func=_cancel_func)
        finished = [c for c in mock_emit.call_args_list
                    if c.args and c.args[0] == POSTPROCESS_FINISHED]
        assert finished, "POSTPROCESS_FINISHED was not emitted on success"

    def test_postprocess_error_emitted_on_exception(self, ppcfg, headless_cb):
        from addaxai.core.events import event_bus
        from addaxai.core.event_types import POSTPROCESS_ERROR
        from addaxai.orchestration.pipeline import run_postprocess
        Path(os.path.join(ppcfg.source_folder,
                          "image_recognition_file.json")).write_text(
            '{"images": [], "detection_categories": {}, "addaxai_metadata": {}}')
        with patch.object(event_bus, "emit") as mock_emit, \
             patch("addaxai.orchestration.pipeline._postprocess_inner",
                   side_effect=RuntimeError("boom")):
            result = run_postprocess(config=ppcfg, callbacks=headless_cb,
                                     cancel_func=_cancel_func)
        err_events = [c for c in mock_emit.call_args_list
                      if c.args and c.args[0] == POSTPROCESS_ERROR]
        assert err_events, "POSTPROCESS_ERROR was not emitted on exception"
        assert result.success is False


# ─── warning on error log ──────────────────────────────────────────────────────

class TestRunPostprocessErrorLogWarning:
    def test_warning_shown_if_error_log_exists(self, ppcfg, headless_cb):
        """If _postprocess_inner writes an error log, on_warning is called."""
        from addaxai.orchestration.pipeline import run_postprocess
        src, dst = ppcfg.source_folder, ppcfg.dest_folder
        Path(os.path.join(src, "image_recognition_file.json")).write_text(
            '{"images": [], "detection_categories": {}, "addaxai_metadata": {}}')
        error_log = os.path.join(dst, "postprocessing_error_log.txt")

        def _inner_creates_error_log(*args, **kwargs):
            # Simulate _postprocess_inner writing an error log
            Path(error_log).write_text("some error\n")

        with patch("addaxai.orchestration.pipeline._postprocess_inner",
                   side_effect=_inner_creates_error_log):
            run_postprocess(config=ppcfg, callbacks=headless_cb,
                            cancel_func=_cancel_func)

        assert headless_cb.on_warning.called


# ─── cancel_func forwarded ────────────────────────────────────────────────────

class TestRunPostprocessCancelFunc:
    def test_cancel_func_forwarded_to_inner(self, ppcfg, headless_cb):
        """cancel_func must be forwarded to _postprocess_inner."""
        from addaxai.orchestration.pipeline import run_postprocess
        Path(os.path.join(ppcfg.source_folder,
                          "image_recognition_file.json")).write_text(
            '{"images": [], "detection_categories": {}, "addaxai_metadata": {}}')

        received_cancel_funcs = []

        def _capture_inner(*args, **kwargs):
            received_cancel_funcs.append(kwargs.get("cancel_func"))

        with patch("addaxai.orchestration.pipeline._postprocess_inner",
                   side_effect=_capture_inner):
            run_postprocess(config=ppcfg, callbacks=headless_cb,
                            cancel_func=_cancel_func)

        assert received_cancel_funcs, "_postprocess_inner was never called"
        assert received_cancel_funcs[0] is _cancel_func
