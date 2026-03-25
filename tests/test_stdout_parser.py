"""Tests for subprocess stdout parsers (Step B5, TDD red phase).

Written BEFORE the implementation.

Tests verify both parse_detection_stdout() and parse_classification_stdout():
  - Error result codes returned on known error lines
  - GPU detection lines update hware kwarg in emit_progress calls
  - tqdm progress lines parsed correctly (pct, cur_it, tot_it)
  - Exception/warning lines routed to log callbacks
  - Warning exclusion patterns are NOT logged
  - Frame extraction mode suppresses tqdm parsing until frames done
  - Classification <EA-status-change> updates status kwarg
  - Classification <EA> lines trigger smooth_handler callback
  - Classification "n_crops_to_classify is zero" returns "no_crops"
  - Empty stdout returns "complete" without error
  - update_ui called once per line in the loop
"""

from unittest.mock import MagicMock, call

import pytest


# ─── helpers ────────────────────────────────────────────────────────────────

def _noop(*args, **kwargs):
    pass


# Realistic tqdm progress line (the code checks '%' in line[0:4])
TQDM_LINE_50 = " 50%|████      |  5/10 [00:05<00:05,  1.0it/s]"
TQDM_LINE_100 = "100%|██████████| 10/10 [00:10<00:00,  1.0it/s]"


# ─── importability ──────────────────────────────────────────────────────────

def test_parse_detection_stdout_importable():
    from addaxai.orchestration.stdout_parser import parse_detection_stdout  # noqa


def test_parse_classification_stdout_importable():
    from addaxai.orchestration.stdout_parser import parse_classification_stdout  # noqa


# ─── signature helpers ──────────────────────────────────────────────────────

def _det_parse(lines, data_type="img", update_ui=None, emit_progress=None,
               emit_error=None, log_line=None, log_exception=None,
               log_warning=None, previous_processed_img="",
               frame_video_choice=None, cancel_func=None):
    from addaxai.orchestration.stdout_parser import parse_detection_stdout
    return parse_detection_stdout(
        stdout_lines=lines,
        data_type=data_type,
        update_ui=update_ui or _noop,
        emit_progress=emit_progress or _noop,
        emit_error=emit_error or _noop,
        log_line=log_line or _noop,
        log_exception=log_exception or _noop,
        log_warning=log_warning or _noop,
        previous_processed_img=previous_processed_img,
        frame_video_choice=frame_video_choice,
        cancel_func=cancel_func or _noop,
    )


def _cls_parse(lines, data_type="img", update_ui=None, emit_progress=None,
               emit_error=None, log_line=None, smooth_handler=None,
               cancel_func=None):
    from addaxai.orchestration.stdout_parser import parse_classification_stdout
    return parse_classification_stdout(
        stdout_lines=lines,
        data_type=data_type,
        update_ui=update_ui or _noop,
        emit_progress=emit_progress or _noop,
        emit_error=emit_error or _noop,
        log_line=log_line or _noop,
        smooth_handler=smooth_handler or _noop,
        cancel_func=cancel_func or _noop,
    )


# ═══════════════════════════════════════════════════════════════════════════
# parse_detection_stdout
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectionEmptyStdout:
    def test_empty_stdout_returns_complete(self):
        result = _det_parse(lines=[])
        assert result == "complete"

    def test_empty_stdout_emit_progress_done(self):
        """After the loop, a final 'done' progress event must be emitted."""
        emit_progress = MagicMock()
        _det_parse(lines=[], emit_progress=emit_progress, data_type="img")
        calls_with_done = [c for c in emit_progress.call_args_list
                           if c.kwargs.get("status") == "done"]
        assert len(calls_with_done) == 1, (
            f"Expected exactly one 'done' progress emit, got: {emit_progress.call_args_list}")


class TestDetectionErrorLines:
    def test_no_images_found_returns_error_code(self):
        lines = ["No image files found in /some/folder\n"]
        result = _det_parse(lines=lines, data_type="img")
        assert result == "no_images"

    def test_no_images_found_emits_error(self):
        emit_error = MagicMock()
        lines = ["No image files found in /some/folder\n"]
        _det_parse(lines=lines, data_type="img", emit_error=emit_error)
        assert emit_error.called
        kwargs = emit_error.call_args.kwargs
        assert "No image files found" in kwargs.get("message", "")

    def test_no_videos_found_returns_error_code(self):
        lines = ["No videos found in /some/folder\n"]
        result = _det_parse(lines=lines, data_type="vid")
        assert result == "no_videos"

    def test_no_frames_extracted_returns_error_code(self):
        lines = ["No frames extracted from /some/video.mp4\n"]
        result = _det_parse(lines=lines, data_type="vid")
        assert result == "no_frames"

    def test_unicode_error_returns_error_code(self):
        lines = ["UnicodeEncodeError: 'charmap' codec can't encode character\n"]
        result = _det_parse(lines=lines, data_type="img")
        assert result == "unicode_error"


class TestDetectionTqdmParsing:
    def test_tqdm_pct_parsed_correctly(self):
        """emit_progress must be called with pct=50.0 for a 50% tqdm line."""
        emit_progress = MagicMock()
        _det_parse(lines=[TQDM_LINE_50 + "\n"], emit_progress=emit_progress)
        tqdm_calls = [c for c in emit_progress.call_args_list
                      if c.kwargs.get("status") == "running"]
        assert tqdm_calls, "No 'running' emit_progress call found"
        assert tqdm_calls[0].kwargs["pct"] == pytest.approx(50.0)

    def test_tqdm_cur_it_parsed_correctly(self):
        emit_progress = MagicMock()
        _det_parse(lines=[TQDM_LINE_50 + "\n"], emit_progress=emit_progress)
        tqdm_calls = [c for c in emit_progress.call_args_list
                      if c.kwargs.get("status") == "running"]
        assert tqdm_calls[0].kwargs["cur_it"] == 5

    def test_tqdm_tot_it_parsed_correctly(self):
        emit_progress = MagicMock()
        _det_parse(lines=[TQDM_LINE_50 + "\n"], emit_progress=emit_progress)
        tqdm_calls = [c for c in emit_progress.call_args_list
                      if c.kwargs.get("status") == "running"]
        assert tqdm_calls[0].kwargs["tot_it"] == 10


class TestDetectionGpuDetection:
    def test_gpu_available_true_sets_hware_gpu(self):
        emit_progress = MagicMock()
        lines = [
            "GPU available: True\n",
            TQDM_LINE_50 + "\n",
        ]
        _det_parse(lines=lines, emit_progress=emit_progress)
        tqdm_calls = [c for c in emit_progress.call_args_list
                      if c.kwargs.get("status") == "running"]
        assert tqdm_calls[0].kwargs["hware"] == "GPU"

    def test_gpu_available_false_sets_hware_cpu(self):
        emit_progress = MagicMock()
        lines = [
            "GPU available: False\n",
            TQDM_LINE_50 + "\n",
        ]
        _det_parse(lines=lines, emit_progress=emit_progress)
        tqdm_calls = [c for c in emit_progress.call_args_list
                      if c.kwargs.get("status") == "running"]
        assert tqdm_calls[0].kwargs["hware"] == "CPU"


class TestDetectionLogging:
    def test_exception_line_sent_to_log_exception(self):
        log_exception = MagicMock()
        lines = ["Exception: something went wrong\n"]
        _det_parse(lines=lines, log_exception=log_exception)
        assert log_exception.called

    def test_warning_line_sent_to_log_warning(self):
        log_warning = MagicMock()
        lines = ["Warning: some unusual condition\n"]
        _det_parse(lines=lines, log_warning=log_warning)
        assert log_warning.called

    def test_excluded_warning_not_logged(self):
        """4 specific warning patterns must NOT be sent to log_warning."""
        log_warning = MagicMock()
        excluded = [
            "Warning: could not determine MegaDetector version\n",
            "Warning: no metadata for unknown detector version\n",
            "Warning: using user-supplied image size\n",
            "Warning: already exists and will be overwritten\n",
        ]
        for line in excluded:
            _det_parse(lines=[line], log_warning=log_warning)
        assert not log_warning.called, (
            f"log_warning should not be called for excluded patterns, "
            f"but was called with: {log_warning.call_args_list}")


class TestDetectionFrameExtraction:
    def test_extracting_frames_line_emits_progress_with_status(self):
        """'Extracting frames for folder' triggers a progress event with
        status='extracting frames' when data_type=='vid'."""
        emit_progress = MagicMock()
        lines = ["Extracting frames for folder /some/vid_folder\n"]
        _det_parse(lines=lines, data_type="vid", emit_progress=emit_progress)
        extracting_calls = [c for c in emit_progress.call_args_list
                            if c.kwargs.get("status") == "extracting frames"]
        assert extracting_calls, (
            f"Expected a progress emit with status='extracting frames', "
            f"got: {emit_progress.call_args_list}")

    def test_tqdm_during_extraction_includes_extracting_frames_txt(self):
        """A tqdm line seen while extracting_frames_mode=True should emit
        with extracting_frames_txt kwarg set."""
        emit_progress = MagicMock()
        lines = [
            "Extracting frames for folder /some/vid\n",
            " 30%|███       |  3/10 [00:03<00:07,  1.0it/s]\n",
        ]
        _det_parse(lines=lines, data_type="vid", emit_progress=emit_progress)
        # Find any emit that has extracting_frames_txt
        ext_txt_calls = [c for c in emit_progress.call_args_list
                         if "extracting_frames_txt" in c.kwargs]
        assert ext_txt_calls, (
            f"Expected emit with extracting_frames_txt, got: {emit_progress.call_args_list}")


# ═══════════════════════════════════════════════════════════════════════════
# parse_classification_stdout
# ═══════════════════════════════════════════════════════════════════════════

class TestClassificationEmptyStdout:
    def test_empty_stdout_returns_complete(self):
        result = _cls_parse(lines=[])
        assert result == "complete"

    def test_empty_stdout_emits_done_progress(self):
        emit_progress = MagicMock()
        _cls_parse(lines=[], emit_progress=emit_progress)
        done_calls = [c for c in emit_progress.call_args_list
                      if c.kwargs.get("status") == "done"]
        assert len(done_calls) == 1


class TestClassificationNoCrops:
    def test_no_crops_line_returns_no_crops(self):
        lines = ["n_crops_to_classify is zero. Nothing to classify.\n"]
        result = _cls_parse(lines=lines)
        assert result == "no_crops"

    def test_no_crops_line_emits_error(self):
        emit_error = MagicMock()
        lines = ["n_crops_to_classify is zero. Nothing to classify.\n"]
        _cls_parse(lines=lines, emit_error=emit_error)
        assert emit_error.called


class TestClassificationEALines:
    def test_ea_line_triggers_smooth_handler(self):
        """<EA>content<EA> lines must invoke smooth_handler with the content."""
        smooth_handler = MagicMock()
        lines = ["<EA>some smoothed output line<EA>\n"]
        _cls_parse(lines=lines, smooth_handler=smooth_handler)
        assert smooth_handler.called
        # The content passed should be the text between the <EA> tags
        called_with = smooth_handler.call_args[0][0]
        assert "some smoothed output line" in called_with

    def test_ea_status_change_updates_status_in_progress(self):
        """<EA-status-change>new_status<EA-status-change> must update the
        status kwarg in subsequent emit_progress calls."""
        emit_progress = MagicMock()
        lines = [
            "<EA-status-change>smoothing<EA-status-change>\n",
            TQDM_LINE_50 + "\n",
        ]
        _cls_parse(lines=lines, emit_progress=emit_progress)
        tqdm_calls = [c for c in emit_progress.call_args_list
                      if c.kwargs.get("pct") == pytest.approx(50.0)]
        assert tqdm_calls, f"No tqdm progress emit found in: {emit_progress.call_args_list}"
        assert tqdm_calls[0].kwargs.get("status") == "smoothing"


class TestClassificationTqdmParsing:
    def test_cls_tqdm_pct_parsed(self):
        emit_progress = MagicMock()
        _cls_parse(lines=[TQDM_LINE_100 + "\n"], emit_progress=emit_progress)
        # At 100% the tqdm parse should fire (status is still "running" at that point)
        running_calls = [c for c in emit_progress.call_args_list
                         if c.kwargs.get("pct") == pytest.approx(100.0)
                         and c.kwargs.get("status") != "done"]
        assert running_calls, f"No 100% tqdm call found in: {emit_progress.call_args_list}"
        assert running_calls[0].kwargs["cur_it"] == 10
        assert running_calls[0].kwargs["tot_it"] == 10
