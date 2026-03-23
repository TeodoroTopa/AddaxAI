"""Tests for UI tab event handler -> ProgressWindow wiring.

These tests verify that DeployTab and PostprocessTab event handlers
correctly forward event kwargs to progress_window.update_values().
"""

import pytest
from unittest.mock import MagicMock

from addaxai.core.events import event_bus
from addaxai.core.event_types import (
    DEPLOY_PROGRESS,
    DEPLOY_ERROR,
    DEPLOY_FINISHED,
    CLASSIFY_PROGRESS,
    POSTPROCESS_PROGRESS,
    POSTPROCESS_ERROR,
    POSTPROCESS_FINISHED,
)
from addaxai.ui.deploy_tab import DeployTab
from addaxai.ui.postprocess_tab import PostprocessTab


@pytest.fixture(autouse=True)
def clear_event_bus():
    """Clear event bus before and after each test."""
    event_bus.clear_all()
    yield
    event_bus.clear_all()


def _make_mock_app_state():
    """Create a mock app_state with a mock progress_window."""
    app_state = MagicMock()
    app_state.progress_window = MagicMock()
    app_state.progress_window.update_values = MagicMock()
    return app_state


def _make_deploy_tab(app_state):
    """Create a DeployTab (needed for event subscription side effects)."""
    return DeployTab(
        parent_frame=MagicMock(),
        start_deploy_callback=MagicMock(),
        app_state=app_state,
    )


def _make_postprocess_tab(app_state):
    """Create a PostprocessTab (needed for event subscription side effects)."""
    return PostprocessTab(parent_frame=MagicMock(), app_state=app_state)


# ============================================================================
# DeployTab -> ProgressWindow forwarding
# ============================================================================

class TestDeployTabForwarding:
    """Test that DeployTab forwards event kwargs to progress_window.update_values()."""

    def test_deploy_progress_load_status(self):
        """Test that DEPLOY_PROGRESS with status='load' is forwarded."""
        app_state = _make_mock_app_state()
        _make_deploy_tab(app_state)

        event_bus.emit(DEPLOY_PROGRESS, pct=0.0, message="Loading detection model",
                       process="img_det", status="load")

        app_state.progress_window.update_values.assert_called_once_with(
            process="img_det", status="load"
        )

    def test_deploy_progress_running_with_rich_params(self):
        """Test that DEPLOY_PROGRESS with status='running' forwards all rich params."""
        app_state = _make_mock_app_state()
        cancel_fn = MagicMock()
        _make_deploy_tab(app_state)

        event_bus.emit(DEPLOY_PROGRESS, pct=50.0, message="Processing: 5/10",
                       process="img_det", status="running",
                       cur_it=5, tot_it=10,
                       time_ela="00:05", time_rem="00:05",
                       speed="1.0it/s", hware="GPU",
                       cancel_func=cancel_fn,
                       frame_video_choice=None)

        app_state.progress_window.update_values.assert_called_once()
        kwargs = app_state.progress_window.update_values.call_args[1]
        assert kwargs['process'] == "img_det"
        assert kwargs['status'] == "running"
        assert kwargs['cur_it'] == 5
        assert kwargs['tot_it'] == 10
        assert kwargs['time_ela'] == "00:05"
        assert kwargs['time_rem'] == "00:05"
        assert kwargs['speed'] == "1.0it/s"
        assert kwargs['hware'] == "GPU"
        assert kwargs['cancel_func'] is cancel_fn
        assert kwargs['frame_video_choice'] is None

    def test_deploy_progress_done_status(self):
        """Test that DEPLOY_PROGRESS with status='done' is forwarded."""
        app_state = _make_mock_app_state()
        _make_deploy_tab(app_state)

        event_bus.emit(DEPLOY_PROGRESS, pct=100.0, message="Detection complete",
                       process="img_det", status="done")

        app_state.progress_window.update_values.assert_called_once_with(
            process="img_det", status="done"
        )

    def test_deploy_progress_extracting_frames(self):
        """Test that status='extracting frames' forwards extracting_frames_txt."""
        app_state = _make_mock_app_state()
        _make_deploy_tab(app_state)

        txt = ["Extracting frames... 50%", "Extrayendo fotogramas... 50%"]
        event_bus.emit(DEPLOY_PROGRESS, pct=50.0, message="Extracting frames...",
                       process="vid_det", status="extracting frames",
                       extracting_frames_txt=txt)

        app_state.progress_window.update_values.assert_called_once()
        kwargs = app_state.progress_window.update_values.call_args[1]
        assert kwargs['process'] == "vid_det"
        assert kwargs['status'] == "extracting frames"
        assert kwargs['extracting_frames_txt'] == txt

    def test_deploy_progress_no_forward_without_process(self):
        """Test that events without 'process' kwarg are not forwarded."""
        app_state = _make_mock_app_state()
        _make_deploy_tab(app_state)

        event_bus.emit(DEPLOY_PROGRESS, pct=50.0, message="Test")

        app_state.progress_window.update_values.assert_not_called()

    def test_deploy_progress_no_forward_without_status(self):
        """Test that events without 'status' kwarg are not forwarded."""
        app_state = _make_mock_app_state()
        _make_deploy_tab(app_state)

        event_bus.emit(DEPLOY_PROGRESS, pct=50.0, message="Test", process="img_det")

        app_state.progress_window.update_values.assert_not_called()

    def test_deploy_progress_no_forward_without_progress_window(self):
        """Test that forwarding is skipped if progress_window is None."""
        app_state = _make_mock_app_state()
        app_state.progress_window = None
        _make_deploy_tab(app_state)

        event_bus.emit(DEPLOY_PROGRESS, pct=50.0, message="Test",
                       process="img_det", status="running")

        # No exception should be raised

    def test_deploy_error_does_not_crash(self):
        """Test that DEPLOY_ERROR triggers show_error with message."""
        app_state = _make_mock_app_state()
        _make_deploy_tab(app_state)

        event_bus.emit(DEPLOY_ERROR, message="No image files found",
                       process="img_det")

        # show_error is a stub, so just verify no crash

    def test_deploy_finished_does_not_crash(self):
        """Test that DEPLOY_FINISHED triggers show_completion."""
        app_state = _make_mock_app_state()
        _make_deploy_tab(app_state)

        event_bus.emit(DEPLOY_FINISHED, results_path="/path/to/results.json",
                       process="img_det")

        # show_completion is a stub, so just verify no crash


# ============================================================================
# DeployTab classify event forwarding
# ============================================================================

class TestDeployTabClassifyForwarding:
    """Test that DeployTab handles classify events correctly."""

    def test_classify_progress_load_status(self):
        """Test that CLASSIFY_PROGRESS with status='load' is forwarded."""
        app_state = _make_mock_app_state()
        _make_deploy_tab(app_state)

        event_bus.emit(CLASSIFY_PROGRESS, pct=0.0, message="Loading classification model",
                       process="img_cls", status="load")

        app_state.progress_window.update_values.assert_called_once_with(
            process="img_cls", status="load"
        )

    def test_classify_progress_running_with_rich_params(self):
        """Test that CLASSIFY_PROGRESS with status='running' forwards rich params."""
        app_state = _make_mock_app_state()
        cancel_fn = MagicMock()
        _make_deploy_tab(app_state)

        event_bus.emit(CLASSIFY_PROGRESS, pct=75.0, message="Classifying: 6/8",
                       process="img_cls", status="running",
                       cur_it=6, tot_it=8,
                       time_ela="00:30", time_rem="00:10",
                       speed="0.8it/s", hware="GPU",
                       cancel_func=cancel_fn)

        app_state.progress_window.update_values.assert_called_once()
        kwargs = app_state.progress_window.update_values.call_args[1]
        assert kwargs['process'] == "img_cls"
        assert kwargs['status'] == "running"
        assert kwargs['cur_it'] == 6
        assert kwargs['tot_it'] == 8
        assert kwargs['cancel_func'] is cancel_fn

    def test_classify_progress_smoothing_status(self):
        """Test that CLASSIFY_PROGRESS with smoothing status is forwarded."""
        app_state = _make_mock_app_state()
        _make_deploy_tab(app_state)

        event_bus.emit(CLASSIFY_PROGRESS, pct=50.0, message="Smoothing",
                       process="vid_cls", status="smoothing",
                       cur_it=3, tot_it=6,
                       time_ela="00:15", time_rem="00:15",
                       speed="0.5it/s", hware="CPU",
                       cancel_func=MagicMock())

        kwargs = app_state.progress_window.update_values.call_args[1]
        assert kwargs['status'] == "smoothing"

    def test_classify_progress_done_status(self):
        """Test that CLASSIFY_PROGRESS with status='done' forwards timing info."""
        app_state = _make_mock_app_state()
        _make_deploy_tab(app_state)

        event_bus.emit(CLASSIFY_PROGRESS, pct=100.0, message="Classification complete",
                       process="img_cls", status="done",
                       time_ela="01:00", speed="0.8it/s")

        kwargs = app_state.progress_window.update_values.call_args[1]
        assert kwargs['process'] == "img_cls"
        assert kwargs['status'] == "done"
        assert kwargs['time_ela'] == "01:00"
        assert kwargs['speed'] == "0.8it/s"


# ============================================================================
# PostprocessTab -> ProgressWindow forwarding
# ============================================================================

class TestPostprocessTabForwarding:
    """Test that PostprocessTab forwards event kwargs to update_values()."""

    def test_postprocess_progress_load_status(self):
        """Test that POSTPROCESS_PROGRESS with status='load' is forwarded."""
        app_state = _make_mock_app_state()
        _make_postprocess_tab(app_state)

        event_bus.emit(POSTPROCESS_PROGRESS, pct=0.0, message="Initializing postprocessing",
                       process="img_pst", status="load")

        app_state.progress_window.update_values.assert_called_once_with(
            process="img_pst", status="load"
        )

    def test_postprocess_progress_running_with_rich_params(self):
        """Test that POSTPROCESS_PROGRESS with status='running' forwards all rich params."""
        app_state = _make_mock_app_state()
        cancel_fn = MagicMock()
        _make_postprocess_tab(app_state)

        event_bus.emit(POSTPROCESS_PROGRESS, pct=50.0, message="Processing: 5/10",
                       process="img_pst", status="running",
                       cur_it=5, tot_it=10,
                       time_ela="00:10", time_rem="00:10",
                       cancel_func=cancel_fn)

        app_state.progress_window.update_values.assert_called_once()
        kwargs = app_state.progress_window.update_values.call_args[1]
        assert kwargs['process'] == "img_pst"
        assert kwargs['status'] == "running"
        assert kwargs['cur_it'] == 5
        assert kwargs['tot_it'] == 10
        assert kwargs['time_ela'] == "00:10"
        assert kwargs['time_rem'] == "00:10"
        assert kwargs['cancel_func'] is cancel_fn

    def test_postprocess_progress_done_status(self):
        """Test that POSTPROCESS_PROGRESS with status='done' is forwarded."""
        app_state = _make_mock_app_state()
        _make_postprocess_tab(app_state)

        event_bus.emit(POSTPROCESS_PROGRESS, pct=100.0, message="Postprocessing complete",
                       process="img_pst", status="done")

        app_state.progress_window.update_values.assert_called_once_with(
            process="img_pst", status="done"
        )

    def test_postprocess_progress_plt_load(self):
        """Test that process='plt' with status='load' is forwarded."""
        app_state = _make_mock_app_state()
        _make_postprocess_tab(app_state)

        event_bus.emit(POSTPROCESS_PROGRESS, process="plt", status="load")

        app_state.progress_window.update_values.assert_called_once_with(
            process="plt", status="load"
        )

    def test_postprocess_progress_plt_running_with_tqdm_params(self):
        """Test that process='plt' with tqdm stats is forwarded."""
        app_state = _make_mock_app_state()
        cancel_fn = MagicMock()
        _make_postprocess_tab(app_state)

        event_bus.emit(POSTPROCESS_PROGRESS, process="plt", status="running",
                       cur_it=3, tot_it=12,
                       time_ela="00:05", time_rem="00:15",
                       cancel_func=cancel_fn)

        kwargs = app_state.progress_window.update_values.call_args[1]
        assert kwargs['process'] == "plt"
        assert kwargs['status'] == "running"
        assert kwargs['cur_it'] == 3
        assert kwargs['tot_it'] == 12
        assert kwargs['cancel_func'] is cancel_fn

    def test_postprocess_progress_plt_done(self):
        """Test that process='plt' with status='done' is forwarded."""
        app_state = _make_mock_app_state()
        _make_postprocess_tab(app_state)

        event_bus.emit(POSTPROCESS_PROGRESS, process="plt", status="done")

        app_state.progress_window.update_values.assert_called_once_with(
            process="plt", status="done"
        )

    def test_postprocess_error_does_not_crash(self):
        """Test that POSTPROCESS_ERROR does not crash."""
        app_state = _make_mock_app_state()
        _make_postprocess_tab(app_state)

        event_bus.emit(POSTPROCESS_ERROR, message="Destination folder not set or invalid")

    def test_postprocess_finished_does_not_crash(self):
        """Test that POSTPROCESS_FINISHED does not crash."""
        app_state = _make_mock_app_state()
        _make_postprocess_tab(app_state)

        event_bus.emit(POSTPROCESS_FINISHED)


# ============================================================================
# Extra kwargs filtering
# ============================================================================

class TestExtraKwargsFiltering:
    """Test that extra kwargs (pct, message) are NOT forwarded to update_values."""

    def test_pct_and_message_not_forwarded(self):
        """Test that 'pct' and 'message' are filtered out of update_values kwargs."""
        app_state = _make_mock_app_state()
        _make_deploy_tab(app_state)

        event_bus.emit(DEPLOY_PROGRESS, pct=50.0, message="Processing: 5/10",
                       process="img_det", status="running",
                       cur_it=5, tot_it=10)

        kwargs = app_state.progress_window.update_values.call_args[1]
        assert 'pct' not in kwargs
        assert 'message' not in kwargs
        assert 'process' in kwargs
        assert 'status' in kwargs
        assert 'cur_it' in kwargs


# ============================================================================
# No duplicate handling
# ============================================================================

class TestNoDuplicateHandling:
    """Test that multiple tab instances both receive events."""

    def test_two_deploy_tabs_both_receive_events(self):
        """Test that creating two DeployTab instances both get events."""
        app_state1 = _make_mock_app_state()
        app_state2 = _make_mock_app_state()
        _make_deploy_tab(app_state1)
        _make_deploy_tab(app_state2)

        event_bus.emit(DEPLOY_PROGRESS, pct=0.0, message="Loading",
                       process="img_det", status="load")

        app_state1.progress_window.update_values.assert_called_once()
        app_state2.progress_window.update_values.assert_called_once()
