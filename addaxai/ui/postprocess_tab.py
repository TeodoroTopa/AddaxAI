"""Postprocessing UI tab — encapsulates postprocessing workflow UI components.

This module provides the PostprocessTab class which manages the postprocessing workflow UI,
including button controls and progress display. The class implements the PostprocessView
protocol to abstract the UI from orchestration logic.
"""

import logging
from typing import Any, Dict, Optional

from addaxai.core.events import event_bus
from addaxai.core.event_types import (
    POSTPROCESS_PROGRESS,
    POSTPROCESS_FINISHED,
    POSTPROCESS_ERROR,
)

logger = logging.getLogger(__name__)


class PostprocessTab:
    """Manages the postprocessing workflow UI.

    Implements the PostprocessView protocol to provide a clean interface between
    the postprocessing orchestration logic (in app.py) and the UI widgets.

    Event handlers receive progress data from orchestrators via the event bus
    and forward it transparently to ProgressWindow.update_values(). The events
    carry the exact same kwargs that update_values() expects (process, status,
    cur_it, tot_it, time_ela, time_rem, cancel_func) so that no information
    is lost in translation.
    """

    def __init__(
        self,
        parent_frame: Any,
        app_state: Optional[Any] = None,
    ) -> None:
        """Initialize the postprocess tab UI.

        Args:
            parent_frame: The parent tkinter frame to add widgets to.
            app_state: Reference to AppState instance for state management.
        """
        self.parent_frame = parent_frame
        self.app_state = app_state
        self._progress_label_ref: Optional[Any] = None

        # Subscribe to postprocessing events
        event_bus.on(POSTPROCESS_PROGRESS, self._on_postprocess_progress)
        event_bus.on(POSTPROCESS_ERROR, self._on_postprocess_error)
        event_bus.on(POSTPROCESS_FINISHED, self._on_postprocess_finished)

    def _forward_to_progress_window(self, **kwargs: Any) -> None:
        """Forward event kwargs to progress_window.update_values().

        Extracts the kwargs that update_values() accepts and passes them
        through unchanged. Any extra kwargs (like 'pct', 'message') are
        ignored — they're for other consumers.
        """
        if not self.app_state or not hasattr(self.app_state, 'progress_window'):
            return
        pw = self.app_state.progress_window
        if pw is None:
            return

        # Build the kwargs dict with only keys update_values() accepts
        uv_kwargs = {}  # type: dict
        for key in ('process', 'status', 'cur_it', 'tot_it', 'time_ela',
                     'time_rem', 'speed', 'hware', 'cancel_func',
                     'extracting_frames_txt', 'frame_video_choice'):
            if key in kwargs:
                uv_kwargs[key] = kwargs[key]

        if 'process' not in uv_kwargs or 'status' not in uv_kwargs:
            return  # Can't call update_values without these two

        try:
            pw.update_values(**uv_kwargs)
        except (AttributeError, TypeError):
            # ProgressWindow may not have the widgets for this process type
            logger.debug("Could not forward to progress_window", exc_info=True)

    def _on_postprocess_progress(self, **kwargs: Any) -> None:
        """Handle postprocessing progress event.

        Forwards all update_values()-compatible kwargs to ProgressWindow.
        """
        pct = kwargs.get('pct', 0.0)
        message = kwargs.get('message', '')
        self.show_progress(pct, message)
        self._forward_to_progress_window(**kwargs)

    def _on_postprocess_error(self, **kwargs: Any) -> None:
        """Handle postprocessing error event."""
        self.show_error(kwargs.get('message', ''))

    def _on_postprocess_finished(self, **kwargs: Any) -> None:
        """Handle postprocessing finished event."""
        self.show_completion({})

    def show_progress(self, pct: float, message: str) -> None:
        """Display postprocessing progress.

        Args:
            pct: Progress percentage (0-100).
            message: Progress message to display.
        """
        if self._progress_label_ref:
            self._progress_label_ref.configure(text=f"{pct}%: {message}")

    def show_error(self, message: str) -> None:
        """Display an error message.

        Args:
            message: Error message text.
        """
        # Error display is typically handled by messageboxes in orchestration logic
        pass

    def show_completion(self, summary: Dict[str, Any]) -> None:
        """Display completion with summary stats.

        Args:
            summary: Dictionary with summary statistics.
        """
        # Completion display is typically handled in orchestration logic
        pass

    def reset(self) -> None:
        """Reset the UI to initial state."""
        if self._progress_label_ref:
            self._progress_label_ref.configure(text="")
