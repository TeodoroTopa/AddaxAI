"""Postprocessing UI tab — encapsulates postprocessing workflow UI components.

This module provides the PostprocessTab class which manages the postprocessing workflow UI,
including button controls and progress display. The class implements the PostprocessView
protocol to abstract the UI from orchestration logic.
"""

from typing import Any, Dict, Optional

from addaxai.core.events import event_bus
from addaxai.core.event_types import (
    POSTPROCESS_PROGRESS,
    POSTPROCESS_FINISHED,
    POSTPROCESS_ERROR,
)


class PostprocessTab:
    """Manages the postprocessing workflow UI.

    Implements the PostprocessView protocol to provide a clean interface between
    the postprocessing orchestration logic (in app.py) and the UI widgets.
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

    def _on_postprocess_progress(self, pct: float, message: str, process: Optional[str] = None, **kwargs: Any) -> None:
        """Handle postprocessing progress event.

        Translates event data to progress_window.update_values() calls.
        """
        self.show_progress(pct, message)

        # Wire to ProgressWindow if available
        if self.app_state and hasattr(self.app_state, 'progress_window') and process:
            try:
                self.app_state.progress_window.update_values(
                    process=process,
                    status="running" if pct < 100 else "done",
                    cur_it=int(pct),
                    tot_it=100,
                )
            except (AttributeError, TypeError):
                pass

    def _on_postprocess_error(self, message: str, **kwargs: Any) -> None:
        """Handle postprocessing error event."""
        self.show_error(message)

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
