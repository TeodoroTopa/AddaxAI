"""Postprocessing UI tab — encapsulates postprocessing workflow UI components.

This module provides the PostprocessTab class which manages the postprocessing workflow UI,
including button controls and progress display. The class implements the PostprocessView
protocol to abstract the UI from orchestration logic.
"""

from typing import Any, Dict, Optional


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
