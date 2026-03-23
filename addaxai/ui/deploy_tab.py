"""Deployment UI tab — encapsulates deployment workflow UI components.

This module provides the DeployTab class which manages the deployment workflow UI,
including the deploy button, progress display, and related event handlers.
The class implements the DeployView protocol to abstract the UI from orchestration logic.
"""

import logging
from typing import Any, Callable, List, Optional

try:
    from tkinter import Button
except ImportError:
    Button = None  # type: ignore

from addaxai.core.events import event_bus
from addaxai.core.event_types import (
    DEPLOY_PROGRESS,
    DEPLOY_FINISHED,
    DEPLOY_ERROR,
    CLASSIFY_PROGRESS,
    CLASSIFY_FINISHED,
    CLASSIFY_ERROR,
)
from addaxai.i18n import t

logger = logging.getLogger(__name__)


class DeployTab:
    """Manages the deployment workflow UI.

    Implements the DeployView protocol to provide a clean interface between
    the deployment orchestration logic (in app.py) and the UI widgets.

    Event handlers receive progress data from orchestrators via the event bus
    and forward it transparently to ProgressWindow.update_values(). The events
    carry the exact same kwargs that update_values() expects (process, status,
    cur_it, tot_it, time_ela, time_rem, speed, hware, cancel_func, etc.)
    so that no information is lost in translation.
    """

    def __init__(
        self,
        parent_frame: Any,  # tk.Frame
        start_deploy_callback: Callable[[], None],
        app_state: Optional[Any] = None,
    ) -> None:
        """Initialize the deploy tab UI.

        Args:
            parent_frame: The parent tkinter frame to add widgets to.
            start_deploy_callback: Callback function for the start deploy button.
            app_state: Reference to AppState instance for state management.
        """
        self.parent_frame = parent_frame
        self.start_deploy_callback = start_deploy_callback
        self.app_state = app_state
        self._button_ref: Optional[Any] = None
        self._progress_label_ref: Optional[Any] = None

        # Subscribe to deployment and classification events
        event_bus.on(DEPLOY_PROGRESS, self._on_deploy_progress)
        event_bus.on(DEPLOY_ERROR, self._on_deploy_error)
        event_bus.on(DEPLOY_FINISHED, self._on_deploy_finished)
        event_bus.on(CLASSIFY_PROGRESS, self._on_classify_progress)
        event_bus.on(CLASSIFY_ERROR, self._on_classify_error)
        event_bus.on(CLASSIFY_FINISHED, self._on_classify_finished)

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
            # (e.g. full-image classifier has no img_det frame)
            logger.debug("Could not forward to progress_window", exc_info=True)

    def _on_deploy_progress(self, **kwargs: Any) -> None:
        """Handle deployment progress event.

        Forwards all update_values()-compatible kwargs to ProgressWindow.
        """
        pct = kwargs.get('pct', 0.0)
        message = kwargs.get('message', '')
        self.show_progress(pct, message)
        self._forward_to_progress_window(**kwargs)

    def _on_deploy_error(self, **kwargs: Any) -> None:
        """Handle deployment error event."""
        self.show_error(kwargs.get('message', ''))

    def _on_deploy_finished(self, **kwargs: Any) -> None:
        """Handle deployment finished event."""
        self.show_completion(kwargs.get('results_path', ''))

    def _on_classify_progress(self, **kwargs: Any) -> None:
        """Handle classification progress event.

        Forwards all update_values()-compatible kwargs to ProgressWindow.
        """
        pct = kwargs.get('pct', 0.0)
        message = kwargs.get('message', '')
        self.show_progress(pct, message)
        self._forward_to_progress_window(**kwargs)

    def _on_classify_error(self, **kwargs: Any) -> None:
        """Handle classification error event."""
        self.show_error(kwargs.get('message', ''))

    def _on_classify_finished(self, **kwargs: Any) -> None:
        """Handle classification finished event."""
        self.show_completion(kwargs.get('results_path', ''))

    def create_button(self) -> Any:
        """Create and return the start deploy button.

        Returns:
            The created button widget.
        """
        if Button is None:
            raise ImportError("tkinter.Button not available")
        row = 12
        btn = Button(
            self.parent_frame,
            text=t("btn_start_deploy"),
            command=self.start_deploy_callback,
        )
        btn.grid(row=row, column=0, columnspan=2, sticky="ew")
        self._button_ref = btn
        return btn

    def set_button_ref(self, button_ref: Any) -> None:
        """Set reference to the deploy button (for integration with existing code).

        Args:
            button_ref: Reference to the button widget created externally.
        """
        self._button_ref = button_ref

    def get_button_ref(self) -> Optional[Any]:
        """Get reference to the deploy button.

        Returns:
            The button widget reference.
        """
        return self._button_ref

    def show_progress(self, pct: float, message: str) -> None:
        """Display deployment progress.

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
        # Error display is typically handled by messageboxes in the orchestration logic
        # This method is here for protocol compliance
        pass

    def show_completion(self, results_path: str) -> None:
        """Display completion message with results path.

        Args:
            results_path: Path to the results JSON file.
        """
        # Completion is typically handled in orchestration logic
        # This method is here for protocol compliance
        pass

    def set_model_list(self, models: List[str]) -> None:
        """Update the list of available detection models.

        Args:
            models: List of available model names.
        """
        # Model list updates are typically handled elsewhere
        # This method is here for protocol compliance
        pass

    def on_cancel(self, callback: Callable[[], None]) -> None:
        """Register a callback for cancel button press.

        Args:
            callback: Function to call when cancel is pressed.
        """
        # Cancel handling is integrated with the button command
        # This method is here for protocol compliance
        pass

    def reset(self) -> None:
        """Reset the UI to initial state."""
        if self._progress_label_ref:
            self._progress_label_ref.configure(text="")
        if self._button_ref:
            self._button_ref.configure(state="normal")
