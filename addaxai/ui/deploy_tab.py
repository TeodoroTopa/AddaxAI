"""Deployment UI tab — encapsulates deployment workflow UI components.

This module provides the DeployTab class which manages the deployment workflow UI,
including the deploy button, progress display, and related event handlers.
The class implements the DeployView protocol to abstract the UI from orchestration logic.
"""

from typing import Any, Callable, List, Optional

from addaxai.i18n import t


class DeployTab:
    """Manages the deployment workflow UI.

    Implements the DeployView protocol to provide a clean interface between
    the deployment orchestration logic (in app.py) and the UI widgets.
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

    def _create_button(self, button_class: Any) -> Any:
        """Create the start deploy button.

        Args:
            button_class: The Button class to use (tk.Button or customtkinter.CTkButton).

        Returns:
            The created button widget.
        """
        row = 12
        btn = button_class(
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
