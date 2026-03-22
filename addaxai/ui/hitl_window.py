"""Human-in-the-loop annotation UI — manages HITL workflow UI components.

This module provides the HITLWindow class which manages the human-in-the-loop
annotation interface. The class implements the HITLView protocol to abstract
the UI from the annotation logic.
"""

from typing import Any, Callable, Dict, List, Optional


class HITLWindow:
    """Manages the human-in-the-loop annotation UI.

    Implements the HITLView protocol to provide a clean interface between
    the HITL annotation logic (in app.py) and the UI widgets.
    """

    def __init__(
        self,
        parent_frame: Any,
        app_state: Optional[Any] = None,
    ) -> None:
        """Initialize the HITL window UI.

        Args:
            parent_frame: The parent tkinter frame (or window) to add widgets to.
            app_state: Reference to AppState instance for state management.
        """
        self.parent_frame = parent_frame
        self.app_state = app_state
        self._canvas_ref: Optional[Any] = None
        self._current_image_path: Optional[str] = None

    def load_annotations(self, data: Dict[str, Any]) -> None:
        """Load annotation data from recognition JSON.

        Args:
            data: Dictionary with recognition results and metadata.
        """
        # Load data into the annotation interface
        pass

    def show_image(self, path: str, boxes: List[Dict[str, Any]]) -> None:
        """Display an image with detection boxes.

        Args:
            path: Path to the image file.
            boxes: List of bounding box dictionaries with coordinates and metadata.
        """
        self._current_image_path = path
        if self._canvas_ref:
            # Render image and boxes to canvas
            pass

    def on_save(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for save button press.

        Args:
            callback: Function to call with updated annotation data when saved.
        """
        # Wire callback to save button
        pass

    def reset(self) -> None:
        """Reset the UI to initial state."""
        self._current_image_path = None
        if self._canvas_ref:
            self._canvas_ref.delete("all")
