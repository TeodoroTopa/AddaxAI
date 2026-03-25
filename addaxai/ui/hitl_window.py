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
        self.lbl_hitl_main: Optional[Any] = None
        self.btn_hitl_main: Optional[Any] = None

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

    def build_widgets(
        self,
        start_hitl_callback: Callable[[], None],
        t_func: Callable[[str], str],
    ) -> None:
        """Construct and grid HITL label and button widgets.

        Args:
            start_hitl_callback: Callback function for the start/continue button.
            t_func: Translation function t(key) for i18n.
        """
        try:
            from tkinter import Button, Label
        except ImportError:
            # For unit tests that don't have tkinter
            return

        # human-in-the-loop
        row_hitl_main = 0
        self.lbl_hitl_main = Label(
            master=self.parent_frame,
            text=t_func('lbl_hitl_main'),
            width=1,
            anchor="w"
        )
        self.lbl_hitl_main.grid(row=row_hitl_main, sticky='nesw', pady=2)

        self.btn_hitl_main = Button(
            master=self.parent_frame,
            text=t_func('start'),
            width=1,
            command=start_hitl_callback
        )
        self.btn_hitl_main.grid(row=row_hitl_main, column=1, sticky='nesw', padx=5)
