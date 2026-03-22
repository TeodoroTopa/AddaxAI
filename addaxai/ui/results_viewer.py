"""Results viewer UI — displays detection and classification results.

This module provides the ResultsViewer class which manages the results display interface,
including filtering and visualization. The class implements the ResultsView protocol
to abstract the UI from the data logic.
"""

from typing import Any, Dict, List, Optional


class ResultsViewer:
    """Manages the results display UI.

    Implements the ResultsView protocol to provide a clean interface between
    the results data logic (in app.py) and the UI widgets.
    """

    def __init__(
        self,
        parent_frame: Any,
        app_state: Optional[Any] = None,
    ) -> None:
        """Initialize the results viewer UI.

        Args:
            parent_frame: The parent tkinter frame to add widgets to.
            app_state: Reference to AppState instance for state management.
        """
        self.parent_frame = parent_frame
        self.app_state = app_state
        self._table_ref: Optional[Any] = None
        self._current_results: Optional[Dict[str, Any]] = None
        self._current_filters: Dict[str, Any] = {}

    def display(self, recognition_json: Dict[str, Any]) -> None:
        """Display recognition results from JSON.

        Args:
            recognition_json: Dictionary with detection and classification results.
        """
        self._current_results = recognition_json
        # Render results to table/list widget
        self._render_results()

    def set_filters(
        self,
        species: Optional[List[str]],
        confidence: Optional[float],
    ) -> None:
        """Apply species and confidence filters to results.

        Args:
            species: List of species names to filter by (None for all).
            confidence: Minimum confidence threshold (None for no filtering).
        """
        self._current_filters = {
            "species": species,
            "confidence": confidence,
        }
        self._render_results()

    def reset(self) -> None:
        """Reset the UI to initial state."""
        self._current_results = None
        self._current_filters = {}
        if self._table_ref:
            # Clear table
            pass

    def _render_results(self) -> None:
        """Render current results with applied filters to the UI."""
        if not self._current_results or not self._table_ref:
            return
        # Apply filters and render to table
        pass
