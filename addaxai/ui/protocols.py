"""View protocols — contracts between orchestration logic and the UI.

Each protocol defines what a UI component must be able to do, without
specifying how (no tkinter/Qt/web types in signatures). Orchestration
code and the event bus talk to these protocols. Concrete implementations
live in the ui/ subpackages.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class DeployView(Protocol):
    """UI contract for the deployment workflow."""

    def show_progress(self, pct: float, message: str) -> None:
        """Display deployment progress."""
        ...

    def show_error(self, message: str) -> None:
        """Display an error message."""
        ...

    def show_completion(self, results_path: str) -> None:
        """Display completion message with results path."""
        ...

    def set_model_list(self, models: List[str]) -> None:
        """Update the list of available detection models."""
        ...

    def on_cancel(self, callback: Callable[[], None]) -> None:
        """Register a callback for cancel button press."""
        ...

    def reset(self) -> None:
        """Reset the UI to initial state."""
        ...


@runtime_checkable
class PostprocessView(Protocol):
    """UI contract for postprocessing."""

    def show_progress(self, pct: float, message: str) -> None:
        """Display postprocessing progress."""
        ...

    def show_error(self, message: str) -> None:
        """Display an error message."""
        ...

    def show_completion(self, summary: Dict[str, Any]) -> None:
        """Display completion with summary stats."""
        ...

    def reset(self) -> None:
        """Reset the UI to initial state."""
        ...


@runtime_checkable
class HITLView(Protocol):
    """UI contract for human-in-the-loop annotation."""

    def load_annotations(self, data: Dict[str, Any]) -> None:
        """Load annotation data from recognition JSON."""
        ...

    def show_image(self, path: str, boxes: List[Dict[str, Any]]) -> None:
        """Display an image with detection boxes."""
        ...

    def on_save(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for save button press."""
        ...

    def reset(self) -> None:
        """Reset the UI to initial state."""
        ...


@runtime_checkable
class ResultsView(Protocol):
    """UI contract for results display."""

    def display(self, recognition_json: Dict[str, Any]) -> None:
        """Display recognition results from JSON."""
        ...

    def set_filters(
        self,
        species: Optional[List[str]],
        confidence: Optional[float],
    ) -> None:
        """Apply species and confidence filters to results."""
        ...

    def reset(self) -> None:
        """Reset the UI to initial state."""
        ...
