"""Tests for view protocols."""

import pytest
from typing import Any, Callable, Dict, List, Optional

from addaxai.ui.protocols import (
    DeployView,
    PostprocessView,
    HITLView,
    ResultsView,
)


class MockDeployView:
    """Mock implementation of DeployView protocol."""

    def show_progress(self, pct: float, message: str) -> None:
        pass

    def show_error(self, message: str) -> None:
        pass

    def show_completion(self, results_path: str) -> None:
        pass

    def set_model_list(self, models: List[str]) -> None:
        pass

    def on_cancel(self, callback: Callable[[], None]) -> None:
        pass

    def reset(self) -> None:
        pass


class MockPostprocessView:
    """Mock implementation of PostprocessView protocol."""

    def show_progress(self, pct: float, message: str) -> None:
        pass

    def show_error(self, message: str) -> None:
        pass

    def show_completion(self, summary: Dict[str, Any]) -> None:
        pass

    def reset(self) -> None:
        pass


class MockHITLView:
    """Mock implementation of HITLView protocol."""

    def load_annotations(self, data: Dict[str, Any]) -> None:
        pass

    def show_image(self, path: str, boxes: List[Dict[str, Any]]) -> None:
        pass

    def on_save(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        pass

    def reset(self) -> None:
        pass


class MockResultsView:
    """Mock implementation of ResultsView protocol."""

    def display(self, recognition_json: Dict[str, Any]) -> None:
        pass

    def set_filters(
        self,
        species: Optional[List[str]],
        confidence: Optional[float],
    ) -> None:
        pass

    def reset(self) -> None:
        pass


class TestDeployViewProtocol:
    """Tests for DeployView protocol."""

    def test_deploy_view_mock_satisfies_protocol(self) -> None:
        """Mock DeployView should satisfy isinstance check."""
        view = MockDeployView()
        assert isinstance(view, DeployView)

    def test_deploy_view_incomplete_implementation_fails(self) -> None:
        """Incomplete DeployView implementation should fail isinstance check."""

        class IncompleteDeployView:
            def show_progress(self, pct: float, message: str) -> None:
                pass

            def show_error(self, message: str) -> None:
                pass

        view = IncompleteDeployView()
        assert not isinstance(view, DeployView)

    def test_deploy_view_protocol_has_all_required_methods(self) -> None:
        """DeployView protocol should have all required methods."""
        required_methods = {
            "show_progress",
            "show_error",
            "show_completion",
            "set_model_list",
            "on_cancel",
            "reset",
        }
        protocol_methods = {
            name
            for name in dir(DeployView)
            if not name.startswith("_") and callable(getattr(DeployView, name))
        }
        assert required_methods.issubset(protocol_methods)


class TestPostprocessViewProtocol:
    """Tests for PostprocessView protocol."""

    def test_postprocess_view_mock_satisfies_protocol(self) -> None:
        """Mock PostprocessView should satisfy isinstance check."""
        view = MockPostprocessView()
        assert isinstance(view, PostprocessView)

    def test_postprocess_view_incomplete_implementation_fails(self) -> None:
        """Incomplete PostprocessView implementation should fail isinstance check."""

        class IncompletePostprocessView:
            def show_progress(self, pct: float, message: str) -> None:
                pass

        view = IncompletePostprocessView()
        assert not isinstance(view, PostprocessView)


class TestHITLViewProtocol:
    """Tests for HITLView protocol."""

    def test_hitl_view_mock_satisfies_protocol(self) -> None:
        """Mock HITLView should satisfy isinstance check."""
        view = MockHITLView()
        assert isinstance(view, HITLView)

    def test_hitl_view_incomplete_implementation_fails(self) -> None:
        """Incomplete HITLView implementation should fail isinstance check."""

        class IncompleteHITLView:
            def load_annotations(self, data: Dict[str, Any]) -> None:
                pass

        view = IncompleteHITLView()
        assert not isinstance(view, HITLView)


class TestResultsViewProtocol:
    """Tests for ResultsView protocol."""

    def test_results_view_mock_satisfies_protocol(self) -> None:
        """Mock ResultsView should satisfy isinstance check."""
        view = MockResultsView()
        assert isinstance(view, ResultsView)

    def test_results_view_incomplete_implementation_fails(self) -> None:
        """Incomplete ResultsView implementation should fail isinstance check."""

        class IncompleteResultsView:
            def display(self, recognition_json: Dict[str, Any]) -> None:
                pass

        view = IncompleteResultsView()
        assert not isinstance(view, ResultsView)


class TestMultipleViewImplementations:
    """Tests for multiple implementations of the same protocol."""

    def test_different_deploy_views_satisfy_protocol(self) -> None:
        """Different implementations should all satisfy the protocol."""

        class AlternativeDeployView:
            def show_progress(self, pct: float, message: str) -> None:
                print(f"{pct}%: {message}")

            def show_error(self, message: str) -> None:
                print(f"ERROR: {message}")

            def show_completion(self, results_path: str) -> None:
                print(f"Done: {results_path}")

            def set_model_list(self, models: List[str]) -> None:
                self.models = models

            def on_cancel(self, callback: Callable[[], None]) -> None:
                self.cancel_cb = callback

            def reset(self) -> None:
                self.models = []

        view1 = MockDeployView()
        view2 = AlternativeDeployView()

        assert isinstance(view1, DeployView)
        assert isinstance(view2, DeployView)
