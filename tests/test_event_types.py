"""Tests for event type constants (addaxai/core/event_types.py)."""

import addaxai.core.event_types as event_types


class TestEventTypeConstants:
    """Tests for event type name constants."""

    def test_all_constants_are_strings(self) -> None:
        """All event constants should be strings."""
        constants = [
            getattr(event_types, name)
            for name in dir(event_types)
            if name.isupper() and not name.startswith("_")
        ]
        assert len(constants) > 0
        assert all(isinstance(c, str) for c in constants)

    def test_all_event_names_are_unique(self) -> None:
        """No two event names should be identical."""
        constants = [
            getattr(event_types, name)
            for name in dir(event_types)
            if name.isupper() and not name.startswith("_")
        ]
        assert len(constants) == len(set(constants))

    def test_event_names_follow_naming_convention(self) -> None:
        """All event names should be lowercase with dots."""
        constants = [
            getattr(event_types, name)
            for name in dir(event_types)
            if name.isupper() and not name.startswith("_")
        ]
        for name in constants:
            assert name.islower() or "." in name
            assert name.count(".") >= 1  # At least one dot for category

    def test_deployment_events_exist(self) -> None:
        """Deployment event constants should exist."""
        assert hasattr(event_types, "DEPLOY_STARTED")
        assert hasattr(event_types, "DEPLOY_PROGRESS")
        assert hasattr(event_types, "DEPLOY_IMAGE_COMPLETE")
        assert hasattr(event_types, "DEPLOY_ERROR")
        assert hasattr(event_types, "DEPLOY_CANCELLED")
        assert hasattr(event_types, "DEPLOY_FINISHED")

    def test_classification_events_exist(self) -> None:
        """Classification event constants should exist."""
        assert hasattr(event_types, "CLASSIFY_STARTED")
        assert hasattr(event_types, "CLASSIFY_PROGRESS")
        assert hasattr(event_types, "CLASSIFY_ERROR")
        assert hasattr(event_types, "CLASSIFY_FINISHED")

    def test_postprocessing_events_exist(self) -> None:
        """Postprocessing event constants should exist."""
        assert hasattr(event_types, "POSTPROCESS_STARTED")
        assert hasattr(event_types, "POSTPROCESS_PROGRESS")
        assert hasattr(event_types, "POSTPROCESS_ERROR")
        assert hasattr(event_types, "POSTPROCESS_FINISHED")

    def test_model_download_events_exist(self) -> None:
        """Model management event constants should exist."""
        assert hasattr(event_types, "MODEL_DOWNLOAD_STARTED")
        assert hasattr(event_types, "MODEL_DOWNLOAD_PROGRESS")
        assert hasattr(event_types, "MODEL_DOWNLOAD_FINISHED")
        assert hasattr(event_types, "MODEL_DOWNLOAD_ERROR")

    def test_deploy_events_have_deploy_prefix(self) -> None:
        """Deployment events should use 'deploy.' prefix."""
        assert event_types.DEPLOY_STARTED.startswith("deploy.")
        assert event_types.DEPLOY_PROGRESS.startswith("deploy.")
        assert event_types.DEPLOY_FINISHED.startswith("deploy.")

    def test_classify_events_have_classify_prefix(self) -> None:
        """Classification events should use 'classify.' prefix."""
        assert event_types.CLASSIFY_STARTED.startswith("classify.")
        assert event_types.CLASSIFY_PROGRESS.startswith("classify.")
        assert event_types.CLASSIFY_FINISHED.startswith("classify.")

    def test_postprocess_events_have_postprocess_prefix(self) -> None:
        """Postprocessing events should use 'postprocess.' prefix."""
        assert event_types.POSTPROCESS_STARTED.startswith("postprocess.")
        assert event_types.POSTPROCESS_PROGRESS.startswith("postprocess.")
        assert event_types.POSTPROCESS_FINISHED.startswith("postprocess.")

    def test_model_events_have_model_prefix(self) -> None:
        """Model management events should use 'model.' prefix."""
        assert event_types.MODEL_DOWNLOAD_STARTED.startswith("model.")
        assert event_types.MODEL_DOWNLOAD_PROGRESS.startswith("model.")
        assert event_types.MODEL_DOWNLOAD_FINISHED.startswith("model.")
        assert event_types.MODEL_DOWNLOAD_ERROR.startswith("model.")

    def test_constant_names_match_values(self) -> None:
        """Constant names should match the event values (converted to snake_case)."""
        # E.g., DEPLOY_STARTED should be "deploy.started"
        assert event_types.DEPLOY_STARTED == "deploy.started"
        assert event_types.DEPLOY_PROGRESS == "deploy.progress"
        assert event_types.DEPLOY_FINISHED == "deploy.finished"
        assert event_types.CLASSIFY_STARTED == "classify.started"
        assert event_types.POSTPROCESS_STARTED == "postprocess.started"
        assert event_types.MODEL_DOWNLOAD_STARTED == "model.download_started"
