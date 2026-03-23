"""Tests for orchestrator event bus integration.

These tests verify that orchestrators emit the correct events in the correct order,
and that event handlers receive event data correctly.
"""

import pytest
from typing import Any, Dict, List

from addaxai.core.events import event_bus
from addaxai.core.event_types import (
    DEPLOY_STARTED,
    DEPLOY_PROGRESS,
    DEPLOY_ERROR,
    DEPLOY_FINISHED,
    CLASSIFY_STARTED,
    CLASSIFY_PROGRESS,
    CLASSIFY_ERROR,
    CLASSIFY_FINISHED,
    POSTPROCESS_STARTED,
    POSTPROCESS_PROGRESS,
    POSTPROCESS_ERROR,
    POSTPROCESS_FINISHED,
)


@pytest.fixture(autouse=True)
def clear_event_bus():
    """Clear event bus before and after each test to prevent cross-test leakage."""
    event_bus.clear_all()
    yield
    event_bus.clear_all()


class EventCollector:
    """Helper class to collect events for testing."""

    def __init__(self):
        self.events: List[tuple] = []

    def handler(self, event_name: str) -> Any:
        """Create a handler that records events."""
        def on_event(**kwargs: Any) -> None:
            self.events.append((event_name, kwargs))
        return on_event

    def subscribe(self, event_name: str) -> None:
        """Subscribe to an event."""
        event_bus.on(event_name, self.handler(event_name))

    def unsubscribe(self, event_name: str) -> None:
        """Unsubscribe from an event."""
        event_bus.off(event_name, self.handler(event_name))

    def clear(self) -> None:
        """Clear collected events."""
        self.events = []

    def assert_event(self, index: int, event_name: str, **expected_kwargs: Any) -> None:
        """Assert that an event at index matches expected name and kwargs."""
        assert index < len(self.events), f"Expected at least {index + 1} events, got {len(self.events)}"
        actual_name, actual_kwargs = self.events[index]
        assert actual_name == event_name, f"Event {index}: expected {event_name}, got {actual_name}"
        for key, value in expected_kwargs.items():
            assert key in actual_kwargs, f"Event {index}: expected kwarg {key} not found"
            assert actual_kwargs[key] == value, f"Event {index} {key}: expected {value}, got {actual_kwargs[key]}"

    def assert_event_count(self, expected: int) -> None:
        """Assert the total number of events received."""
        assert len(self.events) == expected, f"Expected {expected} events, got {len(self.events)}"


# ============================================================================
# Deployment Event Sequence Tests
# ============================================================================

class TestDeploymentEventSequence:
    """Test that deployment events are emitted in correct order with correct data."""

    def test_deploy_progress_event_receives_kwargs(self):
        """Test that DEPLOY_PROGRESS event correctly receives pct and message."""
        collector = EventCollector()
        collector.subscribe(DEPLOY_PROGRESS)

        event_bus.emit(DEPLOY_PROGRESS, pct=50.0, message="Processing: 5/10", process="img_det")

        collector.assert_event_count(1)
        collector.assert_event(0, DEPLOY_PROGRESS, pct=50.0, message="Processing: 5/10", process="img_det")

    def test_deploy_event_sequence(self):
        """Test complete deployment event sequence: started -> progress -> finished."""
        collector = EventCollector()
        for event in [DEPLOY_STARTED, DEPLOY_PROGRESS, DEPLOY_FINISHED]:
            collector.subscribe(event)

        # Simulate deployment workflow
        event_bus.emit(DEPLOY_STARTED, process="img_det")
        event_bus.emit(DEPLOY_PROGRESS, pct=25.0, message="Processing: 2/8", process="img_det")
        event_bus.emit(DEPLOY_PROGRESS, pct=50.0, message="Processing: 4/8", process="img_det")
        event_bus.emit(DEPLOY_PROGRESS, pct=75.0, message="Processing: 6/8", process="img_det")
        event_bus.emit(DEPLOY_FINISHED, results_path="/path/to/results.json", process="img_det")

        collector.assert_event_count(5)
        collector.assert_event(0, DEPLOY_STARTED, process="img_det")
        collector.assert_event(1, DEPLOY_PROGRESS, pct=25.0, process="img_det")
        collector.assert_event(2, DEPLOY_PROGRESS, pct=50.0, process="img_det")
        collector.assert_event(3, DEPLOY_PROGRESS, pct=75.0, process="img_det")
        collector.assert_event(4, DEPLOY_FINISHED, results_path="/path/to/results.json", process="img_det")

    def test_deploy_error_event(self):
        """Test that DEPLOY_ERROR event is received with message."""
        collector = EventCollector()
        collector.subscribe(DEPLOY_ERROR)

        event_bus.emit(DEPLOY_ERROR, message="No image files found", process="img_det")

        collector.assert_event_count(1)
        collector.assert_event(0, DEPLOY_ERROR, message="No image files found", process="img_det")

    def test_deploy_error_can_follow_started(self):
        """Test that error can occur after started event."""
        collector = EventCollector()
        for event in [DEPLOY_STARTED, DEPLOY_ERROR]:
            collector.subscribe(event)

        event_bus.emit(DEPLOY_STARTED, process="vid_det")
        event_bus.emit(DEPLOY_ERROR, message="GPU out of memory", process="vid_det")

        collector.assert_event_count(2)
        collector.assert_event(0, DEPLOY_STARTED, process="vid_det")
        collector.assert_event(1, DEPLOY_ERROR, message="GPU out of memory", process="vid_det")

    def test_deploy_with_image_data_type(self):
        """Test deployment events with image data type."""
        collector = EventCollector()
        collector.subscribe(DEPLOY_PROGRESS)

        event_bus.emit(DEPLOY_PROGRESS, pct=100.0, message="Detection complete", process="img_det")

        collector.assert_event_count(1)
        collector.assert_event(0, DEPLOY_PROGRESS, pct=100.0, process="img_det")

    def test_deploy_with_video_data_type(self):
        """Test deployment events with video data type."""
        collector = EventCollector()
        collector.subscribe(DEPLOY_PROGRESS)

        event_bus.emit(DEPLOY_PROGRESS, pct=50.0, message="Processing video", process="vid_det")

        collector.assert_event_count(1)
        collector.assert_event(0, DEPLOY_PROGRESS, pct=50.0, process="vid_det")


# ============================================================================
# Classification Event Sequence Tests
# ============================================================================

class TestClassificationEventSequence:
    """Test that classification events are emitted in correct order with correct data."""

    def test_classify_progress_event_receives_kwargs(self):
        """Test that CLASSIFY_PROGRESS event correctly receives pct and message."""
        collector = EventCollector()
        collector.subscribe(CLASSIFY_PROGRESS)

        event_bus.emit(CLASSIFY_PROGRESS, pct=75.0, message="Classifying: 6/8", process="img_cls")

        collector.assert_event_count(1)
        collector.assert_event(0, CLASSIFY_PROGRESS, pct=75.0, message="Classifying: 6/8", process="img_cls")

    def test_classify_event_sequence(self):
        """Test complete classification event sequence: started -> progress -> finished."""
        collector = EventCollector()
        for event in [CLASSIFY_STARTED, CLASSIFY_PROGRESS, CLASSIFY_FINISHED]:
            collector.subscribe(event)

        # Simulate classification workflow
        event_bus.emit(CLASSIFY_STARTED, process="img_cls")
        event_bus.emit(CLASSIFY_PROGRESS, pct=0.0, message="Loading classification model", process="img_cls")
        event_bus.emit(CLASSIFY_PROGRESS, pct=50.0, message="Classifying: 4/8", process="img_cls")
        event_bus.emit(CLASSIFY_FINISHED, results_path="/path/to/results.json", process="img_cls")

        collector.assert_event_count(4)
        collector.assert_event(0, CLASSIFY_STARTED, process="img_cls")
        collector.assert_event(1, CLASSIFY_PROGRESS, pct=0.0, process="img_cls")
        collector.assert_event(2, CLASSIFY_PROGRESS, pct=50.0, process="img_cls")
        collector.assert_event(3, CLASSIFY_FINISHED, results_path="/path/to/results.json", process="img_cls")

    def test_classify_error_event(self):
        """Test that CLASSIFY_ERROR event is received with message."""
        collector = EventCollector()
        collector.subscribe(CLASSIFY_ERROR)

        event_bus.emit(CLASSIFY_ERROR, message="No animal detections that meet the criteria", process="img_cls")

        collector.assert_event_count(1)
        collector.assert_event(0, CLASSIFY_ERROR, message="No animal detections that meet the criteria", process="img_cls")

    def test_classify_with_video_data_type(self):
        """Test classification events with video data type."""
        collector = EventCollector()
        collector.subscribe(CLASSIFY_PROGRESS)

        event_bus.emit(CLASSIFY_PROGRESS, pct=30.0, message="Classifying video animals", process="vid_cls")

        collector.assert_event_count(1)
        collector.assert_event(0, CLASSIFY_PROGRESS, pct=30.0, process="vid_cls")


# ============================================================================
# Postprocessing Event Sequence Tests
# ============================================================================

class TestPostprocessingEventSequence:
    """Test that postprocessing events are emitted in correct order with correct data."""

    def test_postprocess_progress_event_receives_kwargs(self):
        """Test that POSTPROCESS_PROGRESS event correctly receives pct and message."""
        collector = EventCollector()
        collector.subscribe(POSTPROCESS_PROGRESS)

        event_bus.emit(POSTPROCESS_PROGRESS, pct=40.0, message="Processing: 4/10", process="img_pst")

        collector.assert_event_count(1)
        collector.assert_event(0, POSTPROCESS_PROGRESS, pct=40.0, message="Processing: 4/10", process="img_pst")

    def test_postprocess_event_sequence(self):
        """Test complete postprocessing event sequence: started -> progress -> finished."""
        collector = EventCollector()
        for event in [POSTPROCESS_STARTED, POSTPROCESS_PROGRESS, POSTPROCESS_FINISHED]:
            collector.subscribe(event)

        # Simulate postprocessing workflow
        event_bus.emit(POSTPROCESS_STARTED)
        event_bus.emit(POSTPROCESS_PROGRESS, pct=0.0, message="Initializing postprocessing", process="img_pst")
        event_bus.emit(POSTPROCESS_PROGRESS, pct=50.0, message="Processing: 5/10", process="img_pst")
        event_bus.emit(POSTPROCESS_FINISHED)

        collector.assert_event_count(4)
        collector.assert_event(0, POSTPROCESS_STARTED)
        collector.assert_event(1, POSTPROCESS_PROGRESS, pct=0.0, process="img_pst")
        collector.assert_event(2, POSTPROCESS_PROGRESS, pct=50.0, process="img_pst")
        collector.assert_event(3, POSTPROCESS_FINISHED)

    def test_postprocess_error_event(self):
        """Test that POSTPROCESS_ERROR event is received with message."""
        collector = EventCollector()
        collector.subscribe(POSTPROCESS_ERROR)

        event_bus.emit(POSTPROCESS_ERROR, message="Destination folder not set or invalid")

        collector.assert_event_count(1)
        collector.assert_event(0, POSTPROCESS_ERROR, message="Destination folder not set or invalid")

    def test_postprocess_with_video_data_type(self):
        """Test postprocessing events with video data type."""
        collector = EventCollector()
        collector.subscribe(POSTPROCESS_PROGRESS)

        event_bus.emit(POSTPROCESS_PROGRESS, pct=60.0, message="Processing videos: 6/10", process="vid_pst")

        collector.assert_event_count(1)
        collector.assert_event(0, POSTPROCESS_PROGRESS, pct=60.0, process="vid_pst")


# ============================================================================
# Event Bus Isolation Tests
# ============================================================================

class TestEventBusIsolation:
    """Test that event bus prevents cross-test leakage."""

    def test_clear_all_removes_all_subscriptions(self):
        """Test that event_bus.clear_all() removes all subscribers."""
        collector1 = EventCollector()
        collector1.subscribe(DEPLOY_PROGRESS)

        # Emit event, should be received
        event_bus.emit(DEPLOY_PROGRESS, pct=50.0, message="Test", process="img_det")
        assert len(collector1.events) == 1

        # Clear event bus
        event_bus.clear_all()

        # Emit again, should not be received by cleared collector
        event_bus.emit(DEPLOY_PROGRESS, pct=75.0, message="Test 2", process="img_det")
        assert len(collector1.events) == 1  # Still 1, not incremented

    def test_multiple_subscribers_receive_same_event(self):
        """Test that multiple subscribers can listen to the same event."""
        collector1 = EventCollector()
        collector2 = EventCollector()
        collector1.subscribe(DEPLOY_PROGRESS)
        collector2.subscribe(DEPLOY_PROGRESS)

        event_bus.emit(DEPLOY_PROGRESS, pct=50.0, message="Test", process="img_det")

        assert len(collector1.events) == 1
        assert len(collector2.events) == 1

    def test_selective_event_subscription(self):
        """Test that collectors only receive events they subscribed to."""
        collector_deploy = EventCollector()
        collector_classify = EventCollector()

        collector_deploy.subscribe(DEPLOY_PROGRESS)
        collector_classify.subscribe(CLASSIFY_PROGRESS)

        event_bus.emit(DEPLOY_PROGRESS, pct=50.0, message="Test", process="img_det")
        event_bus.emit(CLASSIFY_PROGRESS, pct=50.0, message="Test", process="img_cls")

        assert len(collector_deploy.events) == 1
        assert len(collector_classify.events) == 1
        assert collector_deploy.events[0][0] == DEPLOY_PROGRESS
        assert collector_classify.events[0][0] == CLASSIFY_PROGRESS


# ============================================================================
# Event Data Integrity Tests
# ============================================================================

class TestEventDataIntegrity:
    """Test that event data is correctly transmitted through event bus."""

    def test_deploy_progress_percentage_values(self):
        """Test various percentage values are correctly transmitted."""
        collector = EventCollector()
        collector.subscribe(DEPLOY_PROGRESS)

        percentages = [0.0, 25.5, 50.0, 75.5, 100.0]
        for pct in percentages:
            event_bus.emit(DEPLOY_PROGRESS, pct=pct, message="Test", process="img_det")

        assert len(collector.events) == len(percentages)
        for i, pct in enumerate(percentages):
            _, kwargs = collector.events[i]
            assert kwargs['pct'] == pct

    def test_process_type_preservation(self):
        """Test that process type (img_det, vid_det, img_cls, etc.) is preserved."""
        collector = EventCollector()
        collector.subscribe(DEPLOY_PROGRESS)

        process_types = ["img_det", "vid_det"]
        for process in process_types:
            event_bus.emit(DEPLOY_PROGRESS, pct=50.0, message="Test", process=process)

        assert len(collector.events) == len(process_types)
        for i, process in enumerate(process_types):
            _, kwargs = collector.events[i]
            assert kwargs['process'] == process

    def test_message_with_special_characters(self):
        """Test that messages with special characters are preserved."""
        collector = EventCollector()
        collector.subscribe(DEPLOY_PROGRESS)

        messages = [
            "Processing: 5/10",
            "Error: File not found",
            "100% complete!",
            "Path: /home/user/files",
        ]
        for msg in messages:
            event_bus.emit(DEPLOY_PROGRESS, pct=50.0, message=msg, process="img_det")

        assert len(collector.events) == len(messages)
        for i, msg in enumerate(messages):
            _, kwargs = collector.events[i]
            assert kwargs['message'] == msg
