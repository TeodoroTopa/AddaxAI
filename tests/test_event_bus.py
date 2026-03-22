"""Tests for the lightweight event bus (addaxai/core/events.py)."""

from typing import Any, List

import pytest

from addaxai.core.events import EventBus, event_bus


class TestEventBus:
    """Tests for EventBus class."""

    def test_on_and_emit_calls_callback(self) -> None:
        """Registering and emitting should call the callback."""
        bus = EventBus()
        called: List[Any] = []

        def callback(value: int) -> None:
            called.append(value)

        bus.on("test.event", callback)
        bus.emit("test.event", value=42)

        assert called == [42]

    def test_emit_with_multiple_kwargs(self) -> None:
        """Callback should receive all keyword arguments."""
        bus = EventBus()
        results: List[tuple] = []

        def callback(a: int, b: str, c: float) -> None:
            results.append((a, b, c))

        bus.on("test", callback)
        bus.emit("test", a=1, b="hello", c=3.14)

        assert results == [(1, "hello", 3.14)]

    def test_off_removes_callback(self) -> None:
        """Unsubscribing should remove callback."""
        bus = EventBus()
        called: List[int] = []

        def callback(x: int) -> None:
            called.append(x)

        bus.on("test", callback)
        bus.emit("test", x=1)
        assert called == [1]

        bus.off("test", callback)
        bus.emit("test", x=2)
        assert called == [1]  # Not called again

    def test_clear_removes_all_listeners(self) -> None:
        """Clearing an event should remove all listeners."""
        bus = EventBus()
        calls: List[int] = []

        def callback1(x: int) -> None:
            calls.append(x + 1)

        def callback2(x: int) -> None:
            calls.append(x + 2)

        bus.on("test", callback1)
        bus.on("test", callback2)
        bus.emit("test", x=0)
        assert calls == [1, 2]

        bus.clear("test")
        bus.emit("test", x=0)
        assert calls == [1, 2]  # No new calls

    def test_clear_all_removes_all_events(self) -> None:
        """Clearing all should remove all listeners from all events."""
        bus = EventBus()
        calls: List[str] = []

        def cb1() -> None:
            calls.append("event1")

        def cb2() -> None:
            calls.append("event2")

        bus.on("event1", cb1)
        bus.on("event2", cb2)
        bus.clear_all()

        bus.emit("event1")
        bus.emit("event2")
        assert calls == []

    def test_emit_with_no_listeners(self) -> None:
        """Emitting an event with no listeners should not raise."""
        bus = EventBus()
        bus.emit("nonexistent.event", data="test")  # Should not raise

    def test_multiple_callbacks_for_same_event(self) -> None:
        """Multiple callbacks for one event should all be called."""
        bus = EventBus()
        results: List[str] = []

        def cb1() -> None:
            results.append("first")

        def cb2() -> None:
            results.append("second")

        def cb3() -> None:
            results.append("third")

        bus.on("test", cb1)
        bus.on("test", cb2)
        bus.on("test", cb3)
        bus.emit("test")

        assert results == ["first", "second", "third"]

    def test_callback_exception_does_not_stop_other_callbacks(self) -> None:
        """If one callback raises, others should still be called."""
        bus = EventBus()
        results: List[str] = []

        def cb1() -> None:
            results.append("first")
            raise ValueError("intentional error")

        def cb2() -> None:
            results.append("second")

        bus.on("test", cb1)
        bus.on("test", cb2)
        bus.emit("test")

        # Both callbacks should be called despite exception
        assert results == ["first", "second"]

    def test_duplicate_on_not_registered_twice(self) -> None:
        """Registering the same callback twice should only register once."""
        bus = EventBus()
        calls: List[int] = []

        def callback(x: int) -> None:
            calls.append(x)

        bus.on("test", callback)
        bus.on("test", callback)  # Register again
        bus.emit("test", x=1)

        assert calls == [1]  # Only called once, not twice

    def test_module_level_singleton(self) -> None:
        """Module-level event_bus should be an EventBus instance."""
        assert isinstance(event_bus, EventBus)

    def test_module_singleton_persists_across_imports(self) -> None:
        """The singleton should be the same instance."""
        from addaxai.core.events import event_bus as eb2

        assert event_bus is eb2

    def test_off_nonexistent_callback(self) -> None:
        """Removing a callback that wasn't registered should not raise."""
        bus = EventBus()

        def callback() -> None:
            pass

        bus.off("test", callback)  # Should not raise

    def test_clear_nonexistent_event(self) -> None:
        """Clearing an event that has no listeners should not raise."""
        bus = EventBus()
        bus.clear("nonexistent")  # Should not raise

    def test_different_events_independent(self) -> None:
        """Different events should be independent."""
        bus = EventBus()
        results: List[str] = []

        def cb1() -> None:
            results.append("event1")

        def cb2() -> None:
            results.append("event2")

        bus.on("event1", cb1)
        bus.on("event2", cb2)

        bus.emit("event1")
        assert results == ["event1"]

        bus.emit("event2")
        assert results == ["event1", "event2"]

        bus.off("event1", cb1)
        bus.emit("event1")
        bus.emit("event2")
        assert results == ["event1", "event2", "event2"]
