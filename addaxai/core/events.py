"""Lightweight publish/subscribe event bus.

Usage:
    from addaxai.core.events import event_bus

    # Subscribe
    def on_progress(pct: float, message: str) -> None:
        print(f"{pct}%: {message}")

    event_bus.on("deploy.progress", on_progress)

    # Publish
    event_bus.emit("deploy.progress", pct=50.0, message="Processing image 5/10")

    # Unsubscribe
    event_bus.off("deploy.progress", on_progress)

    # Unsubscribe all listeners for an event
    event_bus.clear("deploy.progress")
"""

import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class EventBus:
    """Simple synchronous event bus.

    Not thread-safe — designed for single-threaded tkinter event loop usage.
    """

    def __init__(self) -> None:
        """Initialize the event bus."""
        self._listeners: Dict[str, List[Callable[..., Any]]] = {}

    def on(self, event: str, callback: Callable[..., Any]) -> None:
        """Register a callback for an event.

        Args:
            event: Event name (e.g., "deploy.progress")
            callback: Function to call when event is emitted
        """
        if event not in self._listeners:
            self._listeners[event] = []
        if callback not in self._listeners[event]:
            self._listeners[event].append(callback)

    def off(self, event: str, callback: Callable[..., Any]) -> None:
        """Remove a callback for an event.

        Args:
            event: Event name
            callback: Function to remove
        """
        if event in self._listeners:
            self._listeners[event] = [
                cb for cb in self._listeners[event] if cb is not callback
            ]

    def emit(self, event: str, **kwargs: Any) -> None:
        """Emit an event, calling all registered callbacks with kwargs.

        Args:
            event: Event name
            **kwargs: Data passed to callbacks
        """
        for callback in self._listeners.get(event, []):
            try:
                callback(**kwargs)
            except Exception:
                logger.error(
                    "Error in event handler for '%s'", event, exc_info=True
                )

    def clear(self, event: str) -> None:
        """Remove all listeners for an event.

        Args:
            event: Event name
        """
        self._listeners.pop(event, None)

    def clear_all(self) -> None:
        """Remove all listeners for all events."""
        self._listeners.clear()


# Module-level singleton — import this in all modules.
event_bus = EventBus()
