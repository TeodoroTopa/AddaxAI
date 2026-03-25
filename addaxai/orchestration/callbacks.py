"""Callback dataclass for GUI-free orchestrators.

OrchestratorCallbacks holds all injected callables that the orchestrator
functions need to interact with the outside world (UI, logging, etc.).

In GUI mode, these are wired to tkinter messageboxes and root.update().
In headless mode, these are wired to logging calls and no-ops.

Usage in app.py (GUI mode):
    callbacks = OrchestratorCallbacks(
        on_error=lambda title, msg: mb.showerror(title, msg),
        on_warning=lambda title, msg: mb.showwarning(title, msg),
        on_info=lambda title, msg: mb.showinfo(title, msg),
        on_confirm=lambda title, msg: mb.askyesno(title, msg),
        update_ui=root.update,
        cancel_check=lambda: state.cancel_deploy_model_pressed,
    )
    run_detection(config, callbacks)

Usage in headless / REST API mode:
    callbacks = OrchestratorCallbacks(
        on_error=lambda title, msg: logger.error("[%s] %s", title, msg),
        on_warning=lambda title, msg: logger.warning("[%s] %s", title, msg),
        on_info=lambda title, msg: logger.info("[%s] %s", title, msg),
        on_confirm=lambda title, msg: True,
        update_ui=lambda: None,
        cancel_check=lambda: False,
    )
    run_detection(config, callbacks)
"""

import dataclasses
from typing import Callable


@dataclasses.dataclass
class OrchestratorCallbacks:
    """Injected callbacks for GUI interaction during orchestration.

    In GUI mode: these call mb.showerror(), root.update(), etc.
    In headless mode: these log errors, no-op for UI updates, etc.

    Attributes:
        on_error:     Called with (title, message) to show an error dialog.
        on_warning:   Called with (title, message) to show a warning dialog.
        on_info:      Called with (title, message) to show an info dialog.
        on_confirm:   Called with (title, message); returns True if user confirmed.
        update_ui:    Pumps the GUI event loop (no-op in headless mode).
        cancel_check: Returns True if the user has requested cancellation.
    """

    on_error: Callable[[str, str], None]
    on_warning: Callable[[str, str], None]
    on_info: Callable[[str, str], None]
    on_confirm: Callable[[str, str], bool]
    update_ui: Callable[[], None]
    cancel_check: Callable[[], bool]
