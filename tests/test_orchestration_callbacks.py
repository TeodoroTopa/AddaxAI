"""Tests for OrchestratorCallbacks dataclass (Step B4, TDD red phase).

Written BEFORE the implementation.

Contracts verified:
  - OrchestratorCallbacks is importable from addaxai.orchestration.callbacks
  - It is a dataclass
  - All 6 fields are present: on_error, on_warning, on_info, on_confirm,
    update_ui, cancel_check
  - All field values are callable
  - A headless instance (logging-only callbacks) works without importing tkinter
  - Callable type signatures are honoured:
      on_error(title, message) -> None
      on_warning(title, message) -> None
      on_info(title, message) -> None
      on_confirm(title, message) -> bool
      update_ui() -> None
      cancel_check() -> bool
"""

import dataclasses
import logging

import pytest


# --- Importability ---


def test_orchestrator_callbacks_importable():
    from addaxai.orchestration.callbacks import OrchestratorCallbacks  # noqa: F401


# --- Is a dataclass ---


def test_orchestrator_callbacks_is_dataclass():
    from addaxai.orchestration.callbacks import OrchestratorCallbacks
    assert dataclasses.is_dataclass(OrchestratorCallbacks)


# --- Field presence ---


def test_orchestrator_callbacks_field_names():
    from addaxai.orchestration.callbacks import OrchestratorCallbacks
    fields = {f.name for f in dataclasses.fields(OrchestratorCallbacks)}
    required = {"on_error", "on_warning", "on_info", "on_confirm", "update_ui", "cancel_check"}
    assert required.issubset(fields), f"Missing fields: {required - fields}"


# --- Fixtures ---


@pytest.fixture
def headless_callbacks():
    """A fully-functional OrchestratorCallbacks that uses logging only — no tkinter."""
    from addaxai.orchestration.callbacks import OrchestratorCallbacks

    logger = logging.getLogger("test.headless")

    return OrchestratorCallbacks(
        on_error=lambda title, msg: logger.error("[%s] %s", title, msg),
        on_warning=lambda title, msg: logger.warning("[%s] %s", title, msg),
        on_info=lambda title, msg: logger.info("[%s] %s", title, msg),
        on_confirm=lambda title, msg: True,   # always confirm in headless mode
        update_ui=lambda: None,
        cancel_check=lambda: False,
    )


# --- All fields are callable ---


def test_on_error_is_callable(headless_callbacks):
    assert callable(headless_callbacks.on_error)


def test_on_warning_is_callable(headless_callbacks):
    assert callable(headless_callbacks.on_warning)


def test_on_info_is_callable(headless_callbacks):
    assert callable(headless_callbacks.on_info)


def test_on_confirm_is_callable(headless_callbacks):
    assert callable(headless_callbacks.on_confirm)


def test_update_ui_is_callable(headless_callbacks):
    assert callable(headless_callbacks.update_ui)


def test_cancel_check_is_callable(headless_callbacks):
    assert callable(headless_callbacks.cancel_check)


# --- Callable signatures work correctly ---


def test_on_error_accepts_two_strings(headless_callbacks):
    """on_error(title, message) should not raise."""
    headless_callbacks.on_error("Error Title", "Something went wrong")


def test_on_warning_accepts_two_strings(headless_callbacks):
    headless_callbacks.on_warning("Warning Title", "Be careful")


def test_on_info_accepts_two_strings(headless_callbacks):
    headless_callbacks.on_info("Info Title", "All good")


def test_on_confirm_returns_bool(headless_callbacks):
    result = headless_callbacks.on_confirm("Confirm?", "Are you sure?")
    assert isinstance(result, bool)


def test_update_ui_takes_no_args(headless_callbacks):
    """update_ui() should be callable with no arguments."""
    headless_callbacks.update_ui()


def test_cancel_check_returns_bool(headless_callbacks):
    result = headless_callbacks.cancel_check()
    assert isinstance(result, bool)


# --- cancel_check can return True (cancelled state) ---


def test_cancel_check_can_return_true():
    from addaxai.orchestration.callbacks import OrchestratorCallbacks
    callbacks = OrchestratorCallbacks(
        on_error=lambda t, m: None,
        on_warning=lambda t, m: None,
        on_info=lambda t, m: None,
        on_confirm=lambda t, m: False,
        update_ui=lambda: None,
        cancel_check=lambda: True,
    )
    assert callbacks.cancel_check() is True


# --- on_confirm can return False (user declined) ---


def test_on_confirm_can_return_false():
    from addaxai.orchestration.callbacks import OrchestratorCallbacks
    callbacks = OrchestratorCallbacks(
        on_error=lambda t, m: None,
        on_warning=lambda t, m: None,
        on_info=lambda t, m: None,
        on_confirm=lambda t, m: False,
        update_ui=lambda: None,
        cancel_check=lambda: False,
    )
    assert callbacks.on_confirm("title", "msg") is False


# --- No tkinter required to import the module ---


def test_callbacks_module_importable_without_tkinter():
    """addaxai.orchestration.callbacks must be importable in headless envs."""
    import addaxai.orchestration.callbacks  # noqa: F401
