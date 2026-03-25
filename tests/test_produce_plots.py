"""Tests for produce_plots() extracted from app.py into addaxai/analysis/plots.py.

Written BEFORE the implementation (TDD — Step B2).

Strategy: produce_plots() has heavy optional deps (matplotlib, plotly, folium,
pandas). We mock the plotting sub-calls to test the control-flow contract:
  - cancel_check() is polled at 5 checkpoints; returning True stops execution
  - events are emitted via event_bus at start and end
  - logo overlay runs after all plots complete (only if not cancelled)
  - function signature accepts the required injected parameters
"""

import os
from typing import Any, List, Tuple
from unittest.mock import MagicMock, call, patch

import pytest


# --- Importability ---


def test_produce_plots_importable():
    """produce_plots must be importable from addaxai.analysis.plots."""
    from addaxai.analysis.plots import produce_plots  # noqa: F401


def test_produce_plots_signature():
    """Must accept results_dir, cancel_check, cancel_func, logo_image,
    logo_width, logo_height without raising TypeError."""
    from addaxai.analysis.plots import produce_plots

    # We won't actually run it — just verify signature by inspecting parameters.
    import inspect
    sig = inspect.signature(produce_plots)
    params = set(sig.parameters.keys())
    assert "results_dir" in params
    assert "cancel_check" in params
    assert "cancel_func" in params
    assert "logo_image" in params
    assert "logo_width" in params
    assert "logo_height" in params


# --- Helpers to build minimal CSV fixtures ---


def _write_minimal_csvs(tmpdir: Any) -> str:
    """Write the two CSVs that produce_plots() reads and return results_dir."""
    import csv
    results_dir = str(tmpdir)
    det_csv = os.path.join(results_dir, "results_detections.csv")
    fil_csv = os.path.join(results_dir, "results_files.csv")

    # Minimal detections CSV (no GPS, no timestamps — triggers fewest code paths)
    with open(det_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "label", "DateTimeOriginal", "Latitude", "Longitude",
            "conf", "file",
        ])
        writer.writeheader()
        writer.writerow({
            "label": "animal",
            "DateTimeOriginal": "",
            "Latitude": "",
            "Longitude": "",
            "conf": "0.9",
            "file": "img001.jpg",
        })

    # Minimal files CSV
    with open(fil_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "n_detections"])
        writer.writeheader()
        writer.writerow({"file": "img001.jpg", "n_detections": "1"})

    return results_dir


# --- Cancellation contract ---


@pytest.fixture
def patched_produce_plots():
    """Patch all heavy optional-dep calls inside produce_plots so tests run
    without matplotlib / plotly / folium installed.

    Index map:
      [0] pd.read_csv
      [1] tqdm
      [2] calculate_time_span
      [3] overlay_logo
      [4] os.walk
      [5] _create_time_plots
      [6] _create_geo_plots
      [7] _create_pie_plots_detections
      [8] _create_pie_plots_files
      [9] _create_activity_patterns
    """
    patches = [
        patch("addaxai.analysis.plots.pd.read_csv"),
        patch("addaxai.analysis.plots.tqdm"),
        patch("addaxai.analysis.plots.calculate_time_span", return_value=(0, 0, 0, 0)),
        patch("addaxai.analysis.plots.overlay_logo"),
        patch("addaxai.analysis.plots.os.walk", return_value=[]),
        patch("addaxai.analysis.plots._create_time_plots"),
        patch("addaxai.analysis.plots._create_geo_plots"),
        patch("addaxai.analysis.plots._create_pie_plots_detections"),
        patch("addaxai.analysis.plots._create_pie_plots_files"),
        patch("addaxai.analysis.plots._create_activity_patterns"),
    ]
    mocks = []
    for p in patches:
        mocks.append(p.start())
    yield mocks
    for p in patches:
        p.stop()


def test_cancel_after_time_plots(tmp_path, patched_produce_plots):
    """cancel_check returning True at checkpoint 1 should stop execution before
    geo plots, pie charts, files pie, and activity plots."""
    from addaxai.analysis.plots import produce_plots
    from addaxai.core.events import event_bus
    from addaxai.core.event_types import POSTPROCESS_PROGRESS

    # Mock read_csv to return DataFrames with minimal valid structure
    import pandas as pd
    mock_det = pd.DataFrame({
        "label": ["animal"],
        "DateTimeOriginal": [None],
        "Latitude": [None],
        "Longitude": [None],
    })
    mock_fil = pd.DataFrame({"file": ["img.jpg"], "n_detections": [1]})
    patched_produce_plots[0].side_effect = [mock_det, mock_fil]

    # cancel_check: True at first call (after time plots)
    call_count = [0]
    def cancel_check():
        call_count[0] += 1
        return call_count[0] >= 1  # cancel immediately

    events = []
    def on_event(**kwargs):
        events.append(kwargs)
    event_bus.on(POSTPROCESS_PROGRESS, on_event)

    try:
        overlay_mock = patched_produce_plots[3]
        produce_plots(
            results_dir=str(tmp_path),
            cancel_check=cancel_check,
            cancel_func=lambda: None,
            logo_image=MagicMock(),
            logo_width=135,
            logo_height=50,
        )
        # If cancelled, overlay_logo should NOT have been called
        # (logo is applied only after all plots complete)
        overlay_mock.assert_not_called()
    finally:
        event_bus.off(POSTPROCESS_PROGRESS, on_event)


def test_no_cancel_runs_to_completion(tmp_path, patched_produce_plots):
    """When cancel_check always returns False, function runs to completion and
    emits POSTPROCESS_PROGRESS with status='done'."""
    from addaxai.analysis.plots import produce_plots
    from addaxai.core.events import event_bus
    from addaxai.core.event_types import POSTPROCESS_PROGRESS

    import pandas as pd
    mock_det = pd.DataFrame({
        "label": ["animal"],
        "DateTimeOriginal": [None],
        "Latitude": [None],
        "Longitude": [None],
    })
    mock_fil = pd.DataFrame({"file": ["img.jpg"], "n_detections": [1]})
    patched_produce_plots[0].side_effect = [mock_det, mock_fil]

    events = []
    def on_event(**kwargs):
        events.append(kwargs)
    event_bus.on(POSTPROCESS_PROGRESS, on_event)

    try:
        produce_plots(
            results_dir=str(tmp_path),
            cancel_check=lambda: False,
            cancel_func=lambda: None,
            logo_image=MagicMock(),
            logo_width=135,
            logo_height=50,
        )
    finally:
        event_bus.off(POSTPROCESS_PROGRESS, on_event)

    statuses = [e.get("status") for e in events]
    assert "done" in statuses, f"Expected 'done' in emitted statuses, got: {statuses}"


def test_logo_overlay_called_on_completion(tmp_path, patched_produce_plots):
    """overlay_logo should be called for each PNG found when not cancelled."""
    from addaxai.analysis.plots import produce_plots

    import pandas as pd
    mock_det = pd.DataFrame({
        "label": ["animal"],
        "DateTimeOriginal": [None],
        "Latitude": [None],
        "Longitude": [None],
    })
    mock_fil = pd.DataFrame({"file": ["img.jpg"], "n_detections": [1]})
    patched_produce_plots[0].side_effect = [mock_det, mock_fil]

    # Return two PNG files from os.walk
    fake_walk = [
        (str(tmp_path / "graphs"), [], ["plot1.png", "plot2.png"]),
    ]
    patched_produce_plots[4].return_value = iter(fake_walk)

    logo_mock = MagicMock()
    produce_plots(
        results_dir=str(tmp_path),
        cancel_check=lambda: False,
        cancel_func=lambda: None,
        logo_image=logo_mock,
        logo_width=135,
        logo_height=50,
    )

    overlay_mock = patched_produce_plots[3]
    assert overlay_mock.call_count == 2


def test_load_event_emitted(tmp_path, patched_produce_plots):
    """POSTPROCESS_PROGRESS with status='load' should be emitted at start."""
    from addaxai.analysis.plots import produce_plots
    from addaxai.core.events import event_bus
    from addaxai.core.event_types import POSTPROCESS_PROGRESS

    import pandas as pd
    mock_det = pd.DataFrame({
        "label": ["animal"],
        "DateTimeOriginal": [None],
        "Latitude": [None],
        "Longitude": [None],
    })
    mock_fil = pd.DataFrame({"file": ["img.jpg"], "n_detections": [1]})
    patched_produce_plots[0].side_effect = [mock_det, mock_fil]

    events = []
    def on_event(**kwargs):
        events.append(kwargs)
    event_bus.on(POSTPROCESS_PROGRESS, on_event)

    try:
        produce_plots(
            results_dir=str(tmp_path),
            cancel_check=lambda: False,
            cancel_func=lambda: None,
            logo_image=MagicMock(),
            logo_width=135,
            logo_height=50,
        )
    finally:
        event_bus.off(POSTPROCESS_PROGRESS, on_event)

    statuses = [e.get("status") for e in events]
    assert "load" in statuses, f"Expected 'load' emitted, got: {statuses}"
