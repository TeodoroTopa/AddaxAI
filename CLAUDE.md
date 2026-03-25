# AddaxAI — Developer Handbook

## What This App Does

AddaxAI is a desktop GUI application that helps field ecologists process camera trap images
using local computer vision. Users point it at a folder of images; it runs MegaDetector
(an object detection model) to find animals, people, and vehicles, then optionally runs a
species classification model on the detections. Results are written to JSON, and the app
provides postprocessing tools to separate images by detection type, export to CSV/XLSX/COCO,
generate pie charts and GPS heatmaps, and review detections in a human-in-the-loop (HITL)
workflow using LabelImg. The app supports English, Spanish, and French, runs fully offline,
and is packaged via PyInstaller for non-technical ecologists on Windows and macOS. The
classification models are heterogeneous — different models require different conda environments
and ML frameworks — so each model runs as a subprocess, often in an isolated conda env.

---

## Refactoring History (Phases 1–6)

The application originally lived in a single ~11,200-line file (`AddaxAI_GUI.py`) with no
separation between business logic, UI, model deployment, data processing, and localization.
All state was managed via 35 `global` declarations.

**Phase 1:** Extracted 58 pure backend functions (~1,378 lines) into 12 modules under
`addaxai/`. Each function was parameterized (globals became explicit arguments). The Phase 1
behavioral change log in the git history documents every case where behavior changed during
extraction — read it before touching `models/registry.py`, `models/deploy.py`, or
`processing/export.py`.

**Phase 2:** Replaced 662 inline language array lookups with a proper i18n system: a
`t("key")` function backed by `addaxai/i18n/{en,es,fr}.json` (~300 keys each).

**Phase 3:** Extracted 7 widget/dialog/tab classes into `addaxai/ui/`.

**Phase 4:** Eliminated all global state. An `AppState` dataclass (`addaxai/core/state.py`)
now owns all 35 former globals.

**Phase 5:** Added type annotations on all modules (Python 3.8 compatible), logging
infrastructure (`addaxai/core/logging.py`), and GitHub Actions CI pipeline.

**Phase 6:** Open-source readiness. Added `pyproject.toml`, CI jobs (mypy, coverage),
GitHub issue/PR templates, `.editorconfig`, `Makefile`, `.pre-commit-config.yaml`,
`addaxai/py.typed`. Added JSON schemas (`addaxai/schemas/`), integration test fixtures
(`tests/fixtures/`), `InferenceBackend` protocol (`addaxai/models/backend.py`), model
adapter template (`classification_utils/model_types/_template/`). Added event bus
(`addaxai/core/events.py`, `addaxai/core/event_types.py`) with dual-write wiring in
`deploy_model()`, `classify_detections()`, and `start_postprocess()`. Added view protocols
(`addaxai/ui/protocols.py`) and shell UI modules (`deploy_tab.py`, `postprocess_tab.py`,
`hitl_window.py`, `results_viewer.py`). Renamed `AddaxAI_GUI.py` → `addaxai/app.py`.
Added read-only REST API (`addaxai/api/server.py`). Fixed CI lint/typecheck for legacy
app.py patterns via documented suppressions in `ruff.toml`.

---

## Current State (as of 2026-03-22)

### Numbers

- **53 Python modules** under `addaxai/` (13,503 lines total)
- **`addaxai/app.py`**: 8,624 lines, 169 functions — still the monolith
- **34 test files** under `tests/` (5,214 lines)
- **428 tests** (420 passing, 8 GUI integration, 1 smoke, ~9 skipped for optional deps)
- **3 CI jobs**: unit tests (Python 3.9 + 3.11 with coverage), ruff lint, mypy typecheck

### What's Done

- All business logic extracted into `addaxai/` modules (config, models, processing, utils, i18n, analysis)
- All global state centralized in `AppState` dataclass
- Full type annotations (Python 3.8 compatible)
- Logging throughout (replaces all print() calls)
- Event bus infrastructure (`EventBus` class + 16 standard event type constants)
- Event bus dual-write: `deploy_model()`, `classify_detections()`, `start_postprocess()` emit events alongside old direct `progress_window.update_values()` calls
- View protocols defined: `DeployView`, `PostprocessView`, `HITLView`, `ResultsView`
- Shell UI modules exist (`deploy_tab.py`, `postprocess_tab.py`, `hitl_window.py`, `results_viewer.py`) but contain only event subscriptions and stub method bodies — the actual widget code is still in `app.py`
- JSON schemas for `global_vars.json`, `model_vars.json`, recognition output
- REST API with read-only endpoints: `GET /models`, `GET /results/{folder}`, `GET /health`
- `InferenceBackend` protocol and model adapter template

### What's NOT Done (the remaining extraction)

The event bus emits events but `app.py` still directly calls `state.progress_window.update_values()`,
`root.update()`, and `mb.showerror()` in the three orchestrators. The shell UI modules
subscribe to events but can't act on them because no widget code has been moved into them.
The dual-write needs to be completed: remove the old direct UI calls, make the orchestrators
communicate exclusively through the event bus, then move the widget construction code from
`app.py` into the feature modules. This is the Phase 7 work described below.

---

## Repository Layout

```
addaxai/
├── app.py                  # Main entry point — 8,624 lines of GUI + orchestration code
├── __init__.py
├── py.typed                # PEP 561 marker
├── core/
│   ├── config.py           # load_global_vars / write_global_vars / load_model_vars_for
│   ├── event_types.py      # 16 standard event constants (DEPLOY_*, CLASSIFY_*, POSTPROCESS_*, MODEL_*)
│   ├── events.py           # EventBus class + module-level event_bus singleton
│   ├── logging.py          # setup_logging()
│   ├── paths.py            # Path resolution helpers
│   ├── platform.py         # OS detection, DPI scaling, Python interpreter lookup
│   └── state.py            # AppState dataclass — owns all mutable state
├── models/
│   ├── backend.py          # InferenceBackend protocol (runtime_checkable)
│   ├── deploy.py           # cancel_subprocess, switch_yolov5_version, imitate_object_detection
│   └── registry.py         # fetch_known_models, set_up_unknown_model, environment_needs_downloading
├── processing/
│   ├── annotations.py      # Pascal VOC / COCO / YOLO XML conversion
│   ├── export.py           # csv_to_coco
│   └── postprocess.py      # move_files, format_size
├── analysis/
│   └── plots.py            # fig2img, overlay_logo, calculate_time_span
├── i18n/
│   ├── __init__.py         # t("key"), lang_idx(), i18n_set_language()
│   ├── en.json, es.json, fr.json
├── hitl/
│   └── __init__.py         # Stub — HITL data logic remains in app.py
├── ui/
│   ├── protocols.py        # DeployView, PostprocessView, HITLView, ResultsView (runtime_checkable)
│   ├── deploy_tab.py       # Shell — subscribes to deploy events, stub method bodies
│   ├── postprocess_tab.py  # Shell — subscribes to postprocess events, stub method bodies
│   ├── hitl_window.py      # Shell — stub method bodies only
│   ├── results_viewer.py   # Shell — stub method bodies only
│   ├── widgets/            # InfoButton, CancelButton, GreyTopButton, frames, SpeciesSelectionFrame
│   ├── dialogs/            # ProgressWindow, CustomWindow, PatienceWindow, etc. (1,345 lines total)
│   ├── advanced/           # help_tab.py, about_tab.py
│   └── simple/             # simple_window.py (build_simple_mode)
├── schemas/
│   ├── validate.py         # Manual validation (no jsonschema dependency)
│   ├── global_vars.schema.json, model_vars.schema.json, recognition_output.schema.json
├── api/
│   └── server.py           # FastAPI: GET /models, /results/{folder}, /health
└── utils/
    ├── files.py, images.py, json_ops.py
```

---

## Development Setup

```bash
# Clone the fork
git clone https://github.com/TeodoroTopa/AddaxAI.git
cd AddaxAI

# Unit test environment (Python 3.9+, no GUI deps needed)
python -m venv .venv
.venv/Scripts/pip install -e ".[test]"

# Run unit tests
make test

# Run linter
make lint

# Run type checker
make typecheck

# GUI environment (Python 3.8, full deps including MegaDetector)
# This is the installed app's conda env — do not modify it
C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe

# Launch the GUI for manual testing
make dev

# Run GUI integration tests (boots real GUI, ~15s per test)
make test-gui

# Run GUI smoke test (starts GUI, waits 10s, asserts no crash)
make test-smoke
```

**Remotes:**
- `origin` → `TeodoroTopa/AddaxAI` (fork — push here)
- `upstream` → `PetervanLunteren/AddaxAI` (original — pull updates from here)

---

## Development Conventions

- **TDD:** Write tests first, implement to make them pass, run full suite, commit.
- **One commit per logical step** — small, immediately pushable. Conventional commit
  prefixes: `feat`, `fix`, `refactor`, `ci`, `docs`, `chore`.
- **Extraction rule:** When moving code out of `app.py`, parameterize all globals
  (e.g. `AddaxAI_files` → `base_path`, `var_choose_folder.get()` → `base_folder`).
  Do not change behavior — pure mechanical moves only. Document exceptions in the
  commit message.
- **Type hints:** Use `typing` module generics throughout for Python 3.8 compatibility
  (`List`, `Dict`, `Optional`, not `list[str]`, `dict[str, Any]`, `X | None`).
  Use `Any` for all tkinter/customtkinter widget parameters.
- **Logging:** Use `logging.getLogger(__name__)` at the top of each module. Map levels as:
  function traces → `DEBUG`, subprocess output → `INFO`, warnings → `WARNING`,
  caught exceptions → `ERROR(..., exc_info=True)`.
- **No `global` declarations:** All mutable state goes through `AppState`. If you find
  yourself adding a global, add it to `AppState` instead.
- **Update CLAUDE.md** after any significant change to architecture, conventions, or status.

---

## Git Workflow

This repo is a fork. There are two remotes:
- `origin` → `TeodoroTopa/AddaxAI` (YOUR FORK — push here, open PRs here)
- `upstream` → `PetervanLunteren/AddaxAI` (ORIGINAL — never push here, never open PRs here)

For each feature branch:
```bash
git checkout main && git pull origin main
git checkout -b phase7/<branch-name>
# ... do the work, make commits ...
git push -u origin phase7/<branch-name>
gh pr create --repo TeodoroTopa/AddaxAI --base main --title "..." --body "..."
# Wait for CI, then:
gh pr merge <PR_NUMBER> --repo TeodoroTopa/AddaxAI --merge --delete-branch
git checkout main && git pull origin main
```

**NEVER run `gh pr create` without `--repo TeodoroTopa/AddaxAI`.** Without it, `gh`
defaults to the upstream repo, which creates unwanted PRs on someone else's repository.

---

## Test Suite

Tests are split by runtime because the GUI requires a specific conda env not available in CI.

**Unit tests** (`tests/test_*.py`, excluding GUI tests): Run with `.venv` Python 3.9+.
Fast (~12s). Import `addaxai/` modules directly. No tkinter, no conda, no models.
Current count: ~420 passing, ~9 skipped (optional deps: cv2, matplotlib, customtkinter).

**GUI integration tests** (`tests/test_gui_integration.py`): Run with env-base Python 3.8.
The `tests/gui_test_runner.py` harness `exec()`s `addaxai/app.py` with a patched
`AddaxAI_files` path, suppresses `main()`, initializes frame states manually, then
schedules each test via `root.after()`. 8 tests covering language cycling, mode switching,
folder selection, model dropdowns, frame toggles, reset, deploy validation, state attributes.

**GUI smoke test** (`tests/test_gui_smoke.py`): Launches GUI as subprocess, waits 10s,
asserts process is still alive. 1 test.

**CI** (`.github/workflows/test.yml`): Runs on pushes to `main` and PRs to `main`.
Three jobs: unit tests (Python 3.9 + 3.11 with coverage → Codecov), ruff lint, mypy.

---

## Watchouts and Known Issues

**Python version split:** Unit tests run on Python 3.9+ (`.venv`). The GUI runs on Python
3.8 (`env-base`). All `addaxai/` code must be Python 3.8 compatible — use `typing` generics,
not built-in generic syntax.

**app.py lint/typecheck suppressions:** `app.py` is a legacy monolith predating the
linting setup. Rather than fixing all legacy patterns in place, they are suppressed with
documentation. See `ruff.toml` per-file-ignores (10 codes) and ~29 `# type: ignore`
comments in `app.py`. These suppressions will shrink as code is extracted into modules.

**Phase 1 behavioral changes:** Several extracted functions behave differently from the
original. Key differences documented in git history:
- `sort_checkpoint_files` uses index `[2]` not `[1]` (bug fix)
- `get_hitl_var_in_json` returns `"never-started"` gracefully instead of crashing
- `csv_to_coco` uses `math.isnan()` instead of `type(val) == float` (bug fix)
- `cancel_subprocess` no longer re-enables UI buttons (caller handles that)
- `environment_needs_downloading` returns tuple not list

**customtkinter import pattern:** All UI modules use a try/except fallback pattern so they
can be imported without customtkinter installed (enabling unit tests). This causes mypy
errors suppressed with `# type: ignore` comments. Do not remove these.

**Model adapters:** `classification_utils/model_types/` is untouched. Each adapter runs
as a subprocess in its own conda env. The boilerplate duplication is intentional for
subprocess isolation — don't consolidate it.

---

## app.py TODO Comments (lines 8–54)

The original developer left 47 TODO comments at the top of `app.py`. These are feature
requests, bugs, and polish items accumulated over years. Key items:

- **BUG:** Windows file-in-use error when moving files during postprocessing + XLSX export
- **CLEAN:** Handle deleted images during processing (skip file, continue)
- **RESUME DOWNLOAD:** Atomic downloads — download to temp, move on success
- **ANNOTATION:** 15 sub-items for HITL/LabelImg workflow improvements
- **MERGE JSON:** Combine image and video JSONs (already works for timelapse)
- **JSON confidence tracking:** Keep confidence scores through the full pipeline
- **LAT LON 0,0:** Filter out 0,0 GPS coordinates from map creation
- **CSV/XLSX:** Add frame number, frame rate columns

Most of these become easier to address after the feature extraction in Phase 7. The
Windows file-in-use bug is the only production blocker.

---

## Phase 7 — Implementation Plans

Phase 7 has three tracks. Track A (event bus migration) is the critical path — it
unblocks feature extraction, headless deployment, and future cloud inference. Track C
(testing) is independent and can run in parallel. Track B (bug fixes) is deferred until
after Track A delivers the main architectural win.

### Track A: Event Bus Migration & Feature Extraction

See "Phase 7 Track A — Event Bus Migration" section below for detailed step-by-step
implementation instructions.

### Track B: Bug Fixes & Quick Wins (Deferred)

After Track A is complete, address these contained issues:

1. **Windows file-in-use bug** — Investigate `shutil.move` and file handle leaks in
   postprocessing. Likely involves `openpyxl`/`pandas` not closing file handles. Look
   at the `postprocess()` function and its callers. Fix with explicit `close()` or
   context managers.

2. **LAT LON 0,0 filter** — In the map creation code (search for `folium` or `HeatMap`
   usage in `app.py`), skip coordinates at exactly (0.0, 0.0) before plotting.

3. **Completion messagebox** — At the end of `deploy_model()` (near line 3127), add a
   summary messagebox showing: images processed, detections found, errors/warnings count.
   Use `mb.showinfo()` with a formatted message.

These are all small, contained fixes that don't require architectural changes.

### Track C: Testing Infrastructure

See "Phase 7 Track C — Testing Infrastructure" section below for detailed step-by-step
implementation instructions.

---

## Phase 7 Track A — Event Bus Migration

### Goal

Complete the event bus migration so that the three orchestrators (`deploy_model()`,
`classify_detections()`, `start_postprocess()`) communicate with the UI exclusively
through the event bus. Then move the actual widget code from `app.py` into the shell
UI modules. Target: reduce `app.py` from 8,624 lines to ~5,000 lines.

### Prerequisites

- Event bus (`addaxai/core/events.py`) — DONE
- Event type constants (`addaxai/core/event_types.py`) — DONE
- Dual-write emit calls in all three orchestrators — DONE
- View protocols (`addaxai/ui/protocols.py`) — DONE
- Shell UI modules (deploy_tab, postprocess_tab, hitl_window, results_viewer) — DONE

### Important Context

The three orchestrators currently do two things at each progress point:
1. **Emit an event** (new): `event_bus.emit(DEPLOY_PROGRESS, pct=50.0, message="...")`
2. **Directly update the UI** (old): `state.progress_window.update_values(process=..., status=...)`

Both happen. The goal is to remove #2 and have the UI modules respond to #1 instead.

The `ProgressWindow` class (`addaxai/ui/dialogs/progress.py`, 762 lines) is the primary
UI component that receives these direct calls. It has an `update_values()` method that
takes `process` and `status` parameters and updates labels, progress bars, and button
states.

The three orchestrators also call `mb.showerror()`, `mb.showinfo()`, and `mb.askyesno()`
for user-facing messages. These messagebox calls are tightly coupled to the GUI. For
Phase 7 Track A, leave messagebox calls in place — they are a separate concern that can
be addressed later.

### Branch Setup

```bash
git checkout main && git pull origin main
git checkout -b phase7/event-bus-migration
```

### Step A1: Make deploy_tab.py respond to events by updating ProgressWindow

**What to do:** The `DeployTab` class already subscribes to `DEPLOY_PROGRESS`,
`DEPLOY_ERROR`, and `DEPLOY_FINISHED` events. Currently its handler methods
(`_on_deploy_progress`, etc.) call stub `show_progress()`/`show_error()` methods
that do nothing useful. Wire them to actually call `state.progress_window.update_values()`.

**Files to modify:**
- `addaxai/ui/deploy_tab.py`

**Detailed instructions:**

1. Read `addaxai/ui/dialogs/progress.py` to understand the `update_values()` method
   signature. It takes `process` (string like `"img_det"`, `"vid_det"`) and `status`
   (string like `"load"`, `"done"`, or a percentage string).

2. Read `addaxai/app.py` lines 2840-3140 (`deploy_model()`) and note every
   `state.progress_window.update_values()` call. Record the exact `process` and
   `status` arguments used.

3. Add `data_type` as a parameter to `DeployTab.__init__()` (or accept it via a setter
   method), since the `process` argument to `update_values()` depends on whether we're
   processing images or videos (`"img_det"` vs `"vid_det"`).

4. Update `_on_deploy_progress()` to call
   `state.progress_window.update_values(process=..., status=...)` with the correct
   arguments, translating from the event's `pct`/`message` kwargs to the
   `update_values()` format.

5. Update `_on_deploy_finished()` to call
   `state.progress_window.update_values(process=..., status="done")`.

6. Do NOT remove any code from `app.py` yet. This step only makes the UI module
   capable of responding to events. The old direct calls in `app.py` will still fire
   too (harmless duplication).

**Tests:** Run `make test` — all existing tests must pass. No new tests needed for
this step (it's a wiring change, not new logic).

**Commit:** `feat: wire deploy_tab event handlers to ProgressWindow`

### Step A2: Make postprocess_tab.py respond to events

**What to do:** Same pattern as A1, but for `PostprocessTab`. Read `start_postprocess()`
(lines 1220-1365) and the `postprocess()` function (lines 247-843) to understand all
`state.progress_window.update_values()` calls during postprocessing.

**Files to modify:**
- `addaxai/ui/postprocess_tab.py`

**Detailed instructions:**

1. Read `app.py` lines 247-843 (`postprocess()`) and lines 1220-1365
   (`start_postprocess()`). Note every `progress_window.update_values()` call with
   its `process` and `status` arguments.

2. The postprocessing progress calls use process strings like `"img_pst"` or `"vid_pst"`.

3. Add event handler methods that call `state.progress_window.update_values()` with the
   correct arguments, similar to step A1.

4. The `POSTPROCESS_PROGRESS` event is NOT currently emitted from inside the
   `postprocess()` function (lines 247-843) — only `POSTPROCESS_STARTED`,
   `POSTPROCESS_FINISHED`, and `POSTPROCESS_ERROR` are emitted from `start_postprocess()`.
   If per-step progress events are needed, add `event_bus.emit(POSTPROCESS_PROGRESS, ...)`
   calls inside `postprocess()` at each point where `progress_window.update_values()` is
   called. This is still dual-write — adding emit calls alongside existing direct calls.

5. Do NOT remove any code from `app.py` yet.

**Tests:** `make test` — all pass.

**Commit:** `feat: wire postprocess_tab event handlers to ProgressWindow`

### Step A3: Make classify events work through the event handlers

**What to do:** The classification flow uses `classify_detections()` (lines 2601-2806).
Wire the classify event handlers similarly.

**Files to modify:**
- `addaxai/ui/deploy_tab.py` (classification events are part of the deploy workflow —
  classification runs after detection within the same deployment pipeline)

**Detailed instructions:**

1. Read `classify_detections()` (lines 2601-2806). Note all
   `state.progress_window.update_values()` calls. The process string is
   `"img_cls"` or `"vid_cls"`.

2. `DeployTab` should also subscribe to `CLASSIFY_PROGRESS`, `CLASSIFY_FINISHED`,
   and `CLASSIFY_ERROR` events, since classification is part of the deploy flow.
   Add these subscriptions in `DeployTab.__init__()`.

3. Add corresponding handler methods that translate the event kwargs into
   `progress_window.update_values()` calls.

4. Add `event_bus.emit(CLASSIFY_PROGRESS, ...)` calls at each progress point inside
   `classify_detections()` where they don't already exist (check existing emit calls
   on lines 2727, 2786, 2796). Dual-write — keep the old direct calls.

5. Do NOT remove any code from `app.py` yet.

**Tests:** `make test` — all pass.

**Commit:** `feat: wire classify event handlers to deploy_tab`

### Step A4: Remove dual-write from deploy_model()

**What to do:** Now that the UI modules respond to events, remove the old direct
`state.progress_window.update_values()` calls from `deploy_model()`.

**Files to modify:**
- `addaxai/app.py` (the `deploy_model()` function, lines ~2809-3140)

**Detailed instructions:**

1. In `deploy_model()`, find every `state.progress_window.update_values()` call.
   There are approximately 6 calls (lines 2843, 3051, 3057, 3086, 3100, and a few
   others in that range).

2. For each call, verify that there is a corresponding `event_bus.emit()` call nearby
   that provides equivalent information. If any emit call is missing, add it.

3. Remove the `state.progress_window.update_values()` calls. Leave the
   `event_bus.emit()` calls in place.

4. Also remove any `root.update()` calls that were only needed to force the progress
   window to redraw — the event handler in `deploy_tab.py` should handle this.

5. Do NOT remove messagebox calls (`mb.showerror()`, `mb.askyesno()`, etc.) — those
   stay for now.

6. Do NOT remove `state.progress_window.close()` calls — those stay for now.

**Tests:**
- `make test` — all unit tests pass
- Run the GUI smoke test: `make test-smoke` — GUI still boots without crash
- If possible, run GUI integration tests: `make test-gui` — all 8 pass

**Commit:** `refactor: remove direct progress_window calls from deploy_model (event bus only)`

### Step A5: Remove dual-write from classify_detections()

**What to do:** Same pattern as A4 but for `classify_detections()`.

**Files to modify:**
- `addaxai/app.py` (the `classify_detections()` function, lines ~2601-2806)

**Detailed instructions:**

1. Find every `state.progress_window.update_values()` call in `classify_detections()`.
   There are approximately 3 calls (lines 2609, 2777, 2790).

2. Verify corresponding emit calls exist. Add any missing ones.

3. Remove the direct `progress_window.update_values()` calls.

4. Keep `state.progress_window.close()` (line 2806).

5. Keep all messagebox calls.

**Tests:** Same as A4.

**Commit:** `refactor: remove direct progress_window calls from classify_detections (event bus only)`

### Step A6: Remove dual-write from start_postprocess() and postprocess()

**What to do:** Same pattern for the postprocessing orchestrator.

**Files to modify:**
- `addaxai/app.py` (the `start_postprocess()` function and the `postprocess()` function)

**Detailed instructions:**

1. Find every `state.progress_window.update_values()` call in `start_postprocess()`
   (lines 1220-1365) and `postprocess()` (lines 247-843). There are approximately
   7 calls across both functions.

2. For `postprocess()`, which has per-step progress calls but may not have corresponding
   emit calls yet (since step A2 may have added them), verify all emit calls are in place.

3. Remove the direct `progress_window.update_values()` calls.

4. Keep `state.progress_window.close()` calls (lines 1348, 1363).

5. Keep all messagebox calls.

**Tests:** Same as A4.

**Commit:** `refactor: remove direct progress_window calls from postprocessing (event bus only)`

### Step A7: Add event bus tests for orchestrator event sequences

**What to do:** Write tests that verify the orchestrators emit the correct events in the
correct order. These tests subscribe to the event bus, call the orchestrator (with mocked
subprocess and UI), and assert the event sequence.

**Files to create:**
- `tests/test_orchestrator_events.py`

**Detailed instructions:**

1. Create `tests/test_orchestrator_events.py`.

2. These tests do NOT need a real GUI or real models. They test that the event bus
   receives the right events. The challenge is that the orchestrators (`deploy_model`,
   `classify_detections`, `start_postprocess`) reference many globals and GUI widgets
   from `app.py`. You have two options:

   a. **Test at the event bus level only:** Subscribe to events, manually emit them
      in the order the orchestrator would, and verify handlers receive them. This
      tests the event bus wiring without needing the orchestrators themselves.

   b. **Mock the orchestrator dependencies:** If feasible, import the orchestrator
      function, mock `state`, `root`, `mb`, `subprocess`, etc., and call it. This
      is harder because the orchestrators use many module-level variables from `app.py`.

   Option (a) is recommended for this step. Option (b) is Track C work.

3. Test cases:
   - Subscribe to DEPLOY_STARTED, DEPLOY_PROGRESS, DEPLOY_FINISHED. Emit them in
     order. Verify all callbacks fired with correct kwargs.
   - Subscribe to DEPLOY_ERROR. Emit it. Verify the error message is received.
   - Subscribe to CLASSIFY_STARTED, CLASSIFY_PROGRESS, CLASSIFY_FINISHED. Same pattern.
   - Subscribe to POSTPROCESS_STARTED, POSTPROCESS_PROGRESS, POSTPROCESS_FINISHED. Same.
   - Test that emitting DEPLOY_PROGRESS with pct and message kwargs arrives correctly.
   - Test that the event bus `clear_all()` in setUp/tearDown prevents cross-test leakage.

4. Target: 10-15 tests.

**Tests:** `make test` — all pass including the new tests.

**Commit:** `test: add event sequence tests for orchestrator event flows`

### Step A8: Push, PR, merge

```bash
git push -u origin phase7/event-bus-migration
gh pr create --repo TeodoroTopa/AddaxAI --base main \
  --title "refactor: complete event bus migration — remove dual-write" \
  --body "$(cat <<'EOF'
## Summary
- Wire deploy_tab and postprocess_tab event handlers to ProgressWindow
- Add classify event subscriptions to deploy_tab
- Remove all direct progress_window.update_values() calls from deploy_model(),
  classify_detections(), and start_postprocess()/postprocess()
- Orchestrators now communicate with UI exclusively through event bus
- Add event sequence tests

## Test plan
- [ ] All unit tests pass (make test)
- [ ] GUI smoke test passes (make test-smoke)
- [ ] GUI integration tests pass (make test-gui)
- [ ] Manual test: run detection on a folder, verify progress displays correctly

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
# Wait for CI, then:
gh pr merge <PR_NUMBER> --repo TeodoroTopa/AddaxAI --merge --delete-branch
git checkout main && git pull origin main
```

### Step A9: Extract postprocess widget code from app.py into postprocess_tab.py

**Lessons from Track A steps 1-8:** The previous steps introduced 6 defects because
the implementer (a) did not read `update_values()` to understand its full parameter
contract before writing code, (b) invented a lossy translation layer instead of
forwarding data transparently, and (c) had no verification step to confirm zero direct
calls remained. These rewritten instructions address each failure mode explicitly.

**CRITICAL RULES for all extraction steps:**
- **Read before writing.** Before modifying any file, read the exact code you are about
  to change in its current form. Do not rely on line numbers from this document — they
  shift as you edit. Always re-read after each edit to verify the result.
- **Pure mechanical moves.** Do not change behavior, rename variables, add abstractions,
  or "improve" code as you move it. Copy-paste the exact code, then change only the
  minimum needed (e.g., replacing `fth_step` with `self.parent_frame` if the frame
  reference changes).
- **One commit per extraction.** Each area (postprocess, deploy, HITL) is a separate
  commit. Run `make test` and `make lint` after each commit.
- **Verify with grep.** After each extraction, grep `app.py` to confirm the moved code
  is gone and grep the destination file to confirm it arrived.

**What to do:** Move the postprocessing widget construction code (Step 4 / `fth_step`)
from `app.py` into `PostprocessTab`. This is the simplest extraction because the
postprocess widgets are self-contained within the `fth_step` LabelFrame.

**Branch setup:**
```bash
git checkout main && git pull origin main
git checkout -b phase7/extract-postprocess-widgets
```

**Files to modify:**
- `addaxai/ui/postprocess_tab.py` — add widget construction methods
- `addaxai/app.py` — remove moved widget code, call PostprocessTab methods instead

**Detailed instructions:**

1. **Read the source code you will move.** Open `app.py` and read the section starting
   at `### fourth step` (search for that exact string). This section starts at
   approximately line 8240 and contains:
   - `fth_step` LabelFrame creation and grid configuration (5 lines)
   - `PostprocessTab` instantiation (1 line — already exists, keep it)
   - Output directory widgets: `lbl_output_dir`, `dsp_output_dir`, `btn_output_dir`
   - Separate files checkbox: `lbl_separate_files`, `chb_separate_files`
   - Separation sub-frame: `sep_frame` with threshold, file placement, confidence options
   - Visualize checkbox + sub-frame: `chb_vis_files`, `vis_frame` with size/bbox/blur
   - Crop checkbox: `lbl_crp_files`, `chb_crp_files`
   - Export checkbox + format dropdown: `lbl_exp`, `chb_exp`, `dpd_exp_format`
   - Plot checkbox: `lbl_plt`, `chb_plt`
   - Start postprocess button: `btn_start_postprocess`

   Count the exact number of lines. It should be approximately 200-250 lines between
   `### fourth step` and the next major section.

2. **Read `PostprocessTab` in `addaxai/ui/postprocess_tab.py`.** Currently it has
   `__init__`, event handlers, and stub methods. You will add a `build_widgets()` method.

3. **Add a `build_widgets()` method to `PostprocessTab`.** This method should:
   - Accept the same parameters the widget code currently uses from `app.py`'s scope:
     `global_vars` dict, `var_*` tkinter variables, `green_primary` color, `text_font`,
     `label_width`, `widget_width`, styling constants, and callback functions.
   - Contain the exact widget construction code from `app.py`, with these substitutions:
     - `fth_step` → `self.parent_frame` (the frame passed to `__init__`)
     - Any `state.xxx = widget` assignments should use `self.xxx = widget` instead
   - Return nothing — widgets are attached to `self.parent_frame` via `.grid()`.

4. **In `app.py`, replace the widget construction block** with a single call:
   ```python
   postprocess_view.build_widgets(
       global_vars=global_vars,
       var_output_dir=var_output_dir,
       # ... all other params ...
   )
   ```
   Keep the `fth_step` LabelFrame creation in `app.py` (it belongs to the tab layout).
   Move everything inside the frame into `PostprocessTab.build_widgets()`.

5. **Do NOT move these things:**
   - The `fth_step` LabelFrame creation itself (stays in app.py tab layout)
   - The `start_postprocess()` function (orchestration logic, stays in app.py)
   - Any `state.progress_window` references (handled by event bus)
   - The `postprocess()` function (backend logic, stays in app.py)

6. **Verification checklist** (run all of these):
   - `grep -n "fth_step" addaxai/ui/postprocess_tab.py` — should show zero matches
     (we use `self.parent_frame`, not `fth_step`)
   - `grep -c "lbl_output_dir\|chb_separate_files\|chb_vis_files\|chb_crp_files\|chb_exp\|chb_plt" addaxai/app.py`
     — count should decrease by the number of widget lines you moved
   - `make test` — all 463+ tests pass
   - `make lint` — no new lint errors in modified files
   - `make test-smoke` — GUI boots without crash (if env-base available)

**Tests:** `make test` — all pass. No new tests needed for a pure mechanical move.

**Commit:** `refactor: extract postprocess widget construction into PostprocessTab.build_widgets()`

### Step A10: Extract deploy widget code from app.py into deploy_tab.py

**What to do:** Same pattern as A9 but for the deployment widgets (Step 2 / `snd_step`).
This is harder because the deploy step has more widgets, sub-frames (detection model
dropdown, classification model dropdown, video frame settings), and more complex
interactions (model dropdown updates, frame state toggling).

**Files to modify:**
- `addaxai/ui/deploy_tab.py`
- `addaxai/app.py`

**Detailed instructions:**

1. **Read the source code.** In `app.py`, find `### second step` (approximately line
   7944). The deploy widgets include:
   - `snd_step` LabelFrame creation
   - Detection model dropdown: `dpd_model`, `var_det_model`
   - Classification model dropdown: `dpd_cls_model`, `var_cls_model`
   - Classification sub-frame: `cls_frame` with species selection, confidence threshold
   - Video-specific sub-frame: `vid_frame` with frame extraction settings
   - Deploy button: `btn_start_deploy` (line ~8214)
   - `DeployTab` instantiation (already exists)

   This section runs from `### second step` through just before `### human-in-the-loop
   step` (line ~8222). Approximately 270 lines.

2. **Identify coupled code.** Unlike postprocess, the deploy widgets have external
   coupling:
   - `update_dpd_options()` calls reference `dpd_model` and `dpd_cls_model`
   - `update_frame_states()` references `snd_step`
   - `change_language()` references `snd_step` and dropdown widgets
   - `start_deploy()` references `btn_start_deploy`, `sim_run_btn`
   - Multiple functions reference `btn_start_deploy.configure(state=NORMAL/DISABLED)`

   For each of these, the widget reference must be accessible after extraction. Use
   `deploy_view.get_widget("btn_start_deploy")` or store as `state.btn_start_deploy`
   (already done for the button).

3. **Add a `build_widgets()` method to `DeployTab`.** Same pattern as A9.

4. **Gradually move widgets**, testing after each sub-move:
   a. First move just the detection model dropdown section
   b. Run `make test` and `make test-smoke`
   c. Then move the classification model section
   d. Then move the video frame section
   e. Then move the deploy button (already partially extracted)

5. **Verification checklist:** Same as A9 but check for `snd_step` references.

**Commit:** `refactor: extract deploy widget construction into DeployTab.build_widgets()`

### Step A11: Extract HITL widget code from app.py into hitl_window.py

**What to do:** Move the HITL step widgets (Step 3 / `trd_step`) into `HITLWindow`.
This is the smallest extraction — only ~10 lines of widget code (a label and a button).

**Files to modify:**
- `addaxai/ui/hitl_window.py`
- `addaxai/app.py`

**Detailed instructions:**

1. In `app.py`, find `### human-in-the-loop step` (line ~8222). This section has:
   - `trd_step` LabelFrame creation (5 lines)
   - `HITLWindow` instantiation (1 line — keep)
   - `lbl_hitl_main` label (2 lines)
   - `btn_hitl_main` button with `start_or_continue_hitl` command (2 lines)

2. Move the label and button creation into `HITLWindow.build_widgets()`.

3. The `start_or_continue_hitl` callback stays in `app.py` — pass it as a parameter.

**Commit:** `refactor: extract HITL widget construction into HITLWindow.build_widgets()`

### Step A12: Push, PR, merge

```bash
git push -u origin phase7/extract-postprocess-widgets
gh pr create --repo TeodoroTopa/AddaxAI --base main \
  --title "refactor: extract widget code from app.py into UI modules" \
  --body "$(cat <<'EOF'
## Summary
- Move postprocess widget construction into PostprocessTab.build_widgets()
- Move deploy widget construction into DeployTab.build_widgets()
- Move HITL widget construction into HITLWindow.build_widgets()
- app.py reduced from ~8,600 lines to ~X,XXX lines

## Test plan
- [ ] All unit tests pass (make test)
- [ ] GUI smoke test passes (make test-smoke)
- [ ] GUI integration tests pass (make test-gui)
- [ ] Manual test: all tabs render correctly, buttons work

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
# Wait for CI, then:
gh pr merge <PR_NUMBER> --repo TeodoroTopa/AddaxAI --merge --delete-branch
git checkout main && git pull origin main
```

---

## Phase 7 Track C — Testing Infrastructure

### Goal

Add subprocess-boundary tests that verify the orchestration logic without needing real
models or a GUI. These tests mock `subprocess.Popen` at the boundary, feed canned stdout,
and assert that the orchestration logic produces correct events and JSON output.

### Prerequisites

- Event bus migration (Track A steps A1-A8) is complete.
- The fix PR (#12) is merged — all `event_bus.emit()` calls now carry full `update_values()`
  parameters (`status`, `cur_it`, `tot_it`, `time_ela`, `time_rem`, `speed`, `hware`,
  `cancel_func`, `extracting_frames_txt`, `frame_video_choice`).
- `tests/test_ui_event_wiring.py` already exists with 23 tests covering handler→
  ProgressWindow forwarding. Step C2 is therefore **already done** — do not recreate it.
- `tests/test_orchestrator_events.py` already exists with 20 tests covering event bus
  delivery. Step A7 tests are done.
- `tests/conftest.py` does **not** exist yet — must be created in C1.

### Lessons from Track A (read before starting)

Track A steps 1-8 introduced 6 defects. The root causes were:

1. **Not reading the target interface before writing code.** The implementer did not read
   `ProgressWindow.update_values()` (762 lines in `addaxai/ui/dialogs/progress.py`) to
   understand its full parameter contract. It takes `process`, `status`, `cur_it`, `tot_it`,
   `time_ela`, `time_rem`, `speed`, `hware`, `cancel_func`, `extracting_frames_txt`,
   `frame_video_choice` — but the implementer only passed `pct` and `message`.

2. **Inventing abstractions instead of forwarding data.** The implementer created a lossy
   translation layer that reconstructed `cur_it=int(pct), tot_it=100` from a percentage —
   completely wrong. The fix was transparent forwarding: pass kwargs through unchanged.

3. **No verification step.** The instructions said "remove direct calls" but didn't say
   "then grep to confirm zero remain." Three direct calls in `produce_plots()` were missed.

**To avoid repeating these mistakes:**
- Always read the source file you are about to consume/call BEFORE writing any code.
- When extracting/wrapping functions, pass data through transparently. Do not reinterpret,
  reconstruct, or lossy-compress parameters.
- After every step, run verification greps to confirm expected state.
- When the instructions say "read X", actually read it. Do not skip this.

### Branch Setup

```bash
git checkout main && git pull origin main
git checkout -b phase7/orchestration-tests
```

### Step C1: Create conftest.py with test fixtures

**What to do:** Create `tests/conftest.py` with two fixtures: `mock_app_env` (temporary
directory structure) and `event_collector` (subscribes to all events, collects them).

**File to create:** `tests/conftest.py`

**`tests/conftest.py` does not currently exist.** Verify this by running:
```bash
ls tests/conftest.py 2>&1 || echo "DOES NOT EXIST"
```

**Detailed instructions:**

1. Create `tests/conftest.py` with these exact contents (adjust imports as needed):

```python
"""Shared test fixtures for AddaxAI test suite."""

import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Tuple

import pytest

from addaxai.core.events import event_bus
from addaxai.core import event_types


@pytest.fixture
def mock_app_env():
    """Create a temporary directory structure mimicking AddaxAI_files/.

    Yields a dict with paths:
        base_path: Root temp directory (like AddaxAI_files/)
        image_folder: Folder with copied fixture images
        json_path: Path to a valid global_vars.json
    """
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    images_dir = os.path.join(fixtures_dir, "images")
    global_vars_path = os.path.join(fixtures_dir, "global_vars_valid.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure
        os.makedirs(os.path.join(tmpdir, "models", "det"))
        os.makedirs(os.path.join(tmpdir, "models", "cls"))
        image_folder = os.path.join(tmpdir, "test_images")
        os.makedirs(image_folder)

        # Copy fixture images if they exist
        if os.path.isdir(images_dir):
            for img in os.listdir(images_dir):
                shutil.copy2(os.path.join(images_dir, img), image_folder)

        # Copy global_vars.json
        json_dest = os.path.join(tmpdir, "global_vars.json")
        if os.path.isfile(global_vars_path):
            shutil.copy2(global_vars_path, json_dest)

        yield {
            "base_path": tmpdir,
            "image_folder": image_folder,
            "json_path": json_dest,
        }


@pytest.fixture
def event_collector():
    """Subscribe to ALL event types and collect emitted events.

    Yields a list of (event_name, kwargs) tuples. The list is populated
    as events are emitted during the test.

    Teardown clears all event bus subscriptions to prevent cross-test leakage.
    """
    collected = []  # type: List[Tuple[str, Dict[str, Any]]]

    def make_handler(name):
        # type: (str) -> Any
        def handler(**kwargs):
            # type: (**Any) -> None
            collected.append((name, kwargs))
        return handler

    handlers = {}  # type: Dict[str, Any]
    for attr in dir(event_types):
        val = getattr(event_types, attr)
        if isinstance(val, str) and "." in val:
            h = make_handler(val)
            handlers[val] = h
            event_bus.on(val, h)

    yield collected

    for evt_name, handler in handlers.items():
        event_bus.off(evt_name, handler)
    event_bus.clear_all()
```

2. **Verification:** Run `make test`. All 463+ existing tests must still pass. The new
   fixtures are available but not yet used by any test — they should not cause failures.

3. **Check that the fixture images exist** by running:
   ```bash
   ls tests/fixtures/images/
   ```
   Expected: `test_animal.jpg`, `test_empty.jpg`, `test_multi.jpg`, `test_person.jpg`,
   `test_vehicle.jpg`. If missing, the `mock_app_env` fixture will still work but the
   image folder will be empty.

**Tests:** `make test` — all pass (no new tests yet, just fixtures).

**Commit:** `test: add conftest.py with mock_app_env and event_collector fixtures`

### Step C2: ALREADY DONE — UI event integration tests exist

**This step is already complete.** The file `tests/test_ui_event_wiring.py` contains
23 tests across 5 test classes that verify:
- `DeployTab` forwards DEPLOY_PROGRESS events to `progress_window.update_values()`
  with correct kwargs (status, cur_it, tot_it, time_ela, time_rem, speed, hware,
  cancel_func, frame_video_choice, extracting_frames_txt)
- `DeployTab` forwards CLASSIFY_PROGRESS events with smoothing status support
- `PostprocessTab` forwards POSTPROCESS_PROGRESS events including plt process type
- Extra kwargs (pct, message) are filtered out of update_values calls
- Events without process or status kwargs are not forwarded
- Missing progress_window (None) does not crash

**Do not recreate this file.** Skip to Step C3.

### Step C3: Extract and test subprocess stdout parsing

**What to do:** Extract the stdout parsing loops from `deploy_model()` and
`classify_detections()` in `app.py` into pure functions in `addaxai/models/deploy.py`,
then write tests for those functions.

**Why extract?** The parsing logic is currently embedded inside `deploy_model()` which
references ~20 module-level variables from `app.py` (tkinter vars, `root`, `state`, etc.).
Testing it directly would require mocking all of those. Extracting the parsing into a
pure function with a callback makes it trivially testable.

**Files to modify:**
- `addaxai/models/deploy.py` — add `parse_detection_stdout()` and `parse_classification_stdout()`
- `addaxai/app.py` — replace inline parsing with calls to the extracted functions

**Files to create:**
- `tests/test_deploy_subprocess.py`

**Detailed instructions:**

1. **Read the parsing code you will extract.** In `app.py`, find the `for line in p.stdout:`
   loop inside `deploy_model()` (search for `# read output` near line 2970). Read from
   there through `# process is done` (line ~3089). This is the detection parsing loop.
   It handles these line patterns:
   - `"No image files found"` → error
   - `"No videos found"` → error
   - `"No frames extracted"` → error
   - `"UnicodeEncodeError:"` → error
   - `"Exception:"` → log to error file
   - `"Warning:"` → log to warning file (with 4 exclusion patterns)
   - `"Extracting frames for folder "` → frame extraction mode start
   - `"Extracted frames for"` → frame extraction mode end
   - `'%' in line[0:4]` during extracting_frames_mode → frame extraction progress
   - `"GPU available: False"` → set GPU_param = "CPU"
   - `"GPU available: True"` → set GPU_param = "GPU"
   - `'%' in line[0:4]` in normal mode → parse tqdm progress bar

   The tqdm progress format is: `" 50%|████ | 5/10 [00:05<00:05, 1.0it/s]"`
   The regex parsing extracts: percentage, current_im, total_im, elapsed_time, time_left,
   processing_speed from this format.

2. **Read the classify parsing code.** In `app.py`, find `for line in p.stdout:` inside
   `classify_detections()` (near line 2716). It handles:
   - `"n_crops_to_classify is zero."` → early exit
   - `"<EA>"` lines → smoothing output
   - `"<EA-status-change>"` → status change (e.g., "smoothing")
   - `"GPU available: False/True"` → GPU detection
   - `'%' in line[0:4]` → same tqdm parsing as detection

3. **Create `parse_detection_stdout()` in `addaxai/models/deploy.py`.** The function
   signature should be:

   ```python
   def parse_detection_stdout(
       stdout_lines,          # Iterable[str] — lines from subprocess stdout
       data_type,             # str — "img" or "vid"
       emit_progress,         # Callable[..., None] — called with event kwargs
       emit_error,            # Callable[..., None] — called with event kwargs
       log_exception,         # Callable[[str], None] — log exception lines
       log_warning,           # Callable[[str], None] — log warning lines
       cancel_func_factory,   # Callable[[], Callable] — returns cancel callback
       frame_video_choice,    # Optional[str] — "frame", "video", or None
   ):
       # type: (...) -> Optional[str]
       """Parse MegaDetector subprocess stdout and emit progress events.

       Returns the last successfully processed image path (for error messages),
       or None if no images were processed.
       """
   ```

   **CRITICAL: The `emit_progress` callback must be called with the EXACT SAME kwargs
   that the current `event_bus.emit(DEPLOY_PROGRESS, ...)` calls use.** Read each emit
   call in the existing code and replicate the kwargs exactly. For example, the running
   status emit currently passes: `pct`, `message`, `process`, `status`, `cur_it`, `tot_it`,
   `time_ela`, `time_rem`, `speed`, `hware`, `cancel_func`, `frame_video_choice`. Do NOT
   drop any of these kwargs. Do NOT invent new kwargs.

4. **Create `parse_classification_stdout()` similarly.** It needs `status_setting` handling
   (starts as `"running"`, changes on `<EA-status-change>` lines).

5. **In `app.py`, replace the inline loops** with calls to these functions. Example:
   ```python
   previous_processed_img = parse_detection_stdout(
       stdout_lines=p.stdout,
       data_type=data_type,
       emit_progress=lambda **kw: event_bus.emit(DEPLOY_PROGRESS, **kw),
       emit_error=lambda **kw: event_bus.emit(DEPLOY_ERROR, **kw),
       log_exception=lambda line: ...,
       log_warning=lambda line: ...,
       cancel_func_factory=lambda: (lambda: cancel_deployment(p)),
       frame_video_choice=frame_video_choice,
   )
   ```
   Keep the error messageboxes (`mb.showerror(...)`) in `app.py` — they fire after
   the error emit and depend on GUI state.

   **Actually, wait.** The error handling in the current loop does `emit + mb.showerror +
   return`. If we extract only the parsing, the `return` from inside the loop would need
   to become a different control flow (exception, return code, etc.). The simplest approach:
   have `parse_detection_stdout()` return an enum/string indicating what happened:
   `"complete"`, `"no_images"`, `"no_videos"`, `"no_frames"`, `"unicode_error"`. Then
   `app.py` checks the return value and shows the appropriate messagebox.

6. **Write tests in `tests/test_deploy_subprocess.py`.** Test the extracted functions
   directly:

   ```python
   def test_parse_detection_stdout_basic_progress():
       """Test that tqdm progress lines are parsed correctly."""
       lines = [
           "GPU available: True\n",
           " 25%|██        | 2/8 [00:05<00:15, 0.4it/s]\n",
           " 50%|█████     | 4/8 [00:10<00:10, 0.4it/s]\n",
           "100%|██████████| 8/8 [00:20<00:00, 0.4it/s]\n",
       ]
       collected = []
       def emit_progress(**kwargs):
           collected.append(kwargs)
       def emit_error(**kwargs):
           pass  # Should not be called

       parse_detection_stdout(
           stdout_lines=lines,
           data_type="img",
           emit_progress=emit_progress,
           emit_error=emit_error,
           log_exception=lambda line: None,
           log_warning=lambda line: None,
           cancel_func_factory=lambda: (lambda: None),
           frame_video_choice=None,
       )

       # Verify: 3 progress calls (25%, 50%, 100%)
       assert len(collected) == 3
       assert collected[0]['pct'] == 25.0
       assert collected[0]['cur_it'] == 2
       assert collected[0]['tot_it'] == 8
       assert collected[0]['hware'] == "GPU"
       assert collected[0]['status'] == "running"
   ```

   **Test cases to write (minimum 10):**
   - Basic tqdm progress parsing with correct percentages, cur_it, tot_it
   - GPU detection ("GPU available: True" → hware="GPU", False → "CPU")
   - "No image files found" line triggers emit_error
   - "No videos found" line triggers emit_error
   - "No frames extracted" line triggers emit_error
   - "UnicodeEncodeError:" line triggers emit_error
   - Frame extraction mode: "Extracting frames for folder" → extracting_frames_txt
   - Warning lines logged (but excluded patterns not logged)
   - Exception lines logged
   - Empty stdout (no lines) — function completes without error
   - Classification: status changes on `<EA-status-change>` lines
   - Classification: `<EA>` lines written to smooth output
   - Classification: "n_crops_to_classify is zero" triggers early exit

7. **Verification after extraction:**
   - `grep -c "for line in p.stdout" addaxai/app.py` — should decrease by 2 (the two
     loops moved into deploy.py)
   - `make test` — all tests pass including new ones
   - `make lint` — no new errors in modified files
   - `make typecheck` — passes

**Tests:** `make test` — all pass.

**Commit:** `feat: extract subprocess stdout parsers into addaxai/models/deploy.py + tests`

### Step C4: Extend postprocess pipeline tests with golden output

**What to do:** Add tests to `tests/test_postprocess_pipeline.py` that use the golden
output fixture and the `event_collector` fixture from C1. Test `move_files()` with
data from the golden output JSON.

**File to modify:** `tests/test_postprocess_pipeline.py`

**Existing state:** The file has 2 test classes with 9 tests total:
- `TestMoveFiles` (7 tests): basic move, copy, confidence separation, verified dir,
  empty, nested paths, all confidence buckets
- `TestFileOperationsWithFixtures` (2 tests): fixture images move, multiple fixture images

**Detailed instructions:**

1. **Read the golden output fixture** at `tests/fixtures/golden_output.json`. Understand
   its structure: it should have an `"images"` array where each entry has `"file"`,
   `"detections"` (with `"category"`, `"conf"`, `"bbox"`).

2. **Read the existing tests** in `test_postprocess_pipeline.py` to understand the
   patterns used. Each test creates a temp directory, sets up source files, calls
   `move_files()`, and asserts on the result path and file existence.

3. **Add a new test class `TestGoldenOutputPipeline`** with these tests:

   a. `test_move_files_from_golden_output` — Load golden_output.json, iterate over its
      images, call `move_files()` for each, verify all files end up in correct directories.

   b. `test_golden_output_with_confidence_separation` — Same but with `sep_conf=True`.
      Verify confidence buckets match the detection confidence values in the JSON.

   c. `test_golden_output_missing_source_file` — Call `move_files()` with a file path
      from golden output that doesn't exist on disk. Verify it raises `FileNotFoundError`
      (or the appropriate behavior — read the function to check).

   d. `test_golden_output_empty_detections` — Create an entry with `"detections": []`
      and call `move_files()` with `detection_type="empty"`. Verify the file goes to
      the `empty/` subdirectory.

   e. `test_move_files_with_event_collector` — Use the `event_collector` fixture from
      conftest.py. This test does NOT test `move_files()` directly (it doesn't emit
      events). Instead, emit a `POSTPROCESS_PROGRESS` event manually, then verify the
      `event_collector` captured it. This confirms the fixture works with the postprocess
      event type.

4. **Do NOT try to call the `postprocess()` function from `app.py`** — it depends on
   GUI state (`root`, `state`, tkinter variables). Only test `move_files()` which is
   already extracted and parameterized in `addaxai/processing/postprocess.py`.

**Tests:** `make test` — all pass.

**Commit:** `test: extend postprocess pipeline tests with golden output and event collector`

### Step C5: Push, PR, merge

```bash
git push -u origin phase7/orchestration-tests
gh pr create --repo TeodoroTopa/AddaxAI --base main \
  --title "test: add orchestration and subprocess testing infrastructure" \
  --body "$(cat <<'EOF'
## Summary
- Add conftest.py with mock_app_env and event_collector fixtures
- Extract subprocess stdout parsers (parse_detection_stdout, parse_classification_stdout)
  from app.py into addaxai/models/deploy.py
- Add subprocess parsing tests (10+ tests)
- Extend postprocess pipeline tests with golden output fixtures (5+ tests)

Note: Step C2 (UI event integration tests) was already completed in PR #12
as tests/test_ui_event_wiring.py (23 tests).

## Test plan
- [ ] All unit tests pass (make test)
- [ ] New tests verify event sequences without GUI or real models
- [ ] Lint passes (make lint)
- [ ] Typecheck passes (make typecheck)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
# Wait for CI, then:
gh pr merge <PR_NUMBER> --repo TeodoroTopa/AddaxAI --merge --delete-branch
git checkout main && git pull origin main
```

---

## Ideas for Future Development

### Cloud inference backend
`models/backend.py` defines the `InferenceBackend` protocol. A `CloudBackend`
implementation would upload detection crops (~10KB each) to a hosted classification
endpoint (HuggingFace Inference Endpoints or Replicate). MegaDetector runs locally
(fast, no uploads). Requires headless deployment (event bus migration must be complete).

### Install with no models
Currently the installer bundles large model files. A download-on-demand architecture
would let users install the app (~50MB) and download models later. Requires: atomic
download manager (download to temp, verify, move on success — addresses the RESUME
DOWNLOAD TODO), model registry API, and UI for browsing/downloading models.

### UI framework migration
All business logic is framework-agnostic. Migrating from customtkinter to PySide6 is
a contained effort limited to `addaxai/ui/` once the view protocols are fully
implemented. PySide6 would give native-feeling widgets on macOS and proper MVC.

### REST API write endpoints
`addaxai/api/server.py` has read-only endpoints. Write endpoints (`POST /detect`,
`GET /jobs/{id}`) would let external tools trigger detection headlessly. Requires
headless deployment (event bus migration must be complete).

### Additional languages
The i18n system makes adding languages cheap: create a new JSON file, add the language
index. Portuguese and German would cover the largest remaining user communities.

### HITL improvements
Native annotation UI (eliminating LabelImg dependency), batch-review workflows, and
the 15 sub-items from the app.py TODOs.
