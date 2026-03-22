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

### Step A9 (Future — after A8 is merged): Extract widget code from app.py

This is the big payoff. After the orchestrators use only the event bus, the widget
construction code for each feature area can be moved out of `app.py` into the shell
UI modules. This is Phase 7b work — plan it after A8 is stable.

High-level approach:
1. Identify all widget construction code in `app.py` related to the deployment tab
   (buttons, labels, frames, grid layout). Move it into `deploy_tab.py`.
2. Repeat for postprocessing, HITL, results viewer.
3. Each extraction is a separate commit with its own smoke test run.
4. Target: `app.py` shrinks from ~8,600 to ~5,000 lines.

---

## Phase 7 Track C — Testing Infrastructure

### Goal

Add subprocess-boundary tests that verify the orchestration logic without needing real
models or a GUI. These tests mock `subprocess.Popen` at the boundary, feed canned stdout,
and assert that the orchestration logic produces correct events and JSON output.

### Prerequisites

- Event bus migration (Track A steps A1-A6) should be complete, so tests can subscribe
  to events instead of needing to inspect GUI widget state.
- If Track A is not yet complete, the tests in this track can still be written to test
  event emissions directly (option a from step A7).

### Branch Setup

```bash
git checkout main && git pull origin main
git checkout -b phase7/orchestration-tests
```

### Step C1: Create a test helper for mocking orchestrator dependencies

**What to do:** Create a conftest fixture that provides a minimal mock environment for
testing orchestrator functions without a GUI.

**Files to create:**
- `tests/conftest.py` (or add to existing if it exists)

**Detailed instructions:**

1. Check if `tests/conftest.py` exists. If so, add to it; if not, create it.

2. Create a pytest fixture called `mock_app_env` that:
   - Creates a temporary directory structure mimicking `AddaxAI_files/` (with
     `models/det/`, `models/cls/`, and a test folder with fixture images)
   - Copies the fixture images from `tests/fixtures/images/` into the temp folder
   - Creates a minimal `global_vars.json` using the valid fixture from
     `tests/fixtures/global_vars_valid.json`
   - Returns a dict with paths: `{"base_path": ..., "image_folder": ..., "json_path": ...}`

3. Create a pytest fixture called `event_collector` that:
   - Subscribes to ALL event types from `addaxai/core/event_types.py`
   - Collects emitted events into a list of `(event_name, kwargs)` tuples
   - Calls `event_bus.clear_all()` in teardown to prevent cross-test leakage
   - Returns the list so tests can assert on it

   Example implementation:
   ```python
   @pytest.fixture
   def event_collector():
       from addaxai.core.events import event_bus
       from addaxai.core import event_types
       collected = []

       def make_handler(name):
           def handler(**kwargs):
               collected.append((name, kwargs))
           return handler

       handlers = {}
       for attr in dir(event_types):
           val = getattr(event_types, attr)
           if isinstance(val, str) and "." in val:
               h = make_handler(val)
               handlers[val] = h
               event_bus.on(val, h)

       yield collected

       for event_name, handler in handlers.items():
           event_bus.off(event_name, handler)
       event_bus.clear_all()
   ```

**Tests:** `make test` — all existing tests still pass.

**Commit:** `test: add conftest fixtures for orchestrator testing (mock_app_env, event_collector)`

### Step C2: Test event bus integration with UI modules

**What to do:** Write tests that verify the shell UI modules correctly receive events
and call the right methods.

**Files to create:**
- `tests/test_ui_event_integration.py`

**Detailed instructions:**

1. Test that `DeployTab` receives `DEPLOY_PROGRESS` events:
   - Create a `DeployTab` with a mock `parent_frame` and mock `start_deploy_callback`
   - Emit `DEPLOY_PROGRESS` event with `pct=50.0, message="Processing 5/10"`
   - Verify `show_progress()` was called (mock it or check side effects)

2. Test that `DeployTab` receives `DEPLOY_ERROR` events:
   - Emit `DEPLOY_ERROR` with `message="Something failed"`
   - Verify `show_error()` was called

3. Test that `DeployTab` receives `DEPLOY_FINISHED` events:
   - Emit `DEPLOY_FINISHED` with `results_path="/some/path.json"`
   - Verify `show_completion()` was called

4. Repeat for `PostprocessTab` with `POSTPROCESS_*` events.

5. Test that after `event_bus.clear_all()`, no events are received.

6. Test that creating two `DeployTab` instances doesn't cause duplicate handling.

7. Target: 10-12 tests.

**Important:** Each test must call `event_bus.clear_all()` in teardown (or use the
`event_collector` fixture) to prevent cross-test state leakage.

**Tests:** `make test` — all pass.

**Commit:** `test: add UI event integration tests for deploy_tab and postprocess_tab`

### Step C3: Test subprocess stdout parsing (deploy_model simulation)

**What to do:** Write tests that simulate MegaDetector subprocess stdout and verify
the orchestration logic parses it correctly. This is the closest thing to an E2E test
without needing real models.

**Files to create:**
- `tests/test_deploy_subprocess.py`

**Detailed instructions:**

1. Read `deploy_model()` in `app.py` (lines 2809-3140). Find where it reads subprocess
   stdout. The key pattern is:
   ```python
   for line in process.stdout:
       # parse progress percentage from line
       # update progress
   ```

2. Create a mock subprocess that yields canned stdout lines matching MegaDetector's
   output format. MegaDetector typically prints lines like:
   ```
   Processing image 1/10...
   Processing image 2/10...
   ```
   Read the actual parsing logic in `deploy_model()` to determine the exact format.

3. Write tests that:
   - Mock `subprocess.Popen` to return canned stdout
   - Verify that progress events are emitted with correct percentages
   - Verify that the final JSON file is referenced in the DEPLOY_FINISHED event
   - Verify that a subprocess error (non-zero return code) triggers DEPLOY_ERROR

4. **Important challenge:** `deploy_model()` references many module-level variables from
   `app.py` (like `var_cls_model`, `root`, `state`, etc.). To test it in isolation, you
   would need to either:
   a. Mock all these dependencies (fragile, many mocks needed)
   b. Extract just the subprocess-parsing loop into a helper function in
      `addaxai/models/deploy.py` and test that helper

   **Option (b) is strongly recommended.** Extract a function like:
   ```python
   def parse_deploy_stdout(stdout_lines, emit_progress):
       """Parse MegaDetector subprocess stdout and emit progress events."""
       for line in stdout_lines:
           # ... parsing logic ...
           emit_progress(pct=percentage, message=msg)
   ```
   This function is pure logic with no GUI deps, easy to test. The caller in `app.py`
   passes `event_bus.emit` as the `emit_progress` callback.

5. If you go with option (b), also extract `parse_classify_stdout()` for the
   classification subprocess.

6. Target: 8-12 tests.

**Tests:** `make test` — all pass.

**Commit:** `feat: extract subprocess stdout parsers + add deploy subprocess tests`

### Step C4: Test postprocess pipeline end-to-end (no GUI)

**What to do:** Extend the existing `test_postprocess_pipeline.py` with tests that
exercise the full postprocess flow using the event bus.

**Files to modify:**
- `tests/test_postprocess_pipeline.py`

**Detailed instructions:**

1. Read the existing tests in `test_postprocess_pipeline.py` to understand what's
   already covered.

2. Add tests that:
   - Use the `event_collector` fixture from C1
   - Call `move_files()` with the golden output JSON and fixture images
   - Verify that files are moved to correct subdirectories
   - Verify the file counts are correct
   - Test edge cases: missing images, empty detection list, corrupt JSON

3. If the `postprocess()` function in `app.py` is too coupled to the GUI to call
   directly, test `move_files()` from `addaxai/processing/postprocess.py` instead
   (which is already extracted and parameterized).

4. Target: 5-8 additional tests.

**Tests:** `make test` — all pass.

**Commit:** `test: extend postprocess pipeline tests with event bus integration`

### Step C5: Push, PR, merge

```bash
git push -u origin phase7/orchestration-tests
gh pr create --repo TeodoroTopa/AddaxAI --base main \
  --title "test: add orchestration and subprocess testing infrastructure" \
  --body "$(cat <<'EOF'
## Summary
- Add conftest fixtures (mock_app_env, event_collector)
- Add UI event integration tests for deploy_tab and postprocess_tab
- Extract subprocess stdout parsers into testable functions
- Add deploy subprocess simulation tests
- Extend postprocess pipeline tests with event bus

## Test plan
- [ ] All unit tests pass (make test)
- [ ] New tests verify event sequences without GUI or real models

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
