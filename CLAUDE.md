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

## Refactoring History (Phases 1–7a)

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
adapter template. Added event bus (`addaxai/core/events.py`, `addaxai/core/event_types.py`)
with dual-write wiring. Added view protocols (`addaxai/ui/protocols.py`) and shell UI
modules. Renamed `AddaxAI_GUI.py` → `addaxai/app.py`. Added read-only REST API
(`addaxai/api/server.py`).

**Phase 7a (Track A complete):** Completed the event bus migration. Wired deploy_tab,
postprocess_tab, and hitl_window event handlers to ProgressWindow. Removed all direct
`progress_window.update_values()` calls from the three orchestrators (`deploy_model()`,
`classify_detections()`, `start_postprocess()`/`postprocess()`). Orchestrators now
communicate progress exclusively through `event_bus.emit()`. Extracted widget construction
code into `DeployTab.build_widgets()`, `PostprocessTab.build_widgets()`, and
`HITLWindow.build_widgets()`. Added 43 event wiring tests (`test_orchestrator_events.py`,
`test_ui_event_wiring.py`). Net result: ~645 lines of widget code moved from app.py to
UI modules; app.py reduced from 8,624 to 8,273 lines.

---

## Current State (as of 2026-03-24)

### Numbers

- **53 Python modules** under `addaxai/` (13,928 lines total)
- **`addaxai/app.py`**: 8,273 lines — still the monolith
- **38 test files** under `tests/` (5,983 lines)
- **471 tests** (462 passing, ~10 skipped for optional deps)
- **3 CI jobs**: unit tests (Python 3.9 + 3.11 with coverage), ruff lint, mypy typecheck

### What's Done

- All pure business logic extracted into `addaxai/` modules (config, models, processing,
  utils, i18n, analysis)
- All global state centralized in `AppState` dataclass
- Full type annotations (Python 3.8 compatible)
- Logging throughout (replaces all print() calls)
- Event bus: orchestrators emit events, UI modules subscribe and forward to ProgressWindow
- Widget construction extracted to `build_widgets()` methods in deploy_tab, postprocess_tab,
  hitl_window
- View protocols: `DeployView`, `PostprocessView`, `HITLView`, `ResultsView`
- JSON schemas, REST API (read-only), `InferenceBackend` protocol

### What's NOT Done

The orchestrator functions (`deploy_model()`, `classify_detections()`, `postprocess()`,
`start_postprocess()`) still live in `app.py` and directly reference tkinter variables,
messageboxes, and `root.update()`. They cannot be imported or tested without importing the
entire GUI. This blocks headless deployment, cloud inference, REST API write endpoints, and
proper orchestrator-level testing. Phase 7b addresses this — see below.

---

## Repository Layout

```
addaxai/
├── app.py                  # 8,273 lines — GUI layout + orchestrators (being extracted)
├── __init__.py
├── py.typed                # PEP 561 marker
├── core/
│   ├── config.py           # load_global_vars / write_global_vars / load_model_vars_for
│   ├── event_types.py      # 16 standard event constants (DEPLOY_*, CLASSIFY_*, etc.)
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
│   ├── protocols.py        # DeployView, PostprocessView, HITLView, ResultsView
│   ├── deploy_tab.py       # build_widgets() + event handlers for deploy/classify
│   ├── postprocess_tab.py  # build_widgets() + event handlers for postprocess
│   ├── hitl_window.py      # build_widgets() + stub HITL methods
│   ├── results_viewer.py   # Stub — results display
│   ├── widgets/            # InfoButton, CancelButton, GreyTopButton, SpeciesSelectionFrame
│   ├── dialogs/            # ProgressWindow, CustomWindow, PatienceWindow, etc.
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
Fast (~30s). Import `addaxai/` modules directly. No tkinter, no conda, no models.
Current count: 462 passing, ~10 skipped (optional deps: cv2, matplotlib, customtkinter).

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

**Lessons from Phase 7a:** Track A steps 1–8 introduced 6 defects. Root causes:
(1) Not reading `ProgressWindow.update_values()` to understand its full parameter contract
before writing handlers. (2) Inventing a lossy translation layer instead of forwarding
kwargs transparently. (3) No grep-based verification that all direct calls were removed.
**To avoid repeating:** always read the target interface before writing code; pass data
through transparently; grep to verify after every extraction step.

---

## app.py TODO Comments (lines 8–54)

The original developer left 47 TODO comments at the top of `app.py`. Key items:

- **BUG:** Windows file-in-use error when moving files during postprocessing + XLSX export
- **CLEAN:** Handle deleted images during processing (skip file, continue)
- **RESUME DOWNLOAD:** Atomic downloads — download to temp, move on success
- **ANNOTATION:** 15 sub-items for HITL/LabelImg workflow improvements
- **LAT LON 0,0:** Filter out 0,0 GPS coordinates from map creation
- **CSV/XLSX:** Add frame number, frame rate columns

The Windows file-in-use bug is the only production blocker.

---

## Phase 7b — Orchestrator Decoupling

### Goal

Extract the orchestrator functions from `app.py` so that detection, classification, and
postprocessing can run without any GUI dependency. After this phase, `app.py` should be a
thin GUI shell that reads tkinter variables, calls GUI-free orchestrator functions, and
handles the results. Target: reduce `app.py` from 8,273 lines to ~4,500 lines.

### Why This Matters

Four stated future goals depend on GUI-free orchestrators:
1. **Cloud inference backend** — needs to call `run_detection()` from a server process
2. **REST API write endpoints** — `POST /detect` needs headless detection
3. **UI framework migration** — PySide6 swap is only contained if orchestrators don't
   import tkinter
4. **Orchestrator-level testing** — currently impossible because importing the orchestrators
   imports all of `app.py`

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  app.py  (GUI shell)                                │
│  - Reads tkinter vars → builds config dataclass     │
│  - Creates callbacks struct (messageboxes, etc.)    │
│  - Calls orchestrator function                      │
│  - Handles result (shows messagebox, updates frame) │
└──────────────┬──────────────────────────────────────┘
               │ plain data + callbacks
               ▼
┌─────────────────────────────────────────────────────┐
│  addaxai/orchestration/                             │
│  pipeline.py  — run_detection, run_classification,  │
│                 run_postprocess                      │
│  context.py   — DeployConfig, ClassifyConfig,       │
│                 PostprocessConfig (dataclasses)      │
│  callbacks.py — DeployCallbacks, ClassifyCallbacks,  │
│                 PostprocessCallbacks (dataclasses)   │
│  stdout_parser.py — parse_detection_stdout,         │
│                     parse_classification_stdout      │
└──────────────┬──────────────────────────────────────┘
               │ event_bus.emit()
               ▼
┌─────────────────────────────────────────────────────┐
│  addaxai/core/events.py  (framework-agnostic)       │
│  UI modules subscribe → forward to ProgressWindow   │
└─────────────────────────────────────────────────────┘
```

### Dependency Audit Summary

The four orchestrator functions in `app.py` have these GUI dependencies that must be
replaced with injected callbacks or config parameters:

| Dependency type | deploy_model | classify_detections | postprocess | start_postprocess |
|---|---|---|---|---|
| `mb.showerror()` | 4 calls | 0 | 1 | 3 |
| `mb.askyesno()` | 1 | 0 | 1 | 2 |
| `mb.showinfo()` | 0 | 1 | 0 | 0 |
| `mb.showwarning()` | 1 | 0 | 0 | 1 |
| `root.update()` | 2 | 3 | 4 | 0 |
| tkinter var `.get()` | 6 vars | 6 vars | 3 vars | 12 vars |
| `state.*` fields | 6 fields | 3 fields | 5 fields | 4 fields |
| Other app.py funcs | 4 | 3 | 4 | 3 |

### Critical Rules (carried forward from Phase 7a)

- **Read before writing.** Before modifying any file, read the exact code you are about
  to change in its current form. Do not rely on line numbers from this document — they
  shift as you edit.
- **Pure mechanical moves.** Do not change behavior, rename variables, add abstractions,
  or "improve" code as you move it. Copy-paste, then change only the minimum needed.
- **One commit per extraction.** Run `make test` and `make lint` after each commit.
- **Verify with grep.** After each extraction, grep `app.py` to confirm the moved code
  is gone and grep the destination file to confirm it arrived.
- **Forward data transparently.** Do not reinterpret, reconstruct, or lossy-compress
  parameters. If the original code passes 10 kwargs, the extracted version passes 10 kwargs.

### Step B1: Extract leaf helper functions

**What to do:** Move helper functions that orchestrators depend on but that have no GUI
dependencies themselves. These are pure functions or simple file-system operations that
should have been extracted in Phase 1 but weren't.

**Functions to move (verify each has no GUI dependency before moving):**

| Function | Current location | Destination | Notes |
|---|---|---|---|
| `extract_label_map_from_model()` | app.py ~line 4984 | `addaxai/models/deploy.py` | Has one `mb.showerror` — replace with raising an exception; let caller show the error |
| `taxon_mapping_csv_present()` | app.py ~line 2590 | `addaxai/models/registry.py` | 2-line function. Parameterize: accept `base_path` and `cls_model_name` instead of reading `AddaxAI_files` and `var_cls_model.get()` |
| `load_model_vars()` wrapper | app.py ~line 5415 | stays in app.py | This is thin GUI glue calling `load_model_vars_for()`. But orchestrators should call `load_model_vars_for()` directly with the model dir as a string parameter |

**For `extract_label_map_from_model()`:** Read the function. It loads a model checkpoint
to read class names. The one `mb.showerror()` call should become a logged error + raised
exception. The calling code in `deploy_model()` already has error handling around it.

**Detailed instructions:**

1. Read each function in its entirety before moving it.
2. Parameterize: replace `AddaxAI_files` with `base_path`, `var_cls_model.get()` with
   `cls_model_name`, etc.
3. Add type annotations.
4. Update all call sites in app.py to pass the required arguments.
5. Run `make test` after each function move.

**Commit:** `refactor: extract leaf helper functions from app.py (extract_label_map, taxon_mapping_csv_present)`

### Step B2: Extract `produce_plots()` into `addaxai/analysis/plots.py`

**What to do:** Move `produce_plots()` (~600 lines, app.py lines ~1365-1962) into
`addaxai/analysis/plots.py`, which already has plotting helpers (`fig2img`,
`overlay_logo`, `calculate_time_span`).

**Why separate step:** This is the single largest self-contained function in app.py. It
is called only from `postprocess()`. Its GUI dependencies are limited to `root.update()`
calls embedded in loops (for UI responsiveness) and event bus emits.

**GUI dependencies to remove:**
- `root.update()` calls (~4) → replace with `update_ui: Callable[[], None]` parameter
- `cancel` function reference → replace with `cancel_check: Callable[[], bool]` parameter
- `event_bus.emit(POSTPROCESS_PROGRESS, ...)` calls → keep as-is (event bus is
  framework-agnostic)
- References to `state.postprocessing_error_log` → pass as parameter
- References to `var_choose_folder.get()` → pass `source_folder` as parameter
- References to `i18n_lang_idx()` and `t()` → pass as parameters

**Detailed instructions:**

1. Read `produce_plots()` in its entirety. Note every reference to module-level variables.
2. Create the function signature with all needed parameters (there will be many — this
   function does a lot). Do NOT try to reduce the parameter count by creating a config
   object yet — just make it work.
3. Move the function body verbatim. Change only module-level references → parameters.
4. In `app.py`, replace the function body with a call to the extracted version, passing
   all arguments from the module scope.
5. Run `make test` and `make lint`.

**Commit:** `refactor: extract produce_plots() into addaxai/analysis/plots.py`

### Step B3: Create config dataclasses

**What to do:** Create `addaxai/orchestration/context.py` with plain dataclasses that
replace tkinter variable reads. These are the "shape" of the data each orchestrator needs.

**File to create:** `addaxai/orchestration/__init__.py` and `addaxai/orchestration/context.py`

**Dataclasses to define:**

```python
@dataclass
class DeployConfig:
    """All settings needed to run detection, with no GUI dependencies."""
    base_path: str                    # AddaxAI_files directory
    det_model_dir: str                # path to detection models
    det_model_name: str               # selected detection model name
    det_model_path: str               # path for custom models (empty if not custom)
    cls_model_name: str               # "None" if no classification
    disable_gpu: bool
    use_abs_paths: bool
    source_folder: str                # the folder being processed
    dpd_options_model: List[List[str]]  # model dropdown options (per language)
    lang_idx: int                     # current language index (0=en, 1=es, 2=fr)

@dataclass
class ClassifyConfig:
    """All settings needed to run classification."""
    base_path: str
    cls_model_name: str
    disable_gpu: bool
    cls_detec_thresh: float
    cls_class_thresh: float
    smooth_cls_animal: bool
    tax_fallback: bool
    temp_frame_folder: str
    lang_idx: int

@dataclass
class PostprocessConfig:
    """All settings needed to run postprocessing."""
    source_folder: str
    dest_folder: str
    thresh: float
    separate_files: bool
    file_placement: int               # 1=move, 2=copy
    sep_conf: bool
    vis: bool
    crp: bool
    exp: bool
    plt: bool
    exp_format: str
    data_type: str                    # "img" or "vid"
    vis_blur: bool
    vis_bbox: bool
    vis_size_idx: int
    keep_series: bool
    keep_series_seconds: float
    keep_series_species: List[str]
    current_version: str
    lang_idx: int
```

**Important:** These dataclasses should only hold plain Python values (str, bool, float,
int, List). No tkinter vars, no widget references, no callables. Callables go in the
callbacks dataclass (Step B4).

**Tests:** Write tests in `tests/test_orchestration_context.py` that verify dataclass
instantiation, default values, and that none of the fields are tkinter types.

**Commit:** `feat: add orchestration config dataclasses (DeployConfig, ClassifyConfig, PostprocessConfig)`

### Step B4: Create callback dataclasses

**What to do:** Create `addaxai/orchestration/callbacks.py` with dataclasses that hold
the injected callback functions. These replace all direct GUI calls (messageboxes,
root.update, cancel functions).

**File to create:** `addaxai/orchestration/callbacks.py`

```python
@dataclass
class OrchestratorCallbacks:
    """Injected callbacks for GUI interaction during orchestration.

    In GUI mode: these call mb.showerror(), root.update(), etc.
    In headless mode: these log errors, no-op for UI updates, etc.
    """
    on_error: Callable[[str, str], None]          # (title, message) → show error
    on_warning: Callable[[str, str], None]         # (title, message) → show warning
    on_info: Callable[[str, str], None]            # (title, message) → show info
    on_confirm: Callable[[str, str], bool]         # (title, message) → ask yes/no
    update_ui: Callable[[], None]                  # pump event loop (no-op headless)
    cancel_check: Callable[[], bool]               # check if user requested cancel
```

**Why a single shared callbacks class instead of per-orchestrator:** All three orchestrators
use the same messagebox patterns. A single class with clear method names is simpler than
three similar-but-slightly-different classes.

**Tests:** Write tests verifying that all fields are callable and that a "headless"
instance (logging callbacks) works without importing tkinter.

**Commit:** `feat: add OrchestratorCallbacks dataclass for GUI-free orchestration`

### Step B5: Extract subprocess stdout parsing

**What to do:** Extract the `for line in p.stdout:` parsing loops from `deploy_model()`
and `classify_detections()` into pure functions.

**File to create:** `addaxai/orchestration/stdout_parser.py`

**Functions to create:**

1. `parse_detection_stdout(stdout_lines, data_type, emit_progress, emit_error,
   log_exception, log_warning, cancel_func_factory, frame_video_choice)`
   - Returns a result indicating outcome: `"complete"`, `"no_images"`, `"no_videos"`,
     `"no_frames"`, `"unicode_error"`
   - The `emit_progress` callback must pass through the EXACT SAME kwargs that the
     current `event_bus.emit(DEPLOY_PROGRESS, ...)` calls use. Read each existing emit
     call and replicate kwargs exactly.

2. `parse_classification_stdout(stdout_lines, data_type, emit_progress, emit_error,
   log_exception, log_warning, cancel_func_factory)`
   - Handles `<EA>` smoothing lines, `<EA-status-change>` lines, tqdm progress
   - Returns `"complete"`, `"no_crops"`, etc.

**How to find the parsing code:** In `app.py`, search for `for line in p.stdout:`. There
are two instances — one in `deploy_model()` (detection) and one in `classify_detections()`
(classification). The detection parser handles tqdm progress bars, GPU detection, frame
extraction mode, error/warning lines, and UnicodeEncodeError detection. The classification
parser additionally handles `<EA>` and `<EA-status-change>` protocol lines.

**The tqdm line format is:** `" 50%|████ | 5/10 [00:05<00:05, 1.0it/s]"`
The existing regex parsing extracts: percentage, current_im, total_im, elapsed_time,
time_left, processing_speed.

**Tests to write** (`tests/test_stdout_parser.py`, minimum 12):
- Basic tqdm progress parsing with correct percentages, cur_it, tot_it
- GPU detection (`"GPU available: True"` → hware="GPU", `False` → "CPU")
- `"No image files found"` triggers error result
- `"No videos found"` triggers error result
- `"No frames extracted"` triggers error result
- `"UnicodeEncodeError:"` triggers error result
- Frame extraction mode: `"Extracting frames for folder"` → extracting_frames_txt set
- Warning lines logged (but 4 exclusion patterns not logged)
- Exception lines logged
- Empty stdout — function completes without error, returns "complete"
- Classification: `<EA-status-change>` updates status
- Classification: `"n_crops_to_classify is zero"` → "no_crops" result
- Classification: `<EA>` lines passed to smoothing handler

**Commit:** `feat: extract subprocess stdout parsers into addaxai/orchestration/stdout_parser.py`

### Step B6: Extract `deploy_model()` into `addaxai/orchestration/pipeline.py`

**What to do:** Create `run_detection()` in `pipeline.py` containing the orchestration
logic currently in `deploy_model()`. The function takes `DeployConfig` +
`OrchestratorCallbacks` and returns a result object.

**File to create:** `addaxai/orchestration/pipeline.py`

**The extracted function should:**
1. Accept `DeployConfig`, `OrchestratorCallbacks`, and any remaining state fields
   (cancel flag, error/warning logs, subprocess output holder) that don't fit in config
2. Build the subprocess command (already partially in `addaxai/models/deploy.py`)
3. Launch the subprocess and call `parse_detection_stdout()` from Step B5
4. Handle the parse result (but NOT show messageboxes — return the result)
5. Emit events via the event bus
6. Return a `DetectionResult` dataclass with: `success: bool`, `json_path: Optional[str]`,
   `error_code: Optional[str]`, `error_message: Optional[str]`

**What stays in `app.py`:**
- Reading tkinter vars to build `DeployConfig`
- Creating `OrchestratorCallbacks` pointing to `mb.showerror`, `root.update`, etc.
- Calling `run_detection(config, callbacks)`
- Checking the result and showing messageboxes / updating frames accordingly
- The `classify_detections()` call at the end (becomes: check result, then call
  `run_classification()`)

**Detailed instructions:**

1. Read `deploy_model()` in full (lines ~2806-3127). Note every module-level reference.
2. Read `parse_detection_stdout()` from Step B5 — the parser handles the subprocess loop.
3. The function structure after extraction:
   ```python
   def run_detection(
       config: DeployConfig,
       callbacks: OrchestratorCallbacks,
       cancel_flag: ???,       # needs to be a mutable reference
       error_log_path: str,
       warning_log_path: str,
   ) -> DetectionResult:
   ```
4. **The cancel flag problem:** `deploy_model()` checks
   `state.cancel_deploy_model_pressed` which is set by the cancel button callback.
   The extracted function needs a way to check this. Options:
   - Pass `cancel_check: Callable[[], bool]` (already in OrchestratorCallbacks)
   - Pass a mutable container (e.g., `[False]` list)
   - Use `callbacks.cancel_check()` which reads the flag from wherever the caller stores it

   Use `callbacks.cancel_check()` — it's already in the callbacks struct.

5. **The smoothing warning:** `deploy_model()` shows `mb.askyesno()` about video
   smoothing at the top. Use `callbacks.on_confirm()`. If it returns False, return
   early with a cancelled result.

6. Move the code. Run `make test`.

**Known bug to fix during extraction:** Line ~2974 references `subprocess_output` as a
bare name (should be `state.subprocess_output`). Fix this during the move — document in
the commit message.

**Tests:** Write tests in `tests/test_pipeline.py` that call `run_detection()` with mocked
subprocess (`unittest.mock.patch('subprocess.Popen')`), a `DeployConfig` with test values,
and headless `OrchestratorCallbacks`. Verify correct events are emitted and correct result
is returned for success and error scenarios.

**Commit:** `refactor: extract deploy_model() into addaxai/orchestration/pipeline.run_detection()`

### Step B7: Extract `classify_detections()` into `pipeline.py`

**What to do:** Same pattern as B6 for classification. Create `run_classification()`.

**The extracted function should:**
1. Accept `ClassifyConfig`, `OrchestratorCallbacks`, cancel flag
2. Build the classification subprocess command
3. Launch subprocess and call `parse_classification_stdout()` from Step B5
4. Return `ClassificationResult` with: `success`, `json_path`, `error_code`,
   `error_message`, `n_crops_classified`

**What stays in `app.py`:** Reading tkinter vars → `ClassifyConfig`, creating callbacks,
calling `run_classification()`, handling result.

**Commit:** `refactor: extract classify_detections() into pipeline.run_classification()`

### Step B8: Extract `postprocess()` and `start_postprocess()` into `pipeline.py`

**What to do:** Create `run_postprocess()`. This is the hardest extraction because
`postprocess()` is 972 lines with 4 `root.update()` calls, 1 `mb.askyesno()`, and
references to `cancel`, `state`, and several module-level functions.

**Approach:** Extract in two sub-steps:

**B8a:** First extract `postprocess()` (the inner processing function). It takes many
parameters already — add the remaining ones (`vis_blur`, `vis_bbox`, `vis_size_idx`,
`cancel_check`, `update_ui`, `source_folder`) to make it GUI-free. Move it to
`addaxai/orchestration/pipeline.py`. This is the big move (~972 lines).

**B8b:** Then extract `start_postprocess()` — the wrapper that reads tkinter vars, creates
ProgressWindow, calls postprocess twice (images + videos), and calls `complete_frame()`.
The GUI-free version (`run_postprocess()`) takes `PostprocessConfig` +
`OrchestratorCallbacks` and calls the extracted `postprocess()` internally. It returns
`PostprocessResult`. The ProgressWindow creation and `complete_frame()` call stay in
app.py — they are pure GUI concerns.

**Commit (B8a):** `refactor: extract postprocess() into pipeline module`
**Commit (B8b):** `refactor: extract start_postprocess() wrapper into run_postprocess()`

### Step B9: Push, PR, merge

```bash
git push -u origin phase7/orchestrator-decoupling
gh pr create --repo TeodoroTopa/AddaxAI --base main \
  --title "refactor: decouple orchestrators from GUI — enable headless detection" \
  --body "$(cat <<'EOF'
## Summary
- Create addaxai/orchestration/ package with config dataclasses, callbacks, stdout
  parser, and pipeline functions
- Extract deploy_model() → run_detection()
- Extract classify_detections() → run_classification()
- Extract postprocess()/start_postprocess() → run_postprocess()
- app.py reduced from ~8,273 to ~4,500 lines
- All orchestrators callable without tkinter

## Test plan
- [ ] All unit tests pass (make test)
- [ ] New pipeline tests verify headless orchestration
- [ ] GUI smoke test passes (make test-smoke)
- [ ] Manual test: full detection + classification + postprocess pipeline

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## What Comes After Phase 7b

Phase 7b delivers GUI-free orchestrators but does NOT address:

1. **~50+ callback functions in app.py** — functions like `toggle_cls_frame()`,
   `update_frame_states()`, `change_language()`, `model_options()` are pure GUI glue.
   They reference widgets through `deploy_view.cls_frame`, `hitl_view.btn_hitl_main`,
   etc. These could be moved into the respective UI modules, but the coupling between
   them (e.g., `update_frame_states()` touches frames from multiple tabs) makes this
   harder than it looks. Extracting these requires deciding on a clean inter-module
   communication pattern first.

2. **`start_deploy()` validation function** — this is the function bound to the deploy
   button. It validates inputs, downloads models/environments if needed, creates the
   ProgressWindow, and calls `run_detection()`. The model download/environment check
   flow has its own GUI dependencies (patience window, confirmation dialogs). This is
   a separate extraction effort.

3. **HITL orchestration** — `start_or_continue_hitl()` and its helpers
   (`open_annotation_windows()`, `open_hitl_settings_window()`, etc.) are deeply coupled
   to tkinter. These should be extracted but the HITL workflow is complex enough to
   warrant its own plan.

**How to decide what to do next:** After Phase 7b, evaluate which future goal you want
to pursue first:
- If **cloud inference** or **REST API write endpoints**: build the headless runner and
  API endpoint using the new `run_detection()`/`run_classification()` functions. The
  remaining callback functions in app.py don't block this.
- If **UI framework migration**: tackle the callback functions next, moving them into
  UI modules with a clear event-based communication pattern.
- If **production stability**: tackle Track B (bug fixes), starting with the Windows
  file-in-use bug.

---

## Track B: Bug Fixes & Quick Wins (Deferred)

After the main extraction work is done, address these contained issues:

1. **Windows file-in-use bug** — Investigate `shutil.move` and file handle leaks in
   postprocessing. Likely involves `openpyxl`/`pandas` not closing file handles. Look
   at the `postprocess()` function and its callers. Fix with explicit `close()` or
   context managers.

2. **LAT LON 0,0 filter** — In the map creation code (search for `folium` or `HeatMap`
   usage in `app.py`), skip coordinates at exactly (0.0, 0.0) before plotting.

3. **Completion messagebox** — At the end of `deploy_model()` (near line 3127), add a
   summary messagebox showing: images processed, detections found, errors/warnings count.

These are all small, contained fixes that don't require architectural changes.

---

## Ideas for Future Development

### Cloud inference backend
`models/backend.py` defines the `InferenceBackend` protocol. A `CloudBackend`
implementation would upload detection crops (~10KB each) to a hosted classification
endpoint (HuggingFace Inference Endpoints or Replicate). MegaDetector runs locally
(fast, no uploads). Requires headless deployment (Phase 7b).

### Install with no models
A download-on-demand architecture would let users install the app (~50MB) and download
models later. Requires: atomic download manager, model registry API, and UI for
browsing/downloading models.

### UI framework migration
All business logic is framework-agnostic. Migrating from customtkinter to PySide6 is
a contained effort limited to `addaxai/ui/` once the orchestrators are fully decoupled.

### REST API write endpoints
`addaxai/api/server.py` has read-only endpoints. Write endpoints (`POST /detect`,
`GET /jobs/{id}`) would let external tools trigger detection headlessly. Requires
headless deployment (Phase 7b).

### Additional languages
The i18n system makes adding languages cheap: create a new JSON file, add the language
index. Portuguese and German would cover the largest remaining user communities.

### HITL improvements
Native annotation UI (eliminating LabelImg dependency), batch-review workflows, and
the 15 sub-items from the app.py TODOs.
