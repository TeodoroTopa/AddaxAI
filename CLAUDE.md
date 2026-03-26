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

## Refactoring History (Phases 1–8)

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

**Phase 7a:** Completed the event bus migration. Wired deploy_tab, postprocess_tab, and
hitl_window event handlers to ProgressWindow. Removed all direct `progress_window.update_values()`
calls from the three orchestrators. Orchestrators now communicate progress exclusively
through `event_bus.emit()`. Extracted widget construction into `build_widgets()` methods.
Added 43 event wiring tests. Net result: ~645 lines of widget code moved out of app.py.

**Phase 7b:** Created `addaxai/orchestration/` package — a complete, headless-capable,
tkinter-free re-implementation of all three orchestrators:
- `context.py` — `DeployConfig`, `ClassifyConfig`, `PostprocessConfig` dataclasses
- `callbacks.py` — `OrchestratorCallbacks` dataclass (replaces all mb.* and root.update calls)
- `stdout_parser.py` — `parse_detection_stdout()`, `parse_classification_stdout()`
- `pipeline.py` — `run_detection()`, `run_classification()`, `run_postprocess()`,
  `_postprocess_inner()`, plus result dataclasses

**Phase 7c:** Wired app.py to the orchestration module. Replaced all 4 orchestrator
function bodies with thin shims (~25 lines each) that read tkinter vars into config
dataclasses and delegate to `pipeline.py`. Deleted the 972-line `postprocess()` function.
Extracted `_build_gui_callbacks()` and `_deploy_cancel_factory()` helpers to eliminate
duplicate OrchestratorCallbacks construction. Net result: app.py reduced from 7,656 to
~6,540 lines. Merged as PR #13.

**Phase 8 (in progress):** Systematic extraction of remaining business logic from app.py.
- **Step 8a.1:** Extracted 4 HITL data functions (`verification_status`,
  `check_if_img_needs_converting`, `fetch_confs_per_class`, `update_json_from_img_list`)
  to `addaxai/hitl/data.py`. Parameterized `var_choose_folder.get()` → `base_folder`,
  replaced `patience_dialog` GUI dependency with `progress_callback` callable.
- **Step 8a.2-3:** Extracted model download logic (`needs_update`, `fetch_manifest`,
  `get_download_info`, `download_model_files`, `download_and_extract_env`) to
  `addaxai/models/download.py`. Parameterized `current_AA_version`, replaced progress
  windows with injected callbacks. App.py retains thin shims for user confirmation
  dialogs and progress window creation.
- **Step 8a.5:** Extracted 3 deploy helpers (`scan_media_presence`, `build_deploy_options`,
  `scan_special_characters`) to `addaxai/orchestration/deploy_helpers.py`. Wired all 3
  call sites in `start_deploy()` — removed ~95 lines of inline logic.
- **Step 8a.6:** Extracted `count_annotations_per_class()` to `hitl/data.py` (from
  `produce_graph()`). Extracted `reclassify_speciesnet_detections()` to
  `addaxai/processing/speciesnet.py` (from `deploy_speciesnet()`).

---

## Current State (as of 2026-03-25)

### Numbers

- **63 Python modules** under `addaxai/` (~15,500 lines total)
- **`addaxai/app.py`**: ~6,100 lines — orchestrators wired, business logic extraction ongoing
- **`addaxai/orchestration/`**: 2,230 lines — headless orchestrators + deploy helpers
- **47 test files** under `tests/` (666 passing, ~10 skipped)
- **3 CI jobs**: unit tests (Python 3.9 + 3.11 with coverage), ruff lint, mypy typecheck

### What's Done

- All pure business logic extracted into `addaxai/` modules (config, models, processing,
  utils, i18n, analysis)
- All global state centralized in `AppState` dataclass
- Full type annotations (Python 3.8 compatible)
- Logging throughout (replaces all print() calls)
- Event bus: orchestrators emit events, UI modules subscribe and forward to ProgressWindow
- Widget construction extracted to `build_widgets()` methods
- View protocols: `DeployView`, `PostprocessView`, `HITLView`, `ResultsView`
- JSON schemas, REST API (read-only), `InferenceBackend` protocol
- `addaxai/orchestration/` — headless pipeline functions, wired to app.py via thin shims
- `addaxai/hitl/data.py` — HITL data processing (verification, JSON update, annotation counting)
- `addaxai/models/download.py` — model/env download logic with callback-based progress
- `addaxai/orchestration/deploy_helpers.py` — media scanning, option building, special char scanning
- `addaxai/processing/speciesnet.py` — SpeciesNet JSON conversion (label map merge, reclassification)

### What Remains in app.py (~6,100 lines)

The orchestrators are wired and business logic extraction is ongoing. What remains is:

1. **`start_deploy()`** — ~600 lines. Validation, model/env download, ProgressWindow
   setup, then calls `deploy_model()`. Option building and media scanning now extracted
   to `deploy_helpers.py`. Still contains ~200 lines of GUI validation dialogs and
   ~130 lines of post-deploy JSON management.
2. **HITL orchestration** — ~900 lines across `open_annotation_windows()` (362 lines),
   `open_hitl_settings_window()` (359 lines), `select_detections()` (290 lines), and
   helpers. Deeply coupled to tkinter widget state.
3. **GUI dialogs** — ~700 lines of dialog windows (`show_download_error_window`,
   `show_model_info`, `show_result_info`, `show_release_info`, `show_donation_popup`).
   Pure GUI construction, no business logic. Can move to `addaxai/ui/dialogs/`.
4. **GUI callbacks** — ~850 lines of toggle/frame/language functions. Pure GUI glue
   coupling multiple tabs. Extracting requires inter-module communication design.
5. **`deploy_speciesnet()`** — ~210 lines. SpeciesNet subprocess orchestration.
   JSON conversion logic extracted to `processing/speciesnet.py`; subprocess management
   and GUI output remain.
6. **Bootstrapping** — ~600 lines of window construction, widget creation, `main()`.

---

## Repository Layout

```
addaxai/
├── app.py                  # ~6,300 lines — GUI shell + thin orchestrator shims
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
│   ├── download.py         # needs_update, fetch_manifest, download_model_files, download_and_extract_env
│   └── registry.py         # fetch_known_models, set_up_unknown_model, environment_needs_downloading
├── orchestration/          # Headless orchestrators (wired to app.py via thin shims)
│   ├── __init__.py
│   ├── context.py          # DeployConfig, ClassifyConfig, PostprocessConfig dataclasses
│   ├── callbacks.py        # OrchestratorCallbacks dataclass
│   ├── deploy_helpers.py   # scan_media_presence, build_deploy_options, scan_special_characters
│   ├── stdout_parser.py    # parse_detection_stdout, parse_classification_stdout
│   └── pipeline.py         # run_detection, run_classification, run_postprocess, _postprocess_inner
├── processing/
│   ├── annotations.py      # Pascal VOC / COCO / YOLO XML conversion
│   ├── export.py           # csv_to_coco
│   ├── postprocess.py      # move_files, format_size
│   └── speciesnet.py       # reclassify_speciesnet_detections
├── analysis/
│   └── plots.py            # fig2img, overlay_logo, calculate_time_span, _produce_plots_extracted
├── i18n/
│   ├── __init__.py         # t("key"), lang_idx(), i18n_set_language()
│   └── en.json, es.json, fr.json
├── hitl/
│   ├── __init__.py
│   └── data.py             # verification_status, check_if_img_needs_converting, update_json_from_img_list, count_annotations_per_class
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
│   └── global_vars.schema.json, model_vars.schema.json, recognition_output.schema.json
├── api/
│   └── server.py           # FastAPI: GET /models, /results/{folder}, /health
└── utils/
    └── files.py, images.py, json_ops.py
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
git checkout -b phase8/<branch-name>
# ... do the work, make commits ...
git push -u origin phase8/<branch-name>
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
Current count: ~666 passing, ~10 skipped (optional deps: cv2, matplotlib, customtkinter).

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

## Phase 8 — Extract Remaining Business Logic from app.py

### Goal

Systematically extract all non-GUI business logic from app.py into focused modules,
leaving app.py as a pure GUI shell. This enables headless operation (CLI, REST API),
UI framework migration, and makes the codebase approachable for open-source contributors.

### Extraction Pattern

Every extraction follows the same recipe:
1. **Parameterize** — replace `var_foo.get()` with an explicit parameter
2. **Inject callbacks** — replace `mb.showerror()` / `root.update()` / progress windows
   with callable parameters (see `OrchestratorCallbacks` pattern)
3. **TDD** — write tests first, implement to make them pass
4. **Leave a thin shim** — app.py retains a 5–30 line wrapper that reads tkinter vars
   and calls the extracted function

### Prioritized Targets

**Immediate (headless-critical):**
- `start_deploy()` decomposition (~730 lines) — extract input validation (~200 lines),
  option building (~100 lines), special-character scanning (~80 lines), and post-deploy
  JSON management (~130 lines) into pure functions. Leaves ~220-line GUI shim.
- `deploy_speciesnet()` (~244 lines) — subprocess + JSON conversion. Mixed GUI/logic.

**Medium (declutter app.py, improve maintainability):**
- GUI dialog windows (~700 lines) — `show_download_error_window`, `show_model_info`,
  `show_result_info`, `show_release_info`, `show_donation_popup`. Pure GUI construction,
  move to `addaxai/ui/dialogs/`.
- HITL windows/data (~900 lines) — `open_annotation_windows`, `open_hitl_settings_window`,
  `select_detections`. Deeply coupled to tkinter widget state; needs `SelectionCriteria`
  dataclass to decouple.

**Later (UI framework readiness):**
- GUI callback functions (~850 lines) — 50+ toggle/frame/language functions. Pure GUI
  glue. Extracting requires inter-module communication design (event bus or widget
  registry). This is the prerequisite for UI framework migration.

---

## Track B: Bug Fixes & Quick Wins (Deferred)

After Phase 7c, address these contained issues:

1. **Windows file-in-use bug** — Investigate `shutil.move` and file handle leaks in
   postprocessing. Likely involves `openpyxl`/`pandas` not closing file handles. Look
   at `_postprocess_inner()` in `pipeline.py` and its callers. Fix with explicit `close()`
   or context managers.

2. **LAT LON 0,0 filter** — In the map creation code (search for `folium` or `HeatMap`
   usage in `plots.py`), skip coordinates at exactly (0.0, 0.0) before plotting.

3. **Completion messagebox** — At the end of `run_detection()` in `pipeline.py`, add a
   summary messagebox showing: images processed, detections found, errors/warnings count.

---

## Ideas for Future Development

### Cloud inference backend
`models/backend.py` defines the `InferenceBackend` protocol. A `CloudBackend`
implementation would upload detection crops (~10KB each) to a hosted classification
endpoint (HuggingFace Inference Endpoints or Replicate). MegaDetector runs locally
(fast, no uploads). Requires Phase 7c complete.

### Install with no models
A download-on-demand architecture would let users install the app (~50MB) and download
models later. Requires: atomic download manager, model registry API, and UI for
browsing/downloading models.

### UI framework migration
All business logic is framework-agnostic. Migrating from customtkinter to PySide6 is
a contained effort limited to `addaxai/ui/` once Phase 7c is complete.

### REST API write endpoints
`addaxai/api/server.py` has read-only endpoints. Write endpoints (`POST /detect`,
`GET /jobs/{id}`) would let external tools trigger detection headlessly. Requires Phase 7c.

### Additional languages
The i18n system makes adding languages cheap: create a new JSON file, add the language
index. Portuguese and German would cover the largest remaining user communities.

### HITL improvements
Native annotation UI (eliminating LabelImg dependency), batch-review workflows, and
the 15 sub-items from the app.py TODOs.
