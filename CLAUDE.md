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

## Refactoring History (Phases 1–7b)

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

**Critical gap:** Phase 7b built the orchestration module as a parallel implementation but
did NOT wire `app.py` to call it. The original `deploy_model()`, `classify_detections()`,
`postprocess()`, and `start_postprocess()` in `app.py` are unchanged and still run the
app. The new pipeline functions are currently dead code from the app's perspective.
Phase 7c (below) does the wiring.

---

## Current State (as of 2026-03-25)

### Numbers

- **58 Python modules** under `addaxai/` (16,112 lines total)
- **`addaxai/app.py`**: 7,656 lines — still contains the 4 original orchestrator functions
- **`addaxai/orchestration/`**: 2,076 lines — new headless orchestrators (not yet wired)
- **42 test files** under `tests/` (615 tests collected, ~462 passing, ~10 skipped)
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
- `addaxai/orchestration/` package — complete, tested, headless-capable pipeline functions

### What's NOT Done (Phase 7c target)

The 4 orchestrator functions in `app.py` still run the app directly, with tkinter variable
reads, messageboxes, and `root.update()` embedded in their bodies:
- `postprocess()` — 972 lines, called by `start_postprocess()` and `start_deploy()`
- `start_postprocess()` — 146 lines
- `classify_detections()` — 198 lines, called by `deploy_model()`
- `deploy_model()` — 324 lines

Total to remove: ~1,640 lines. After Phase 7c, these become thin shims (~25 lines each).

---

## Repository Layout

```
addaxai/
├── app.py                  # 7,656 lines — GUI shell + 4 orchestrator bodies (being wired)
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
├── orchestration/          # Phase 7b — headless orchestrators (not yet wired to app.py)
│   ├── __init__.py
│   ├── context.py          # DeployConfig, ClassifyConfig, PostprocessConfig dataclasses
│   ├── callbacks.py        # OrchestratorCallbacks dataclass
│   ├── stdout_parser.py    # parse_detection_stdout, parse_classification_stdout
│   └── pipeline.py         # run_detection, run_classification, run_postprocess, _postprocess_inner
├── processing/
│   ├── annotations.py      # Pascal VOC / COCO / YOLO XML conversion
│   ├── export.py           # csv_to_coco
│   └── postprocess.py      # move_files, format_size
├── analysis/
│   └── plots.py            # fig2img, overlay_logo, calculate_time_span, _produce_plots_extracted
├── i18n/
│   ├── __init__.py         # t("key"), lang_idx(), i18n_set_language()
│   └── en.json, es.json, fr.json
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
Current count: ~462 passing, ~10 skipped (optional deps: cv2, matplotlib, customtkinter).

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

## Phase 7c — Wire app.py to the Extracted Orchestrators

### Goal

Replace the bodies of the 4 original orchestrator functions in `app.py` with thin shims
that call the pipeline functions from `addaxai/orchestration/pipeline.py`, then delete the
dead implementations. Target: reduce `app.py` from 7,656 to ~6,100 lines by removing
~1,640 lines of implementation that now live in `pipeline.py`.

### Why This Is Needed

Phase 7b built `addaxai/orchestration/pipeline.py` as a parallel implementation. The four
original functions (`deploy_model`, `classify_detections`, `postprocess`,
`start_postprocess`) in `app.py` are unchanged and still run the app. The new pipeline
functions are dead code until this wiring phase is complete.

### Architecture After Phase 7c

```
app.py (GUI shell)
  start_postprocess() — 40 lines: read tkinter vars → PostprocessConfig + callbacks → call run_postprocess()
  deploy_model()      — 30 lines: read tkinter vars → DeployConfig + callbacks → call run_detection()
  classify_detections()— 30 lines: read tkinter vars → ClassifyConfig + callbacks → call run_classification()
  [postprocess() — DELETED, was 972 lines]

addaxai/orchestration/pipeline.py (no tkinter dependency)
  run_detection()       — builds subprocess, parses stdout, returns DetectionResult
  run_classification()  — builds subprocess, parses stdout, returns ClassificationResult
  run_postprocess()     — validates, calls _postprocess_inner for img/vid, returns PostprocessResult
  _postprocess_inner()  — 680-line body, all GUI replaced by injected callbacks
```

### Critical Rules

- **Read before writing.** Before replacing any function body, read the EXACT current code
  of both the app.py function and the pipeline.py equivalent in their current form. Line
  numbers shift as you edit — do not rely on cached line numbers.
- **Pure mechanical wiring.** Do not change behavior, rename variables, or improve code.
  The shim's job is to translate module-level tkinter state into plain-data arguments.
- **One commit per step.** Run `make test` and `make lint` after each commit.
- **Verify with grep.** After replacing each function body, grep for any remaining callers
  of the old function to confirm all call sites are accounted for.
- **No new tests needed.** The pipeline functions already have full test coverage. The
  validation for Phase 7c is `make test` (existing tests must still pass) plus manual
  smoke test via `make dev`.

### Step C1: Wire `start_postprocess()` → `run_postprocess()`

**What to do:** Replace the 146-line body of `start_postprocess()` in `app.py` with a
~40-line shim.

**Read first:** Read `start_postprocess()` in app.py (search for `def start_postprocess`)
and `run_postprocess()` in `pipeline.py` (search for `def run_postprocess`). Understand
every caller and every module-level variable the old function reads.

**Call sites for `postprocess()`** (the inner function called by `start_postprocess()`):
- `start_postprocess()` at 2 locations — these disappear when the shim calls `run_postprocess()` instead
- `start_deploy()` at 4 locations (simple-mode auto-postprocess) — these remain until Step C2

**Shim structure:**

```python
def start_postprocess():
    logger.debug("EXECUTED: %s", sys._getframe().f_code.co_name)
    from addaxai.orchestration.pipeline import run_postprocess
    from addaxai.orchestration.context import PostprocessConfig
    from addaxai.orchestration.callbacks import OrchestratorCallbacks

    # save settings for next time (stays in app.py — pure GUI/config concern)
    write_global_vars(AddaxAI_files, { ... all the same keys as before ... })

    # read vars needed for ProgressWindow sizing before calling run_postprocess
    src_dir = var_choose_folder.get()
    img_json = os.path.isfile(os.path.join(src_dir, "image_recognition_file.json"))
    vid_json = os.path.isfile(os.path.join(src_dir, "video_recognition_file.json"))

    # build config — every field maps 1:1 to a tkinter var read
    config = PostprocessConfig(
        source_folder=src_dir,
        dest_folder=var_output_dir.get(),
        thresh=var_thresh.get(),
        separate_files=var_separate_files.get(),
        file_placement=var_file_placement.get(),
        sep_conf=var_sep_conf.get(),
        vis=var_vis_files.get(),
        crp=var_crp_files.get(),
        exp=var_exp.get(),
        plt=var_plt.get(),
        exp_format=var_exp_format.get(),
        data_type="img",            # run_postprocess handles both img and vid internally
        vis_blur=var_vis_blur.get(),
        vis_bbox=var_vis_bbox.get(),
        vis_size_idx=t('dpd_vis_size').index(var_vis_size.get()),
        keep_series=var_keep_series.get(),
        keep_series_seconds=var_keep_series_seconds.get(),
        keep_series_species=global_vars.get('var_keep_series_species', []),
        current_version=current_AA_version,
        lang_idx=i18n_lang_idx(),
    )

    callbacks = OrchestratorCallbacks(
        on_error=mb.showerror,
        on_warning=mb.showwarning,
        on_info=mb.showinfo,
        on_confirm=mb.askyesno,
        update_ui=root.update,
        cancel_check=lambda: state.cancel_var,
    )

    # open ProgressWindow only if JSON files exist (run_postprocess will handle the
    # no-JSON error case via callbacks.on_error without needing a window)
    if img_json or vid_json:
        processes = []
        if img_json: processes.append("img_pst")
        if config.plt: processes.append("plt")
        if vid_json: processes.append("vid_pst")
        state.progress_window = ProgressWindow(
            processes=processes, master=root,
            scale_factor=scale_factor, padx=PADX, pady=PADY, green_primary=green_primary)
        state.progress_window.open()

    state.cancel_var = False

    def _cancel():
        state.cancel_var = True

    result = run_postprocess(
        config=config,
        callbacks=callbacks,
        cancel_func=_cancel,
        produce_plots_func=produce_plots,
        base_path=AddaxAI_files,
        cls_model_name=var_cls_model.get(),
    )

    if result.success:
        complete_frame(fth_step)
    if img_json or vid_json:
        state.progress_window.close()
```

**After replacing the body:** Verify `postprocess()` is still called from `start_deploy()`.
Do NOT delete `postprocess()` yet — that happens in Step C2.

**Commit:** `refactor: wire start_postprocess() to call run_postprocess() (Step C1)`

### Step C2: Update `start_deploy()` simple-mode calls, then delete `postprocess()`

**What to do:** After C1, `postprocess()` is only called from `start_deploy()` in simple
mode (4 call sites, around lines 3125-3184). Replace each call with a direct call to
`_postprocess_inner()` from `pipeline.py`. Then delete the `postprocess()` function
entirely (removes 972 lines).

**Read first:** Read each of the 4 `postprocess()` call sites in `start_deploy()` to
understand the exact arguments passed. Read `_postprocess_inner()` in `pipeline.py` to
understand the added parameters.

**What the simple-mode calls need:** The existing calls already pass all positional
arguments. The `_postprocess_inner()` requires additional keyword arguments that the old
`postprocess()` was reading from module scope:
- `vis_blur=var_vis_blur.get()`
- `vis_bbox=var_vis_bbox.get()`
- `vis_size_idx=t('dpd_vis_size').index(var_vis_size.get())`
- `cancel_check=lambda: state.cancel_var`
- `update_ui=root.update`
- `cancel_func=cancel` (or whatever the cancel callable is in that context — read the code)
- `produce_plots_func=produce_plots`
- `on_confirm=mb.askyesno`
- `on_error=mb.showerror`
- `current_version=current_AA_version`
- `lang_idx=i18n_lang_idx()`
- `base_path=AddaxAI_files`
- `cls_model_name=var_cls_model.get()`

**Note:** In simple mode, `keep_series_species` is hardcoded as `[]` in the existing
calls — pass it through as-is.

**After updating all 4 call sites:** Grep app.py for `postprocess(` to confirm no remaining
callers. Then delete the `postprocess()` function (lines ~250–1221).

**Commit:** `refactor: replace postprocess() call sites in start_deploy(), delete postprocess() (Step C2)`

### Step C3: Wire `classify_detections()` → `run_classification()`

**What to do:** Replace the 198-line body of `classify_detections()` with a ~30-line shim.

**Read first:** Read `classify_detections()` in app.py and `run_classification()` in
`pipeline.py`. Note that `classify_detections()` is called only from within `deploy_model()`
(2 call sites). Its signature is `(json_fpath, data_type, simple_mode=False)`.

**Shim structure:**

```python
def classify_detections(json_fpath, data_type, simple_mode=False):
    logger.debug("EXECUTED: %s", sys._getframe().f_code.co_name)
    from addaxai.orchestration.pipeline import run_classification
    from addaxai.orchestration.context import ClassifyConfig
    from addaxai.orchestration.callbacks import OrchestratorCallbacks

    config = ClassifyConfig(
        base_path=AddaxAI_files,
        cls_model_name=var_cls_model.get(),
        disable_gpu=var_disable_GPU.get(),
        cls_detec_thresh=var_cls_detec_thresh.get(),
        cls_class_thresh=var_cls_class_thresh.get(),
        smooth_cls_animal=var_smooth_cls_animal.get(),
        tax_fallback=var_tax_fallback.get(),
        temp_frame_folder="",       # read from state or model_vars as needed
        lang_idx=i18n_lang_idx(),
    )

    callbacks = OrchestratorCallbacks(
        on_error=mb.showerror,
        on_warning=mb.showwarning,
        on_info=mb.showinfo,
        on_confirm=mb.askyesno,
        update_ui=root.update,
        cancel_check=lambda: state.cancel_deploy_model_pressed,
    )

    def _cancel_factory(proc):
        def _cancel():
            cancel_subprocess(proc)
            state.cancel_deploy_model_pressed = True
            state.btn_start_deploy.configure(state=NORMAL)
            state.sim_run_btn.configure(state=NORMAL)
            state.progress_window.close()
        return _cancel

    run_classification(
        config=config,
        callbacks=callbacks,
        json_fpath=json_fpath,
        data_type=data_type,
        cancel_func_factory=_cancel_factory,
        simple_mode=simple_mode,
    )
```

**Important:** Read `ClassifyConfig` in `context.py` to verify every field name matches
exactly. Read `run_classification()` signature to verify argument names and types.

**After replacing the body:** Run `make test`. The existing `test_pipeline.py` tests for
`run_classification()` should still pass. The shim itself has no new tests needed.

**Commit:** `refactor: wire classify_detections() to call run_classification() (Step C3)`

### Step C4: Wire `deploy_model()` → `run_detection()`

**What to do:** Replace the 324-line body of `deploy_model()` with a ~35-line shim.

**Read first:** Read `deploy_model()` in app.py carefully — it currently calls
`classify_detections()` at the end (after detection succeeds). Read `run_detection()` in
`pipeline.py` — it does NOT call classification. The classification call must remain in the
`deploy_model()` shim after the `run_detection()` call.

**Current call chain:**
```
start_deploy() → deploy_model() → [run_detection, then classify_detections]
```
**After C4 the call chain becomes:**
```
start_deploy() → deploy_model() → [run_detection shim + classify_detections shim]
```

**Shim structure:**

```python
def deploy_model(path_to_image_folder, selected_options, data_type, simple_mode=False):
    logger.debug("EXECUTED: %s", sys._getframe().f_code.co_name)
    from addaxai.orchestration.pipeline import run_detection
    from addaxai.orchestration.context import DeployConfig
    from addaxai.orchestration.callbacks import OrchestratorCallbacks

    config = DeployConfig(
        base_path=AddaxAI_files,
        det_model_dir=DET_DIR,
        det_model_name=var_det_model.get(),
        det_model_path=var_det_model_path.get(),
        cls_model_name=var_cls_model.get(),
        disable_gpu=var_disable_GPU.get(),
        use_abs_paths=var_abs_paths.get(),
        source_folder=path_to_image_folder,
        dpd_options_model=state.dpd_options_model,
        lang_idx=i18n_lang_idx(),
    )

    callbacks = OrchestratorCallbacks(
        on_error=mb.showerror,
        on_warning=mb.showwarning,
        on_info=mb.showinfo,
        on_confirm=mb.askyesno,
        update_ui=root.update,
        cancel_check=lambda: state.cancel_deploy_model_pressed,
    )

    def _cancel_factory(proc):
        def _cancel():
            cancel_subprocess(proc)
            state.cancel_deploy_model_pressed = True
            state.btn_start_deploy.configure(state=NORMAL)
            state.sim_run_btn.configure(state=NORMAL)
            state.progress_window.close()
        return _cancel

    result = run_detection(
        config=config,
        callbacks=callbacks,
        data_type=data_type,
        selected_options=selected_options,
        simple_mode=simple_mode,
        cancel_func_factory=_cancel_factory,
        error_log_path=state.model_error_log,
        warning_log_path=state.model_warning_log,
        current_version=current_AA_version,
        smooth_cls_animal=var_smooth_cls_animal.get(),
        warn_smooth_vid=state.warn_smooth_vid,
    )

    # preserve existing behavior: classify after successful detection
    if result.success and var_cls_model.get() != t('none'):
        json_fpath = result.json_path
        classify_detections(json_fpath, data_type, simple_mode=simple_mode)
```

**Critical:** Read `DeployConfig` in `context.py` and `run_detection()` signature in
`pipeline.py` to verify every field name and argument exactly. Do not guess field names.

**After replacing the body:** Run `make test` and `make lint`. Then run `make dev` and do a
manual smoke test to verify the GUI still starts and the deploy button is reachable.

**Commit:** `refactor: wire deploy_model() to call run_detection() (Step C4)`

### Step C5: Push, PR, merge

```bash
git push -u origin phase7/wire-orchestrators
gh pr create --repo TeodoroTopa/AddaxAI --base main \
  --title "refactor: wire app.py orchestrators to pipeline module (Phase 7c)" \
  --body "$(cat <<'EOF'
## Summary
- Wire start_postprocess() → run_postprocess() (removes 146-line body)
- Replace simple-mode postprocess() calls in start_deploy() with _postprocess_inner()
- Delete postprocess() function (removes 972 lines)
- Wire classify_detections() → run_classification() (removes 198-line body)
- Wire deploy_model() → run_detection() (removes 324-line body)
- Net: ~1,640 lines removed from app.py; orchestrators now GUI-free at runtime

## Test plan
- [ ] All unit tests pass (make test)
- [ ] Linter passes (make lint)
- [ ] GUI smoke test passes (make test-smoke)
- [ ] Manual: launch GUI, verify deploy button reachable, postprocess tab loads

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## What Comes After Phase 7c

Phase 7c delivers a wired, GUI-free orchestration layer. What remains in `app.py` (~6,100
lines) is:
1. **~50+ callback functions** — `toggle_cls_frame()`, `update_frame_states()`,
   `change_language()`, `model_options()`, etc. These are pure GUI glue coupling multiple
   tabs. Extracting them requires deciding on inter-module communication first.
2. **`start_deploy()` validation** — the 700-line function bound to the deploy button.
   Validates inputs, downloads models/environments, creates ProgressWindow, calls
   `deploy_model()`. The model download/environment check flow has its own GUI dependencies.
3. **HITL orchestration** — `start_or_continue_hitl()` and helpers are deeply coupled to
   tkinter and warrant their own plan.

**How to decide what to do next:**
- For **cloud inference** or **REST API write endpoints**: the pipeline functions are now
  callable without tkinter. Build a headless CLI runner or REST endpoint using
  `run_detection()` / `run_classification()` / `run_postprocess()` directly.
- For **UI framework migration**: tackle the callback functions, moving them into UI
  modules with an event-based inter-module communication pattern.
- For **production stability**: Track B (bug fixes), starting with the Windows file-in-use
  bug in postprocessing.

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
