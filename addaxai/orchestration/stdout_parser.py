"""Subprocess stdout parsers for detection and classification orchestrators.

These functions process line-by-line output from MegaDetector and classification
subprocesses, routing progress updates, errors, warnings, and status changes to
injected callbacks. They have no GUI or tkinter dependencies.

Both functions take an iterable of stdout lines (e.g. from p.stdout) plus a set
of callback functions, and return a result code string.
"""

import re
from typing import Callable, Iterable, Optional

# Pre-compiled tqdm progress bar patterns (used in both parsers).
_RE_TQDM_TIMES = re.compile(r"(\[.*?\])")
_RE_TQDM_BAR = re.compile(r"^[^\/]*[^[^ ]*")
_RE_PERCENTAGE = re.compile(r"\d*%")
_RE_CURRENT_IT = re.compile(r"\d*\/")
_RE_TOTAL_IT = re.compile(r"\/\d*")
_RE_ELAPSED = re.compile(r"(?<=\[)(.*)(?=<)")
_RE_TIME_LEFT = re.compile(r"(?<=<)(.*)(?=,)")
_RE_SPEED = re.compile(r"(?<=,)(.*)(?=])")

# Warning patterns from MegaDetector that are not actionable and should NOT
# be written to the warning log.
_WARNING_EXCLUSIONS = (
    "could not determine MegaDetector version",
    "no metadata for unknown detector version",
    "using user-supplied image size",
    "already exists and will be overwritten",
)


def parse_detection_stdout(
    stdout_lines: Iterable[str],
    data_type: str,
    update_ui: Callable[[], None],
    emit_progress: Callable[..., None],
    emit_error: Callable[..., None],
    log_line: Callable[[str], None],
    log_exception: Callable[[str], None],
    log_warning: Callable[[str], None],
    previous_processed_img: str,
    frame_video_choice: Optional[str],
    cancel_func: Callable[[], None],
) -> str:
    """Process detection subprocess stdout line by line.

    Args:
        stdout_lines:           Iterable of output lines from the subprocess.
        data_type:              "img" or "vid".
        update_ui:              Pump the GUI event loop (no-op in headless mode).
        emit_progress:          Emit a DEPLOY_PROGRESS event with kwargs.
        emit_error:             Emit a DEPLOY_ERROR event with kwargs.
        log_line:               Log each raw stdout line.
        log_exception:          Called with a line containing "Exception:".
        log_warning:            Called with a warning line (after exclusion filter).
        previous_processed_img: Initial "no previously processed image" text,
                                updated to last seen "Processing image" line.
        frame_video_choice:     "video", "frame", or None — forwarded to
                                emit_progress as frame_video_choice kwarg.
        cancel_func:            Function passed to emit_progress as cancel_func.

    Returns:
        "complete"       — loop exhausted without error.
        "no_images"      — subprocess reported no images found.
        "no_videos"      — subprocess reported no videos found.
        "no_frames"      — subprocess reported no frames extracted.
        "unicode_error"  — subprocess reported a UnicodeEncodeError.
    """
    GPU_param = "Unknown"
    extracting_frames_mode = False

    for line in stdout_lines:
        log_line(line.rstrip())

        # catch model errors — emit and return early
        if line.startswith("No image files found"):
            emit_error(message="No image files found", process=f"{data_type}_det")
            return "no_images"
        if line.startswith("No videos found"):
            emit_error(message="No videos found", process=f"{data_type}_det")
            return "no_videos"
        if line.startswith("No frames extracted"):
            emit_error(message="No frames extracted", process=f"{data_type}_det")
            return "no_frames"
        if line.startswith("UnicodeEncodeError:"):
            emit_error(message="UnicodeEncodeError: Unparsable special character in filename",
                       process=f"{data_type}_det",
                       previous_processed_img=previous_processed_img)
            return "unicode_error"

        # track last successfully processed image (for unicode error messages)
        if line.startswith("Processing image "):
            previous_processed_img = line.replace("Processing image ", "")

        # log exceptions and warnings to files
        if "Exception:" in line:
            log_exception(line)
        if "Warning:" in line:
            if not any(excl in line for excl in _WARNING_EXCLUSIONS):
                log_warning(line)

        # frame extraction mode
        if "Extracting frames for folder " in line and data_type == "vid":
            emit_progress(pct=0.0, message="Extracting frames...",
                          process=f"{data_type}_det", status="extracting frames")
            extracting_frames_mode = True
        if extracting_frames_mode:
            if '%' in line[0:4]:
                emit_progress(pct=float(line[:3]), message="Extracting frames...",
                              process=f"{data_type}_det", status="extracting frames",
                              extracting_frames_txt=[f"Extracting frames... {line[:3]}%",
                                                     f"Extrayendo fotogramas... {line[:3]}%"])
        if "Extracted frames for" in line and data_type == "vid":
            extracting_frames_mode = False
        if extracting_frames_mode:
            update_ui()
            continue

        # GPU detection
        if line.startswith("GPU available: False"):
            GPU_param = "CPU"
        elif line.startswith("GPU available: True"):
            GPU_param = "GPU"
        elif '%' in line[0:4]:
            # parse tqdm progress bar
            times = _RE_TQDM_TIMES.search(line)[1]
            progress_bar = _RE_TQDM_BAR.search(line.replace(times, ""))[0]
            percentage = _RE_PERCENTAGE.search(progress_bar)[0][:-1]
            current_im = _RE_CURRENT_IT.search(progress_bar)[0][:-1]
            total_im = _RE_TOTAL_IT.search(progress_bar)[0][1:]
            elapsed_time = _RE_ELAPSED.search(times)[1]
            time_left = _RE_TIME_LEFT.search(times)[1]
            processing_speed = _RE_SPEED.search(times)[1].strip()

            emit_progress(pct=float(percentage),
                          message=f"Processing: {current_im}/{total_im}",
                          process=f"{data_type}_det", status="running",
                          cur_it=int(current_im), tot_it=int(total_im),
                          time_ela=elapsed_time, time_rem=time_left,
                          speed=processing_speed, hware=GPU_param,
                          cancel_func=cancel_func,
                          frame_video_choice=frame_video_choice)

        update_ui()

    # loop exhausted — emit done
    emit_progress(pct=100.0, message="Detection complete",
                  process=f"{data_type}_det", status="done")
    update_ui()
    return "complete"


def parse_classification_stdout(
    stdout_lines: Iterable[str],
    data_type: str,
    update_ui: Callable[[], None],
    emit_progress: Callable[..., None],
    emit_error: Callable[..., None],
    log_line: Callable[[str], None],
    smooth_handler: Callable[[str], None],
    cancel_func: Callable[[], None],
) -> str:
    """Process classification subprocess stdout line by line.

    Args:
        stdout_lines:   Iterable of output lines from the subprocess.
        data_type:      "img" or "vid".
        update_ui:      Pump the GUI event loop (no-op in headless mode).
        emit_progress:  Emit a CLASSIFY_PROGRESS event with kwargs.
        emit_error:     Emit a CLASSIFY_ERROR event with kwargs.
        log_line:       Log each raw stdout line.
        smooth_handler: Called with content extracted from <EA>…<EA> protocol lines.
        cancel_func:    Function passed to emit_progress as cancel_func.

    Returns:
        "complete"   — loop exhausted without error.
        "no_crops"   — subprocess reported zero crops to classify.
    """
    GPU_param = "Unknown"
    status_setting = "running"
    elapsed_time = ""
    processing_speed = ""

    for line in stdout_lines:
        log_line(line.rstrip())

        # early exit if nothing to classify
        if line.startswith("n_crops_to_classify is zero. Nothing to classify."):
            emit_error(message="No animal detections that meet the criteria",
                       process=f"{data_type}_cls")
            return "no_crops"

        # smoothing output lines — extract content between <EA> tags
        if "<EA>" in line:
            smooth_match = re.search(r"<EA>(.+)<EA>", line)
            if smooth_match:
                smooth_output_line = smooth_match.group(1)
                smooth_handler(smooth_output_line)

        # status change from smoothing protocol
        if "<EA-status-change>" in line:
            status_match = re.search(r"<EA-status-change>(.+)<EA-status-change>", line)
            if status_match:
                status_setting = status_match.group(1)

        # GPU detection
        if line.startswith("GPU available: False"):
            GPU_param = "CPU"
        elif line.startswith("GPU available: True"):
            GPU_param = "GPU"
        elif '%' in line[0:4]:
            # parse tqdm progress bar
            times = _RE_TQDM_TIMES.search(line)[1]
            progress_bar = _RE_TQDM_BAR.search(line.replace(times, ""))[0]
            percentage = _RE_PERCENTAGE.search(progress_bar)[0][:-1]
            current_im = _RE_CURRENT_IT.search(progress_bar)[0][:-1]
            total_im = _RE_TOTAL_IT.search(progress_bar)[0][1:]
            elapsed_time = _RE_ELAPSED.search(times)[1]
            time_left = _RE_TIME_LEFT.search(times)[1]
            processing_speed = _RE_SPEED.search(times)[1].strip()

            emit_progress(pct=float(percentage),
                          message=f"Classifying: {current_im}/{total_im}",
                          process=f"{data_type}_cls", status=status_setting,
                          cur_it=int(current_im), tot_it=int(total_im),
                          time_ela=elapsed_time, time_rem=time_left,
                          speed=processing_speed, hware=GPU_param,
                          cancel_func=cancel_func)

        update_ui()

    # loop exhausted — emit done
    emit_progress(pct=100.0, message="Classification complete",
                  process=f"{data_type}_cls", status="done",
                  time_ela=elapsed_time, speed=processing_speed)
    update_ui()
    return "complete"
