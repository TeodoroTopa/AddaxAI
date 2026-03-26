"""Pure-logic helpers extracted from start_deploy() — no GUI dependencies.

Functions:
  scan_media_presence   – detect images/videos in a folder tree
  build_deploy_options  – assemble CLI option lists for detection subprocess
  scan_special_characters – find files with special characters in paths
"""

import os
from typing import Dict, List, Optional, Tuple

from addaxai.utils.files import contains_special_characters

# Default media extensions matching the cameratraps constants.
# Kept here so unit tests work without the cameratraps package.
_IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff')
_VID_EXTENSIONS = ('.mp4', '.avi', '.mpeg', '.mpg', '.wmv', '.mov', '.mkv', '.flv', '.webm')

# Extensions checked during special-character scanning (matches app.py inline list).
_SCAN_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mpeg', '.mpg')


def scan_media_presence(
    folder: str,
    check_img: bool,
    check_vid: bool,
    exclude_subs: bool,
    img_extensions: Optional[Tuple[str, ...]] = None,
    vid_extensions: Optional[Tuple[str, ...]] = None,
) -> Tuple[bool, bool]:
    """Detect whether *folder* contains image and/or video files.

    Args:
        folder:         Root directory to scan.
        check_img:      Whether to look for image files.
        check_vid:      Whether to look for video files.
        exclude_subs:   If True, only scan the top-level directory.
        img_extensions: Tuple of image suffixes (default: common camera-trap formats).
        vid_extensions: Tuple of video suffixes (default: common camera-trap formats).

    Returns:
        ``(img_present, vid_present)`` booleans.
    """
    if img_extensions is None:
        img_extensions = _IMG_EXTENSIONS
    if vid_extensions is None:
        vid_extensions = _VID_EXTENSIONS

    img_present = False
    vid_present = False

    if exclude_subs:
        for f in os.listdir(folder):
            if check_img and f.lower().endswith(img_extensions):
                img_present = True
            if check_vid and f.lower().endswith(vid_extensions):
                vid_present = True
            if (img_present and vid_present) or \
                (img_present and not check_vid) or \
                    (vid_present and not check_img) or \
                        (not check_img and not check_vid):
                break
    else:
        for _root, _, files in os.walk(folder):
            for file in files:
                if check_img and file.lower().endswith(img_extensions):
                    img_present = True
                if check_vid and file.lower().endswith(vid_extensions):
                    vid_present = True
            if (img_present and vid_present) or \
                (img_present and not check_vid) or \
                    (vid_present and not check_img) or \
                        (not check_img and not check_vid):
                break

    return img_present, vid_present


def build_deploy_options(
    simple_mode: bool,
    exclude_subs: bool = False,
    use_checkpoints: bool = False,
    checkpoint_freq: str = "",
    cont_checkpoint: bool = False,
    checkpoint_path: str = "",
    custom_img_size: str = "",
    not_all_frames: bool = False,
    nth_frame: str = "1",
    timelapse_mode: bool = False,
    temp_frame_folder: str = "",
) -> Tuple[List[str], List[str]]:
    """Build CLI option lists for the image and video detection subprocesses.

    Returns:
        ``(img_options, vid_options)`` lists of CLI flags/arguments.
    """
    img_opts: List[str] = ["--output_relative_filenames"]
    vid_opts: List[str] = ["--json_confidence_threshold=0.01"]

    if timelapse_mode:
        vid_opts.append("--include_all_processed_frames")

    if temp_frame_folder:
        vid_opts.append("--frame_folder=" + temp_frame_folder)
        vid_opts.append("--keep_extracted_frames")

    if simple_mode:
        img_opts.append("--recursive")
        vid_opts.append("--recursive")
        vid_opts.append("--time_sample=1")
    else:
        if not exclude_subs:
            img_opts.append("--recursive")
            vid_opts.append("--recursive")
        if use_checkpoints:
            img_opts.append("--checkpoint_frequency=" + checkpoint_freq)
        if cont_checkpoint and checkpoint_path:
            img_opts.append("--resume_from_checkpoint=" + checkpoint_path)
        if custom_img_size:
            img_opts.append("--image_size=" + custom_img_size)
        if not_all_frames:
            vid_opts.append("--time_sample=" + nth_frame)

    return img_opts, vid_opts


def scan_special_characters(
    folder: str,
    extensions: Optional[Tuple[str, ...]] = None,
) -> Dict:
    """Walk *folder* and catalogue files whose paths contain special characters.

    Args:
        folder:     Root directory to scan.
        extensions: File suffixes to check (default: common image/video formats).

    Returns:
        Dict with keys ``"total_files"`` (int) and ``"paths"`` (dict mapping each
        faulty path component to ``[count, offending_char]``).
    """
    if extensions is None:
        extensions = _SCAN_EXTENSIONS

    isolated: Dict = {}
    total_files = 0

    for main_dir, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(main_dir, file)
            if os.path.splitext(file_path)[1].lower() not in extensions:
                continue
            has_special, char = contains_special_characters(file_path)
            if not has_special:
                continue
            drive, rest_of_path = os.path.splitdrive(file_path)
            path_components = rest_of_path.split(os.sep)
            isolated_path = drive
            for component in path_components:
                isolated_path = os.path.join(isolated_path, component)
                if contains_special_characters(component)[0]:
                    total_files += 1
                    if isolated_path in isolated:
                        isolated[isolated_path][0] += 1
                    else:
                        isolated[isolated_path] = [1, char]

    return {"total_files": total_files, "paths": isolated}
