"""Config dataclasses for GUI-free orchestrators.

Each dataclass holds only plain Python values (str, bool, float, int, List).
No tkinter variables, widget references, or callables — those go in
addaxai/orchestration/callbacks.py.

Usage in app.py (GUI mode):
    config = DeployConfig(
        base_path=AddaxAI_files,
        det_model_dir=DET_DIR,
        det_model_name=var_det_model.get(),
        ...
    )
    run_detection(config, callbacks)

Usage in headless / REST API mode:
    config = DeployConfig(base_path=..., ...)
    run_detection(config, HeadlessCallbacks())
"""

import dataclasses
from typing import List


@dataclasses.dataclass
class DeployConfig:
    """All settings needed to run MegaDetector detection — no GUI dependencies.

    Attributes:
        base_path:        AddaxAI_files root directory.
        det_model_dir:    Path to the detection models directory (DET_DIR).
        det_model_name:   Name of the selected detection model (from dropdown).
        det_model_path:   Path to custom model weights; empty string if not custom.
        cls_model_name:   Name of the selected classification model; "None" if none.
        disable_gpu:      True to set CUDA_VISIBLE_DEVICES="" before the subprocess.
        use_abs_paths:    True to write absolute paths in the recognition JSON.
        source_folder:    Path to the folder of images/videos to process.
        dpd_options_model: Per-language list of model dropdown option lists.
        lang_idx:         Current language index (0=en, 1=es, 2=fr).
    """

    base_path: str
    det_model_dir: str
    det_model_name: str
    det_model_path: str
    cls_model_name: str
    disable_gpu: bool
    use_abs_paths: bool
    source_folder: str
    dpd_options_model: List[List[str]]
    lang_idx: int


@dataclasses.dataclass
class ClassifyConfig:
    """All settings needed to run species classification — no GUI dependencies.

    Attributes:
        base_path:          AddaxAI_files root directory.
        cls_model_name:     Name of the selected classification model.
        disable_gpu:        True to disable GPU for classification subprocess.
        cls_detec_thresh:   Detection confidence threshold for classification.
        cls_class_thresh:   Classification confidence threshold.
        smooth_cls_animal:  True to average predictions across video frames.
        tax_fallback:       True to use taxonomic fallback when confidence is low.
        temp_frame_folder:  Path to extracted video frames; empty string if N/A.
        lang_idx:           Current language index (0=en, 1=es, 2=fr).
    """

    base_path: str
    cls_model_name: str
    disable_gpu: bool
    cls_detec_thresh: float
    cls_class_thresh: float
    smooth_cls_animal: bool
    tax_fallback: bool
    temp_frame_folder: str
    lang_idx: int


@dataclasses.dataclass
class PostprocessConfig:
    """All settings needed to run postprocessing — no GUI dependencies.

    Attributes:
        source_folder:        Folder containing the recognition JSON(s).
        dest_folder:          Folder where results will be written.
        thresh:               Detection confidence threshold (files below are "empty").
        separate_files:       True to separate files into subdirectories by label.
        file_placement:       1=move original files, 2=copy.
        sep_conf:             True to further separate by confidence tier.
        vis:                  True to draw bounding boxes on image copies.
        crp:                  True to crop detections to separate files.
        exp:                  True to export results to CSV/XLSX/COCO.
        plt:                  True to produce pie/time/map plots.
        exp_format:           Export format string (e.g. "CSV", "XLSX", "COCO").
        data_type:            "img" for images, "vid" for videos.
        vis_blur:             True to blur detections instead of drawing boxes.
        vis_bbox:             True to draw bounding boxes on visualized images.
        vis_size_idx:         Index into the vis-size dropdown (0=small, …).
        keep_series:          True to keep image series together during separation.
        keep_series_seconds:  Time window (seconds) that defines a series.
        keep_series_species:  List of species labels to keep together as a series.
        current_version:      AddaxAI version string written into exports.
        lang_idx:             Current language index (0=en, 1=es, 2=fr).
    """

    source_folder: str
    dest_folder: str
    thresh: float
    separate_files: bool
    file_placement: int
    sep_conf: bool
    vis: bool
    crp: bool
    exp: bool
    plt: bool
    exp_format: str
    data_type: str
    vis_blur: bool
    vis_bbox: bool
    vis_size_idx: int
    keep_series: bool
    keep_series_seconds: float
    keep_series_species: List[str]
    current_version: str
    lang_idx: int
