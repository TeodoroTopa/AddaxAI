"""GUI-free orchestrator pipeline functions for AddaxAI.

Each function in this module corresponds to one phase of the processing
pipeline: detection, classification, and postprocessing.  They accept
plain-data config objects and injected callback structs, so they can be
called from a GUI, a REST API handler, or a headless command-line runner
without any tkinter dependency.

NOTE: addaxai.i18n must be initialised (via addaxai.i18n.init()) before
calling these functions, because error-message titles are looked up via
t().  In headless mode the caller should call init(lang_idx) once at
startup with the desired language index.
"""

import csv
import dataclasses
import datetime
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from subprocess import Popen
from typing import Any, Callable, Dict, List, Optional

from addaxai.core.config import load_model_vars_for
from addaxai.core.event_types import (
    CLASSIFY_ERROR,
    CLASSIFY_FINISHED,
    CLASSIFY_PROGRESS,
    CLASSIFY_STARTED,
    DEPLOY_CANCELLED,
    DEPLOY_FINISHED,
    DEPLOY_PROGRESS,
    DEPLOY_STARTED,
    DEPLOY_ERROR,
    POSTPROCESS_ERROR,
    POSTPROCESS_FINISHED,
    POSTPROCESS_PROGRESS,
    POSTPROCESS_STARTED,
)
from addaxai.core.events import event_bus
from addaxai.core.platform import get_python_interpreter
from addaxai.i18n import t
from addaxai.models.deploy import (
    extract_label_map_from_model,
    imitate_object_detection_for_full_image_classifier,
    switch_yolov5_version,
)
from addaxai.models.registry import taxon_mapping_csv_present
from addaxai.orchestration.callbacks import OrchestratorCallbacks
from addaxai.orchestration.context import ClassifyConfig, DeployConfig
from addaxai.orchestration.stdout_parser import (
    parse_classification_stdout,
    parse_detection_stdout,
)
from addaxai.processing.export import csv_to_coco, format_datetime, generate_unique_id
from addaxai.processing.postprocess import move_files
from addaxai.utils.images import blur_box, build_image_timestamp_index, find_series_images
from addaxai.utils.json_ops import (
    check_json_paths,
    fetch_label_map_from_json,
    get_hitl_var_in_json,
    make_json_absolute,
    make_json_relative,
)
from addaxai.utils.json_ops import append_to_json

logger = logging.getLogger(__name__)


# ============================================================================
# Result dataclasses
# ============================================================================

@dataclasses.dataclass
class DetectionResult:
    """Result returned by run_detection().

    Attributes:
        success:       True if detection completed without error or cancellation.
        json_path:     Absolute path to the recognition JSON file, or None.
        error_code:    Short machine-readable code: "no_images", "no_videos",
                       "no_frames", "unicode_error", "cancelled",
                       "cancelled_smooth_vid", or None on success.
        error_message: Human-readable description of the error, or None.
    """
    success: bool
    json_path: Optional[str]
    error_code: Optional[str]
    error_message: Optional[str]


# ============================================================================
# run_detection
# ============================================================================

def run_detection(
    config: DeployConfig,
    callbacks: OrchestratorCallbacks,
    data_type: str,
    selected_options: List[str],
    simple_mode: bool,
    cancel_func_factory: Callable[[Any], Callable[[], None]],
    error_log_path: str,
    warning_log_path: str,
    current_version: str,
    smooth_cls_animal: bool = True,
    warn_smooth_vid: bool = False,
) -> DetectionResult:
    """Run MegaDetector detection on a folder of images or videos.

    Args:
        config:               Deployment configuration (paths, model names, flags).
        callbacks:            Injected callbacks for UI interaction (errors,
                              warnings, confirmations, UI pump).
        data_type:            "img" for images, "vid" for videos.
        selected_options:     Extra CLI flags to pass to the detection script.
                              May be mutated if a custom model is used.
        simple_mode:          True when running in simplified (beginner) mode.
        cancel_func_factory:  Called with the Popen subprocess instance after
                              launch; must return a zero-argument callable that
                              cancels the subprocess.  The returned callable is
                              forwarded as cancel_func to emit_progress.
        error_log_path:       File path for appending Exception lines from stdout.
        warning_log_path:     File path for appending Warning lines from stdout.
        current_version:      AddaxAI version string written into result JSON.
        smooth_cls_animal:    True if classification smoothing is enabled.
                              Used only in the pre-flight smoothing warning check.
        warn_smooth_vid:      True if the user should be warned about analysing
                              videos without smoothing before proceeding.

    Returns:
        DetectionResult — check .success and .error_code for outcome.
    """
    logger.debug("EXECUTED: run_detection")

    # emit started event
    event_bus.emit(DEPLOY_STARTED, process=f"{data_type}_det")

    # pre-flight: warn user if analysing videos without smoothing
    if (config.cls_model_name != t('none')) and \
            (not smooth_cls_animal) and \
            data_type == 'vid' and \
            (not simple_mode) and \
            warn_smooth_vid:
        if not callbacks.on_confirm(
            t('information'),
            ["You are about to analyze videos without smoothing the confidence scores. "
             "Typically, a video may contain many frames of the same animal, increasing "
             "the likelihood that at least one of the labels could be a false prediction. "
             f"With '{t('lbl_smooth_cls_animal')}' enabled, all predictions from a single "
             "video will be averaged, resulting in only one label per video. Do you wish to"
             " continue without smoothing?\n\nPress 'No' to go back.",
             "Estás a punto de analizar videos sin suavizado habilitado. Normalmente, un "
             "video puede contener muchos cuadros del mismo animal, lo que aumenta la "
             "probabilidad de que al menos una de las etiquetas pueda ser una predicción "
             f"falsa. Con '{t('lbl_smooth_cls_animal')}' habilitado, todas las predicciones "
             "de un solo video se promediarán, lo que resultará en una sola etiqueta por "
             "video. ¿Deseas continuar sin suavizado habilitado?\n\nPresiona 'No' para "
             "regresar.",
             "Vous êtes sur le point d'analyser des vidéos sans lisser les scores de "
             "confiance. Typiquement, un vidéo peut contenir plusieurs images d'un même "
             "animal, ce qui augmente les chances qu'au moins un des labels puisse être une "
             f"fausse prédiction. Avec '{t('lbl_smooth_cls_animal')}' activé, toutes les "
             "prédictions d'un seul vidéo seront moyennées, résultant en un seul label par "
             "vidéo. Souhaitez-vous continuer sans lissage?\n\nAppuyer sur 'Non' pour "
             "revenir en arrière."
             ][config.lang_idx]
        ):
            return DetectionResult(
                success=False,
                json_path=None,
                error_code="cancelled_smooth_vid",
                error_message="User cancelled due to smoothing warning",
            )

    # display loading — event bus updates the progress window
    event_bus.emit(DEPLOY_PROGRESS, pct=0.0, message="Loading detection model",
                   process=f"{data_type}_det", status="load")

    # prepare common paths
    chosen_folder = str(Path(config.source_folder))
    run_detector_batch_py = os.path.join(
        config.base_path, "cameratraps", "megadetector", "detection",
        "run_detector_batch.py")
    image_recognition_file = os.path.join(chosen_folder, "image_recognition_file.json")
    process_video_py = os.path.join(
        config.base_path, "cameratraps", "megadetector", "detection", "process_video.py")
    video_recognition_file = (
        "--output_json_file=" + os.path.join(chosen_folder, "video_recognition_file.json"))
    python_executable = get_python_interpreter(config.base_path, "base")

    # ── model selection ──────────────────────────────────────────────────────
    custom_model_bool = False
    label_map: Dict[str, Any] = {}
    det_model_fpath: str = ""

    if simple_mode:
        det_model_fpath = os.path.join(
            config.det_model_dir, "MegaDetector 5a", "md_v5a.0.0.pt")
        switch_yolov5_version("old models", config.base_path)

    elif config.det_model_name != config.dpd_options_model[config.lang_idx][-1]:
        # standard (non-custom) model
        model_vars = load_model_vars_for(config.base_path, "det", config.det_model_name)
        det_model_fname = model_vars["model_fname"]
        det_model_fpath = os.path.join(
            config.det_model_dir, config.det_model_name, det_model_fname)
        switch_yolov5_version("old models", config.base_path)

    else:
        # custom model
        det_model_fpath = config.det_model_path
        custom_model_bool = True
        switch_yolov5_version("new models", config.base_path)

        # extract classes from the checkpoint
        label_map = extract_label_map_from_model(det_model_fpath)

        # write label map alongside results so downstream steps can use it
        json_object = json.dumps(label_map, indent=1)
        native_model_classes_json_file = os.path.join(
            chosen_folder, "native_model_classes.json")
        with open(native_model_classes_json_file, "w") as outfile:
            outfile.write(json_object)

        selected_options.append(
            "--class_mapping_filename=" + native_model_classes_json_file)

    # ── full-image-classifier shortcut ────────────────────────────────────
    # Some classifiers operate on the full frame; for those we skip the normal
    # detection subprocess and synthesise a detection JSON instead.
    cls_model_vars = load_model_vars_for(
        config.base_path, "cls", config.cls_model_name)
    full_image_cls = cls_model_vars.get("full_image_cls", False)

    if full_image_cls:
        imitate_object_detection_for_full_image_classifier(chosen_folder)

    else:
        # ── build subprocess command ─────────────────────────────────────
        if os.name == 'nt':
            if selected_options == []:
                img_command = [python_executable, run_detector_batch_py,
                               det_model_fpath, '--threshold=0.01',
                               chosen_folder, image_recognition_file]
                vid_command = [python_executable, process_video_py,
                               '--max_width=1280', '--verbose', '--quality=85',
                               '--allow_empty_video', video_recognition_file,
                               det_model_fpath, chosen_folder]
            else:
                img_command = [python_executable, run_detector_batch_py,
                               det_model_fpath, *selected_options,
                               '--threshold=0.01', chosen_folder,
                               image_recognition_file]
                vid_command = [python_executable, process_video_py,
                               *selected_options, '--max_width=1280',
                               '--verbose', '--quality=85', '--allow_empty_video',
                               video_recognition_file, det_model_fpath,
                               chosen_folder]
        else:
            if selected_options == []:
                img_command = [
                    f"'{python_executable}' '{run_detector_batch_py}'"
                    f" '{det_model_fpath}' '--threshold=0.01'"
                    f" '{chosen_folder}' '{image_recognition_file}'"]
                vid_command = [
                    f"'{python_executable}' '{process_video_py}'"
                    " '--max_width=1280' '--verbose' '--quality=85'"
                    f" '--allow_empty_video' '{video_recognition_file}'"
                    f" '{det_model_fpath}' '{chosen_folder}'"]
            else:
                selected_options_str = "' '".join(selected_options)
                img_command = [
                    f"'{python_executable}' '{run_detector_batch_py}'"
                    f" '{det_model_fpath}' '{selected_options_str}'"
                    f" '--threshold=0.01' '{chosen_folder}'"
                    f" '{image_recognition_file}'"]
                vid_command = [
                    f"'{python_executable}' '{process_video_py}'"
                    f" '{selected_options_str}' '--max_width=1280'"
                    " '--verbose' '--quality=85' '--allow_empty_video'"
                    f" '{video_recognition_file}' '{det_model_fpath}'"
                    f" '{chosen_folder}'"]

        command = img_command if data_type == "img" else vid_command

        # ── GPU disable ──────────────────────────────────────────────────
        if config.disable_gpu and not simple_mode:
            if os.name == 'nt':
                command[:0] = ['set', 'CUDA_VISIBLE_DEVICES=""', '&']
            elif platform.system() == 'Darwin':
                callbacks.on_warning(
                    t('warning'),
                    ["Disabling GPU processing is currently only supported for CUDA "
                     "devices on Linux and Windows machines, not on macOS. Proceeding "
                     "without GPU disabled.",
                     "Deshabilitar el procesamiento de la GPU actualmente sólo es "
                     "compatible con dispositivos CUDA en máquinas Linux y Windows, no "
                     "en macOS. Proceder sin GPU desactivada.",
                     "La désactivation du traitement par GPU est uniquement supportée "
                     "sur les dispositifs CUDA sous Linux et Windows, pas sous MacOS. "
                     "Poursuite du traitement sans désactiver le GPU."
                     ""][config.lang_idx],
                )
            else:
                command = "CUDA_VISIBLE_DEVICES='' " + command  # type: ignore[assignment]

        logger.debug("Command: %s", command)

        # ── launch subprocess ────────────────────────────────────────────
        if os.name == 'nt':
            p = Popen(command,
                      stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT,
                      bufsize=1,
                      shell=True,
                      universal_newlines=True)
        else:
            p = Popen(command,
                      stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT,
                      bufsize=1,
                      shell=True,
                      universal_newlines=True,
                      preexec_fn=os.setsid)  # type: ignore[attr-defined]

        # obtain cancel function now that we have the process
        cancel_func = cancel_func_factory(p)

        # ── file-based log callbacks ─────────────────────────────────────
        def _log_exception(line: str) -> None:
            with open(error_log_path, 'a+') as f:
                f.write(f"{line}\n")

        def _log_warning(line: str) -> None:
            with open(warning_log_path, 'a+') as f:
                f.write(f"{line}\n")

        # ── event-bus emit callbacks ─────────────────────────────────────
        # Capture previous_processed_img forwarded in unicode-error emit so we
        # can include it in the error dialog shown to the user.
        unicode_error_img: List[str] = [
            ["There is no previously processed image. The problematic character "
             "is in the first image to analyse.",
             "No hay ninguna imagen previamente procesada. El personaje problemático "
             "está en la primera imagen a analizar.",
             "Il n'y a aucune image traitée précédemment. Le caractère problématique "
             "est dans la première image à analyser."
             ][config.lang_idx]
        ]

        def _emit_progress(**kwargs: Any) -> None:
            event_bus.emit(DEPLOY_PROGRESS, **kwargs)

        def _emit_error(**kwargs: Any) -> None:
            event_bus.emit(DEPLOY_ERROR, **kwargs)
            if "previous_processed_img" in kwargs:
                unicode_error_img[0] = kwargs["previous_processed_img"]

        # ── frame/video display unit ─────────────────────────────────────
        if data_type == "vid" and config.cls_model_name == t('none'):
            frame_video_choice: Optional[str] = "video"
        elif data_type == "vid" and config.cls_model_name != t('none'):
            frame_video_choice = "frame"
        else:
            frame_video_choice = None

        # ── initial "no previously processed image" text (multilingual) ─
        previous_processed_img = [
            "There is no previously processed image. The problematic character "
            "is in the first image to analyse.",
            "No hay ninguna imagen previamente procesada. El personaje problemático "
            "está en la primera imagen a analizar.",
            "Il n'y a aucune image traitée précédemment. Le caractère problématique "
            "est dans la première image à analyser."
        ][config.lang_idx]

        # ── parse subprocess stdout ──────────────────────────────────────
        parse_result = parse_detection_stdout(
            stdout_lines=p.stdout,
            data_type=data_type,
            update_ui=callbacks.update_ui,
            emit_progress=_emit_progress,
            emit_error=_emit_error,
            log_line=logger.info,
            log_exception=_log_exception,
            log_warning=_log_warning,
            previous_processed_img=previous_processed_img,
            frame_video_choice=frame_video_choice,
            cancel_func=cancel_func,
        )

        # ── handle error result codes ────────────────────────────────────
        if parse_result == "no_images":
            callbacks.on_error(
                t('msg_no_images_found'),
                [f"There are no images found in '{chosen_folder}'. \n\nAre you sure "
                 f"you specified the correct folder? If the files are in subdirectories,"
                 f" make sure you don't tick '{t('lbl_exclude_subs')}'.",
                 f"No se han encontrado imágenes en '{chosen_folder}'. \n\n¿Está seguro "
                 f"de haber especificado la carpeta correcta? Si los archivos están en "
                 f"subdirectorios, asegúrese de no marcar la casilla "
                 f"'{t('lbl_exclude_subs')}'.",
                 f"Aucune image trouvée dans '{chosen_folder}'. \n\nAvez-vous spécifié le"
                 f" bon dossier? Si les fichiers sont dans des sous-dossiers, assurez-vous"
                 f" ne pas avoir coché '{t('lbl_exclude_subs')}'."][config.lang_idx],
            )
            return DetectionResult(success=False, json_path=None,
                                   error_code="no_images",
                                   error_message="No image files found")

        if parse_result == "no_videos":
            callbacks.on_error(
                t('msg_no_videos_found'),
                [f"\n\nAre you sure you specified the correct folder? If the files are in"
                 f" subdirectories, make sure you don't tick '{t('lbl_exclude_subs')}'.",
                 f"\n\n¿Está seguro de haber especificado la carpeta correcta? Si los "
                 f"archivos están en subdirectorios, asegúrese de no marcar la casilla "
                 f"'{t('lbl_exclude_subs')}'.",
                 f"\n\nAvez-vous spécifié le bon dossier? Si les fichiers sont dans des "
                 f"sous-dossiers, assurez-vous ne pas avoir coché "
                 f"'{t('lbl_exclude_subs')}'."][config.lang_idx],
            )
            return DetectionResult(success=False, json_path=None,
                                   error_code="no_videos",
                                   error_message="No videos found")

        if parse_result == "no_frames":
            callbacks.on_error(
                t('msg_could_not_extract_frames'),
                ["\n\nConverting the videos to .mp4 might fix the issue.",
                 "\n\nConvertir los vídeos a .mp4 podría solucionar el problema.",
                 "\n\nConvertir les vidéos au format .mp4 pourrait régler le problème."
                 ][config.lang_idx],
            )
            return DetectionResult(success=False, json_path=None,
                                   error_code="no_frames",
                                   error_message="No frames extracted")

        if parse_result == "unicode_error":
            callbacks.on_error(
                "Unparsable special character",
                ["There seems to be a special character in a filename that cannot be "
                 "parsed. Unfortunately, it's not possible to point you to the "
                 "problematic file directly, but I can tell you that the last "
                 f"successfully analysed image was\n\n{unicode_error_img[0]}\n\n"
                 "The problematic character should be in the file or folder name of "
                 "the next image, alphabetically. Please remove any special characters "
                 "from the path and try again.",
                 "Parece que hay un carácter especial en un nombre de archivo que no se "
                 "puede analizar. Lamentablemente, no es posible indicarle directamente "
                 "el archivo problemático, pero puedo decirle que la última imagen "
                 f"analizada con éxito fue\n\n{unicode_error_img[0]}\n\nEl carácter "
                 "problemático debe estar en el nombre del archivo o carpeta de la "
                 "siguiente imagen, alfabéticamente. Elimine los caracteres especiales de "
                 "la ruta e inténtelo de nuevo.",
                 "Il semble y avoir un caractère spécial non-reconnu dans le nom d'un "
                 "fichier. Malheureusement, il est impossible d'identifier le fichier "
                 "directement, cependant la dernière images correctement analysée était "
                 f"\n\n{unicode_error_img[0]}\n\nLe caractère problématique devrait être "
                 "dans le nom de fichier ou de dossier de la prochaine image, "
                 "alphabetiquement. SVP remplacer tout caractère spécial du chemin et du "
                 "nom de fichier et réessayer."
                 ][config.lang_idx],
            )
            return DetectionResult(success=False, json_path=None,
                                   error_code="unicode_error",
                                   error_message="UnicodeEncodeError: Unparsable special "
                                                 "character in filename")

    # ── write AddaxAI metadata into the recognition JSON ─────────────────────
    addaxai_metadata: Dict[str, Any] = {
        "addaxai_metadata": {
            "version": current_version,
            "custom_model": custom_model_bool,
            "custom_model_info": {},
        }
    }
    if custom_model_bool:
        addaxai_metadata["addaxai_metadata"]["custom_model_info"] = {
            "model_name": os.path.basename(os.path.normpath(det_model_fpath)),
            "label_map": label_map,
        }

    # reuse variable names to match original code (video_recognition_file was
    # previously the --output_json_file= flag string; reassign to plain path here)
    image_recognition_file = os.path.join(chosen_folder, "image_recognition_file.json")
    video_recognition_file = os.path.join(chosen_folder, "video_recognition_file.json")

    if data_type == "img" and os.path.isfile(image_recognition_file):
        append_to_json(image_recognition_file, addaxai_metadata)
        if config.use_abs_paths:
            make_json_absolute(image_recognition_file, config.source_folder)
    if data_type == "vid" and os.path.isfile(video_recognition_file):
        append_to_json(video_recognition_file, addaxai_metadata)
        if config.use_abs_paths:
            make_json_absolute(video_recognition_file, config.source_folder)

    # ── emit finished or cancelled event ────────────────────────────────────
    if not callbacks.cancel_check():
        results_path = (image_recognition_file if data_type == "img"
                        else video_recognition_file)
        event_bus.emit(DEPLOY_FINISHED, results_path=results_path,
                       process=f"{data_type}_det")
        return DetectionResult(success=True, json_path=results_path,
                               error_code=None, error_message=None)
    else:
        event_bus.emit(DEPLOY_CANCELLED, process=f"{data_type}_det")
        return DetectionResult(success=False, json_path=None,
                               error_code="cancelled",
                               error_message="User cancelled")


# ============================================================================
# ClassificationResult
# ============================================================================

@dataclasses.dataclass
class ClassificationResult:
    """Result returned by run_classification().

    Attributes:
        success:       True if classification completed without error.
        json_path:     Path to the recognition JSON that was classified, or None.
        error_code:    Short machine-readable code: "no_crops", or None on success.
        error_message: Human-readable description of the error, or None.
    """
    success: bool
    json_path: Optional[str]
    error_code: Optional[str]
    error_message: Optional[str]


# ============================================================================
# run_classification
# ============================================================================

def run_classification(
    config: ClassifyConfig,
    callbacks: OrchestratorCallbacks,
    json_fpath: str,
    data_type: str,
    cancel_func_factory: Callable[[Any], Callable[[], None]],
    simple_mode: bool = False,
) -> ClassificationResult:
    """Run species classification on MegaDetector detections.

    Args:
        config:              Classification configuration (paths, thresholds, flags).
        callbacks:           Injected callbacks for UI interaction.
        json_fpath:          Path to the MegaDetector recognition JSON to classify.
        data_type:           "img" for images, "vid" for videos.
        cancel_func_factory: Called with the Popen subprocess instance after launch;
                             must return a zero-argument callable that cancels the
                             subprocess.  The returned callable is forwarded as
                             cancel_func to emit_progress.
        simple_mode:         True when running in simplified (beginner) mode.
                             Overrides thresholds with model-default values.

    Returns:
        ClassificationResult — check .success and .error_code for outcome.
    """
    logger.debug("EXECUTED: run_classification")

    # emit started event
    event_bus.emit(CLASSIFY_STARTED, process=f"{data_type}_cls")

    # show user it's loading
    callbacks.update_ui()
    event_bus.emit(CLASSIFY_PROGRESS, pct=0.0, message="Loading classification model",
                   process=f"{data_type}_cls", status="load")

    # load model-specific variables
    model_vars = load_model_vars_for(config.base_path, "cls", config.cls_model_name)
    cls_model_fname = model_vars["model_fname"]
    cls_model_type = model_vars["type"]
    cls_model_fpath = os.path.join(
        config.base_path, "models", "cls", config.cls_model_name, cls_model_fname)

    # check if taxonomic fallback should be the default
    taxon_mapping_csv_is_present = taxon_mapping_csv_present(
        config.base_path, config.cls_model_name)
    taxon_mapping_is_default = model_vars.get("var_tax_fallback_default", False)

    # pick OS-specific conda environment
    if os.name == 'nt':
        cls_model_env = model_vars.get("env-windows", model_vars["env"])
    elif platform.system() == 'Darwin':
        cls_model_env = model_vars.get("env-macos", model_vars["env"])
    else:
        cls_model_env = model_vars.get("env-linux", model_vars["env"])

    # resolve parameter values (simple_mode overrides config thresholds)
    cls_tax_fallback = False
    cls_tax_levels_idx = 0
    if simple_mode:
        cls_disable_GPU = False
        cls_detec_thresh = model_vars["var_cls_detec_thresh_default"]
        cls_class_thresh = model_vars["var_cls_class_thresh_default"]
        cls_animal_smooth = False
        if taxon_mapping_csv_is_present:
            if taxon_mapping_is_default:
                cls_tax_fallback = True
    else:
        cls_disable_GPU = config.disable_gpu
        cls_detec_thresh = config.cls_detec_thresh
        cls_class_thresh = config.cls_class_thresh
        cls_animal_smooth = config.smooth_cls_animal
        if taxon_mapping_csv_is_present:
            cls_tax_fallback = config.tax_fallback
            cls_tax_levels_idx = model_vars["var_tax_levels_idx"]

    # init paths
    python_executable = get_python_interpreter(config.base_path, cls_model_env)
    inference_script = os.path.join(
        config.base_path, "AddaxAI", "classification_utils", "model_types",
        cls_model_type, "classify_detections.py")

    # create command argument list
    command_args = []
    command_args.append(python_executable)
    command_args.append(inference_script)
    command_args.append(config.base_path)
    command_args.append(cls_model_fpath)
    command_args.append(str(cls_detec_thresh))
    command_args.append(str(cls_class_thresh))
    command_args.append(str(cls_animal_smooth))
    command_args.append(json_fpath)
    if config.temp_frame_folder:
        command_args.append(config.temp_frame_folder)
    else:
        command_args.append("None")
    command_args.append(str(cls_tax_fallback))
    command_args.append(str(cls_tax_levels_idx))

    # adjust command for unix OS
    if os.name != 'nt':
        command_args = "'" + "' '".join(command_args) + "'"  # type: ignore[assignment]

    # prepend with OS-specific GPU/env commands
    if os.name == 'nt':
        if cls_disable_GPU:
            command_args = ['set CUDA_VISIBLE_DEVICES="" &'] + command_args  # type: ignore[operator]
    elif platform.system() == 'Darwin':
        command_args = "export PYTORCH_ENABLE_MPS_FALLBACK=1 && " + command_args  # type: ignore[operator]
    else:
        if cls_disable_GPU:
            command_args = "CUDA_VISIBLE_DEVICES='' " + command_args  # type: ignore[operator]
        else:
            command_args = "export PYTORCH_ENABLE_MPS_FALLBACK=1 && " + command_args  # type: ignore[operator]

    logger.debug("Command: %s", command_args)

    # launch subprocess
    if os.name == 'nt':
        p = Popen(command_args,
                  stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT,
                  bufsize=1,
                  shell=True,
                  universal_newlines=True)
    else:
        p = Popen(command_args,
                  stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT,
                  bufsize=1,
                  shell=True,
                  universal_newlines=True,
                  preexec_fn=os.setsid)  # type: ignore[attr-defined]

    # obtain cancel function now that we have the process
    cancel_func = cancel_func_factory(p)

    # smooth-output file path (same directory as the recognition JSON)
    smooth_output_file = os.path.join(os.path.dirname(json_fpath), "smooth-output.txt")

    def _smooth_handler(smooth_output_line: str) -> None:
        with open(smooth_output_file, 'a+') as f:
            f.write(f"{smooth_output_line}\n")

    def _emit_progress(**kwargs: Any) -> None:
        event_bus.emit(CLASSIFY_PROGRESS, **kwargs)

    def _emit_error(**kwargs: Any) -> None:
        event_bus.emit(CLASSIFY_ERROR, **kwargs)

    # parse subprocess stdout
    parse_result = parse_classification_stdout(
        stdout_lines=p.stdout,
        data_type=data_type,
        update_ui=callbacks.update_ui,
        emit_progress=_emit_progress,
        emit_error=_emit_error,
        log_line=logger.info,
        smooth_handler=_smooth_handler,
        cancel_func=cancel_func,
    )

    # handle no_crops result
    if parse_result == "no_crops":
        callbacks.on_info(
            t('information'),
            ["There are no animal detections that meet the criteria. You either "
             "have selected images without any animals present, or you have set "
             "your detection confidence threshold to high.",
             "No hay detecciones de animales que cumplan los criterios. O bien ha "
             "seleccionado imágenes sin presencia de animales, o bien ha establecido "
             "el umbral de confianza de detección en alto.",
             "Aucune détection d'animal ne rencontre les critères. Vous avez soit "
             "sélectionner des images sans animaux présents, ou vous avez régler le "
             "seuil de confiance de détection trop haut."
             ][config.lang_idx],
        )
        event_bus.emit(CLASSIFY_PROGRESS, pct=100.0, message="Classification complete",
                       process=f"{data_type}_cls", status="done",
                       time_ela="", speed="")
        event_bus.emit(CLASSIFY_FINISHED, results_path=json_fpath,
                       process=f"{data_type}_cls")
        callbacks.update_ui()
        return ClassificationResult(
            success=False, json_path=None,
            error_code="no_crops",
            error_message="No animal detections that meet the criteria",
        )

    # emit finished event
    event_bus.emit(CLASSIFY_FINISHED, results_path=json_fpath, process=f"{data_type}_cls")
    callbacks.update_ui()

    return ClassificationResult(
        success=True, json_path=json_fpath,
        error_code=None, error_message=None,
    )


# ============================================================================
# PostprocessResult
# ============================================================================

@dataclasses.dataclass
class PostprocessResult:
    """Result returned by run_postprocess().

    Attributes:
        success:       True if postprocessing completed without error.
        error_code:    Short machine-readable code: "no_json", "invalid_dest",
                       "exception", or None on success.
        error_message: Human-readable description of the error, or None.
    """
    success: bool
    error_code: Optional[str]
    error_message: Optional[str]


# ============================================================================
# _postprocess_inner  (B8a — extracted postprocess() body)
# ============================================================================

# dtype dict for pandas CSV imports (mirrors the module-level dtypes in app.py)
_POSTPROCESS_DTYPES = {
    'absolute_path': 'str', 'relative_path': 'str', 'data_type': 'str',
    'label': 'str', 'confidence': 'float64', 'human_verified': 'bool',
    'bbox_left': 'str', 'bbox_top': 'str', 'bbox_right': 'str',
    'bbox_bottom': 'str', 'file_height': 'str', 'file_width': 'str',
    'DateTimeOriginal': 'str', 'DateTime': 'str', 'DateTimeDigitized': 'str',
    'Latitude': 'str', 'Longitude': 'str', 'GPSLink': 'str', 'Altitude': 'str',
    'Make': 'str', 'Model': 'str', 'Flash': 'str', 'ExifOffset': 'str',
    'ResolutionUnit': 'str', 'YCbCrPositioning': 'str', 'XResolution': 'str',
    'YResolution': 'str', 'ExifVersion': 'str', 'ComponentsConfiguration': 'str',
    'FlashPixVersion': 'str', 'ColorSpace': 'str', 'ExifImageWidth': 'str',
    'ISOSpeedRatings': 'str', 'ExifImageHeight': 'str', 'ExposureMode': 'str',
    'WhiteBalance': 'str', 'SceneCaptureType': 'str', 'ExposureTime': 'str',
    'Software': 'str', 'Sharpness': 'str', 'Saturation': 'str',
    'ReferenceBlackWhite': 'str', 'n_detections': 'int64',
    'max_confidence': 'float64',
}


def _postprocess_inner(
    src_dir,               # type: str
    dst_dir,               # type: str
    thresh,                # type: float
    sep,                   # type: bool
    keep_series,           # type: bool
    keep_series_seconds,   # type: float
    file_placement,        # type: int
    sep_conf,              # type: bool
    vis,                   # type: bool
    crp,                   # type: bool
    exp,                   # type: bool
    plt,                   # type: bool
    exp_format,            # type: str
    data_type,             # type: str
    keep_series_species=None,          # type: Optional[List[str]]
    vis_blur=False,                    # type: bool
    vis_bbox=True,                     # type: bool
    vis_size_idx=0,                    # type: int
    cancel_check=None,                 # type: Optional[Callable[[], bool]]
    update_ui=None,                    # type: Optional[Callable[[], None]]
    cancel_func=None,                  # type: Optional[Callable[[], None]]
    produce_plots_func=None,           # type: Optional[Callable[[str], None]]
    on_confirm=None,                   # type: Optional[Callable[[str, str], bool]]
    on_error=None,                     # type: Optional[Callable[[str, str], None]]
    current_version="",                # type: str
    lang_idx=0,                        # type: int
    base_path="",                      # type: str
    cls_model_name="",                 # type: str
):
    # type: (...) -> None
    """Extracted body of postprocess() — all GUI dependencies replaced by parameters.

    Called by run_postprocess() for each data_type ("img" or "vid").
    Heavy imports (cv2, PIL, pandas, gpsphoto, bb) are done lazily inside
    the function body so they are not required for unit testing.
    """
    # lazy heavy imports
    import cv2  # type: ignore[import]
    import gpsphoto  # type: ignore[import]
    import pandas as pd  # type: ignore[import]
    import PIL  # type: ignore[import]
    import PIL.ExifTags  # type: ignore[import]
    import PIL.Image  # type: ignore[import]
    from visualise_detection.bounding_box import bounding_box as bb  # type: ignore[import]

    _cancel_check = cancel_check if cancel_check is not None else (lambda: False)
    _update_ui = update_ui if update_ui is not None else (lambda: None)
    _cancel_func = cancel_func if cancel_func is not None else (lambda: None)
    _on_confirm = on_confirm if on_confirm is not None else (lambda t, m: True)
    _on_error = on_error if on_error is not None else (lambda t, m: None)

    logger.debug("EXECUTED: _postprocess_inner data_type=%s", data_type)

    # update progress window via event bus
    event_bus.emit(POSTPROCESS_PROGRESS, pct=0.0, message="Initializing postprocessing",
                   process=f"{data_type}_pst", status="load")

    # plt needs csv files so make sure to produce them, even if the user didn't specify
    remove_csv = False
    if plt and not exp:
        if not (os.path.isfile(os.path.join(dst_dir, "results_detections.csv")) and
                os.path.isfile(os.path.join(dst_dir, "results_files.csv"))):
            exp = True
            exp_format = t('dpd_exp_format')[1]  # CSV
            remove_csv = True

    # get correct json file
    if data_type == "img":
        recognition_file = os.path.join(src_dir, "image_recognition_file.json")
    else:
        recognition_file = os.path.join(src_dir, "video_recognition_file.json")

    # check if user is not in the middle of an annotation session
    if data_type == "img" and get_hitl_var_in_json(recognition_file) == "in-progress":
        if not _on_confirm(
            t('msg_verification_in_progress_title'),
            [f"Your verification session is not yet done. You can finish the session by clicking "
             f"'Continue' at '{t('lbl_hitl_main')}', or just continue to post-process with the "
             f"results as they are now.\n\nDo you want to continue to post-process?",
             f"La sesión de verificación aún no ha finalizado. Puede finalizarla haciendo clic en "
             f"'Continuar' en '{t('lbl_hitl_main')}', o simplemente continuar con el "
             f"posprocesamiento con los resultados tal como están ahora.\n\n¿Quieres continuar "
             f"con el posprocesamiento?",
             f"Votre session de vérification n'est pas encore terminée. Vous pouvez la compléter "
             f"en cliquant sur '{t('lbl_hitl_main')}', ou juste continuer le post-traitement avec "
             f"les résultats actuels.\n\nSouhaitez-vous continuer le post-traitement?"
             ][lang_idx]
        ):
            return

    # init vars
    start_time = time.time()
    nloop = 1

    # warn user about unsupported features for videos
    if data_type == "vid":
        if vis or crp or plt:
            _on_error(
                t('error'),
                ["Visualizing, cropping, and plotting are not supported for videos. "
                 "These options will be disabled.",
                 "La visualización, el recorte y los gráficos no son compatibles con vídeos. "
                 "Estas opciones serán desactivadas.",
                 "La visualisation, le rognage et les graphiques ne sont pas pris en charge "
                 "pour les vidéos. Ces options seront désactivées."
                 ][lang_idx],
            )
            vis, crp, plt = False, False, False

    # fetch label map
    label_map = fetch_label_map_from_json(recognition_file)
    inverted_label_map = {v: k for k, v in label_map.items()}

    # create list with colours for visualisation
    if vis:
        colors = ["fuchsia", "blue", "orange", "yellow", "green", "red", "aqua",
                  "navy", "teal", "olive", "lime", "maroon", "purple"]
        colors = colors * 30

    # make sure json has relative paths
    json_paths_converted = False
    if check_json_paths(recognition_file, src_dir) != "relative":
        make_json_relative(recognition_file, src_dir)
        json_paths_converted = True

    # open json file
    with open(recognition_file) as image_recognition_file_content:
        data = json.load(image_recognition_file_content)
    n_images = len(data['images'])

    # series support: build timestamp index once for efficiency
    if keep_series:
        try:
            _file_list_for_index = [img['file'] for img in data.get('images', [])]
            timestamp_index = build_image_timestamp_index(src_dir, _file_list_for_index)
        except Exception:
            timestamp_index = {}
    # used to avoid moving the same original file multiple times
    already_moved_files = set()  # type: ignore[var-annotated]

    # initialise the csv files
    if exp:
        csv_for_files = os.path.join(dst_dir, "results_files.csv")
        if not os.path.isfile(csv_for_files):
            df = pd.DataFrame(list(), columns=[
                "absolute_path", "relative_path", "data_type", "n_detections",
                "file_height", "file_width", "max_confidence", "human_verified",
                'DateTimeOriginal', 'DateTime', 'DateTimeDigitized', 'Latitude',
                'Longitude', 'GPSLink', 'Altitude', 'Make', 'Model', 'Flash',
                'ExifOffset', 'ResolutionUnit', 'YCbCrPositioning', 'XResolution',
                'YResolution', 'ExifVersion', 'ComponentsConfiguration',
                'FlashPixVersion', 'ColorSpace', 'ExifImageWidth', 'ISOSpeedRatings',
                'ExifImageHeight', 'ExposureMode', 'WhiteBalance', 'SceneCaptureType',
                'ExposureTime', 'Software', 'Sharpness', 'Saturation',
                'ReferenceBlackWhite'])
            df.to_csv(csv_for_files, encoding='utf-8', index=False)

        csv_for_detections = os.path.join(dst_dir, "results_detections.csv")
        if not os.path.isfile(csv_for_detections):
            df = pd.DataFrame(list(), columns=[
                "absolute_path", "relative_path", "data_type", "label", "confidence",
                "human_verified", "bbox_left", "bbox_top", "bbox_right", "bbox_bottom",
                "file_height", "file_width", 'DateTimeOriginal', 'DateTime',
                'DateTimeDigitized', 'Latitude', 'Longitude', 'GPSLink', 'Altitude',
                'Make', 'Model', 'Flash', 'ExifOffset', 'ResolutionUnit',
                'YCbCrPositioning', 'XResolution', 'YResolution', 'ExifVersion',
                'ComponentsConfiguration', 'FlashPixVersion', 'ColorSpace',
                'ExifImageWidth', 'ISOSpeedRatings', 'ExifImageHeight', 'ExposureMode',
                'WhiteBalance', 'SceneCaptureType', 'ExposureTime', 'Software',
                'Sharpness', 'Saturation', 'ReferenceBlackWhite'])
            df.to_csv(csv_for_detections, encoding='utf-8', index=False)

    # set error log path
    postprocessing_error_log = os.path.join(dst_dir, "postprocessing_error_log.txt")

    # count rows to ensure they don't exceed excel's limit
    if exp and exp_format == t('dpd_exp_format')[0]:  # XLSX
        n_rows_files = 1
        n_rows_detections = 1
        for image in data['images']:
            n_rows_files += 1
            if 'detections' in image:
                for detection in image['detections']:
                    if detection["conf"] >= thresh:
                        n_rows_detections += 1
        if n_rows_detections > 1048576 or n_rows_files > 1048576:
            _on_error(
                t('msg_too_many_rows'),
                ["The XLSX file you are trying to create is too large!\n\nThe maximum "
                 f"number of rows in an XSLX file is 1048576, while you are trying to "
                 f"create a sheet with {max(n_rows_files, n_rows_detections)} rows.\n\n"
                 f"If you require the results in XLSX format, please run the process on "
                 f"smaller chunks so that it doesn't exceed Microsoft's row limit. Or "
                 f"choose CSV as {t('lbl_exp_format')} in advanced mode.",
                 "¡El archivo XLSX que está intentando crear es demasiado grande!\n\n"
                 f"El número máximo de filas en un archivo XSLX es 1048576, mientras que "
                 f"usted está intentando crear una hoja con "
                 f"{max(n_rows_files, n_rows_detections)} filas.\n\nSi necesita los "
                 f"resultados en formato XLSX, ejecute el proceso en trozos más pequeños "
                 f"para que no supere el límite de filas de Microsoft. O elija CSV como "
                 f"{t('lbl_exp_format')} en modo avanzado.",
                 "Le fichier XLSX que vous tenter de créer est trop long!\n\nLe nombre "
                 f"maximum de lignes dans un fichier XSLX est 1048576, alors que vous "
                 f"tenter de créer une feuille avec "
                 f"{max(n_rows_files, n_rows_detections)} lignes.\n\nSi vous souhaitez "
                 f"des résultats sous format XLSX, svp exécuter le processus sur de plus "
                 f"petites portions de sorte à ne pas excéder la limite de lignes de "
                 f"Microsoft. Ou choisissez le format CSV comme {t('lbl_exp_format')} "
                 f"dans le mode avancé."][lang_idx],
            )
            return

    # loop through images
    for image in data['images']:

        # cancel process if required
        if _cancel_check():
            break

        # check for failure
        if "failure" in image:
            with open(postprocessing_error_log, 'a+') as f:
                f.write(f"File '{image['file']}' was skipped by post processing features "
                        f"because '{image['failure']}'\n")

            elapsed_time_sep = str(datetime.timedelta(
                seconds=round(time.time() - start_time)))
            time_left_sep = str(datetime.timedelta(
                seconds=round(((time.time() - start_time) * n_images / nloop) -
                               (time.time() - start_time))))
            percentage = (nloop / n_images) * 100
            event_bus.emit(POSTPROCESS_PROGRESS, pct=float(percentage),
                           message=f"Processing: {nloop}/{n_images}",
                           process=f"{data_type}_pst", status="running",
                           cur_it=nloop, tot_it=n_images,
                           time_ela=elapsed_time_sep, time_rem=time_left_sep,
                           cancel_func=_cancel_func)
            nloop += 1
            _update_ui()
            continue

        # get image info
        file = image['file']
        detections_list = image['detections']
        n_detections = len(detections_list)

        # check if it has been manually verified
        manually_checked = False
        if 'manually_checked' in image:
            if image['manually_checked']:
                manually_checked = True

        # init vars
        max_detection_conf = 0.0
        unique_labels = []  # type: ignore[var-annotated]
        bbox_info = []  # type: ignore[var-annotated]

        # open files
        if vis or crp or exp:
            if data_type == "img":
                im_to_vis = cv2.imread(os.path.normpath(os.path.join(src_dir, file)))

                if im_to_vis is None:
                    with open(postprocessing_error_log, 'a+') as f:
                        f.write(f"File '{image['file']}' was skipped by post processing "
                                f"features. This might be due to the file being moved or "
                                f"deleted after analysis, or because of a special character "
                                f"in the file path.\n")
                    elapsed_time_sep = str(datetime.timedelta(
                        seconds=round(time.time() - start_time)))
                    time_left_sep = str(datetime.timedelta(
                        seconds=round(((time.time() - start_time) * n_images / nloop) -
                                       (time.time() - start_time))))
                    percentage = (nloop / n_images) * 100
                    event_bus.emit(POSTPROCESS_PROGRESS, pct=float(percentage),
                                   message=f"Processing: {nloop}/{n_images}",
                                   process=f"{data_type}_pst", status="running",
                                   cur_it=nloop, tot_it=n_images,
                                   time_ela=elapsed_time_sep, time_rem=time_left_sep,
                                   cancel_func=_cancel_func)
                    nloop += 1
                    _update_ui()
                    continue

                im_to_crop_path = os.path.join(src_dir, file)

                # load old image and extract EXIF
                origImage = PIL.Image.open(os.path.join(src_dir, file))
                try:
                    exif = origImage.info['exif']
                except Exception:
                    exif = None
                origImage.close()
            else:
                vid = cv2.VideoCapture(os.path.join(src_dir, file))

            # read image dates etc
            if exp:
                try:
                    img_for_exif = PIL.Image.open(os.path.join(src_dir, file))
                    metadata = {
                        PIL.ExifTags.TAGS[k]: v
                        for k, v in img_for_exif._getexif().items()
                        if k in PIL.ExifTags.TAGS
                    }
                    img_for_exif.close()
                except Exception:
                    metadata = {
                        'GPSInfo': None, 'ResolutionUnit': None, 'ExifOffset': None,
                        'Make': None, 'Model': None, 'DateTime': None,
                        'YCbCrPositioning': None, 'XResolution': None,
                        'YResolution': None, 'ExifVersion': None,
                        'ComponentsConfiguration': None, 'ShutterSpeedValue': None,
                        'DateTimeOriginal': None, 'DateTimeDigitized': None,
                        'FlashPixVersion': None, 'UserComment': None,
                        'ColorSpace': None, 'ExifImageWidth': None,
                        'ExifImageHeight': None,
                    }

                try:
                    gpsinfo = gpsphoto.getGPSData(os.path.join(src_dir, file))
                    if 'Latitude' in gpsinfo and 'Longitude' in gpsinfo:
                        gpsinfo['GPSLink'] = (
                            f"https://maps.google.com/?q={gpsinfo['Latitude']},"
                            f"{gpsinfo['Longitude']}")
                except Exception:
                    gpsinfo = {'Latitude': None, 'Longitude': None, 'GPSLink': None}

                exif_data = {**metadata, **gpsinfo}

                exif_params = []
                for param in [
                    'DateTimeOriginal', 'DateTime', 'DateTimeDigitized', 'Latitude',
                    'Longitude', 'GPSLink', 'Altitude', 'Make', 'Model', 'Flash',
                    'ExifOffset', 'ResolutionUnit', 'YCbCrPositioning', 'XResolution',
                    'YResolution', 'ExifVersion', 'ComponentsConfiguration',
                    'FlashPixVersion', 'ColorSpace', 'ExifImageWidth', 'ISOSpeedRatings',
                    'ExifImageHeight', 'ExposureMode', 'WhiteBalance', 'SceneCaptureType',
                    'ExposureTime', 'Software', 'Sharpness', 'Saturation',
                    'ReferenceBlackWhite',
                ]:
                    try:
                        if param.startswith('DateTime'):
                            datetime_raw = str(exif_data[param])
                            param_value = datetime.datetime.strptime(
                                datetime_raw, '%Y:%m:%d %H:%M:%S'
                            ).strftime('%d/%m/%y %H:%M:%S')
                        else:
                            param_value = str(exif_data[param])
                    except Exception:
                        param_value = "NA"
                    exif_params.append(param_value)

        # loop through detections
        if 'detections' in image:
            for detection in image['detections']:
                conf = detection["conf"]

                if manually_checked:
                    max_detection_conf = "NA"  # type: ignore[assignment]
                elif conf > max_detection_conf:
                    max_detection_conf = conf

                if conf >= thresh:
                    if manually_checked:
                        conf = "NA"  # type: ignore[assignment]

                    category = detection["category"]
                    label = label_map[category]
                    if sep:
                        unique_labels.append(label)
                        unique_labels = sorted(list(set(unique_labels)))

                    if vis or crp or exp:
                        if data_type == "img":
                            height, width = im_to_vis.shape[:2]
                        else:
                            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))

                        w_box = detection['bbox'][2]
                        h_box = detection['bbox'][3]
                        xo = detection['bbox'][0] + (w_box / 2)
                        yo = detection['bbox'][1] + (h_box / 2)
                        left = int(round(detection['bbox'][0] * width))
                        top = int(round(detection['bbox'][1] * height))
                        right = int(round(w_box * width)) + left
                        bottom = int(round(h_box * height)) + top

                        bbox_info.append([label, conf, manually_checked, left, top,
                                          right, bottom, height, width, xo, yo,
                                          w_box, h_box])

        # collect info to append to csv files
        if exp:
            if data_type == "img":
                img = cv2.imread(os.path.normpath(os.path.join(src_dir, file)))
                height, width = img.shape[:2]
            else:
                cap = cv2.VideoCapture(os.path.join(src_dir, file))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

            row = pd.DataFrame([[src_dir, file, data_type, len(bbox_info), height,
                                  width, max_detection_conf, manually_checked,
                                  *exif_params]])
            row.to_csv(csv_for_files, encoding='utf-8', mode='a', index=False,
                       header=False)

            rows = []
            for bbox in bbox_info:
                row = [src_dir, file, data_type, *bbox[:9], *exif_params]  # type: ignore[assignment]
                rows.append(row)
            rows = pd.DataFrame(rows)
            rows.to_csv(csv_for_detections, encoding='utf-8', mode='a', index=False,
                        header=False)

            if sep:
                if n_detections == 0:
                    detection_type = "empty"
                else:
                    if len(unique_labels) > 1:
                        detection_type = "_".join(unique_labels)
                    elif len(unique_labels) == 0:
                        detection_type = "empty"
                    else:
                        detection_type = label  # type: ignore[assignment]

                should_keep_series = False
                if keep_series:
                    if keep_series_species is None:
                        keep_series_species = []

                    labels_in_detection = set(unique_labels) if n_detections > 0 else set()

                    try:
                        cur_model_vars = load_model_vars_for(base_path, "cls",
                                                             cls_model_name)
                        cur_model_classes = set(cur_model_vars.get("all_classes", []) or [])
                    except Exception:
                        cur_model_classes = set()

                    keep_series_species_effective = [
                        c for c in keep_series_species if c in cur_model_classes]
                    keep_series_species_effective_set = set(keep_series_species_effective)

                    if keep_series_species_effective:
                        should_keep_series = bool(
                            labels_in_detection.intersection(
                                keep_series_species_effective_set))
                    else:
                        should_keep_series = bool(
                            labels_in_detection.difference({'person', 'vehicle'}))

                if should_keep_series:
                    series_files = find_series_images(
                        file, timestamp_index, window_seconds=keep_series_seconds)
                    for sf in series_files:
                        if sf in already_moved_files:
                            continue
                        moved_rel = move_files(sf, detection_type, file_placement,
                                               max_detection_conf, sep_conf,
                                               dst_dir, src_dir, manually_checked)
                        already_moved_files.add(sf)
                        if sf == file:
                            file = moved_rel
                else:
                    if file not in already_moved_files:
                        orig_file_for_move = file
                        file = move_files(orig_file_for_move, detection_type,
                                          file_placement, max_detection_conf,
                                          sep_conf, dst_dir, src_dir, manually_checked)
                        already_moved_files.add(orig_file_for_move)

        # visualize images
        if vis and len(bbox_info) > 0:
            if vis_blur:
                for bbox in bbox_info:
                    if bbox[0] == "person":
                        im_to_vis = blur_box(im_to_vis, *bbox[3:7], bbox[8], bbox[7])

            if vis_bbox:
                for bbox in bbox_info:
                    if manually_checked:
                        vis_label = f"{bbox[0]} (verified)"
                    else:
                        conf_label = round(bbox[1], 2) if round(bbox[1], 2) != 1.0 else 0.99
                        vis_label = f"{bbox[0]} {conf_label}"
                    color = colors[int(inverted_label_map[bbox[0]])]
                    bb.add(im_to_vis, *bbox[3:7], vis_label, color, size=vis_size_idx)

            im = os.path.join(dst_dir, file)
            Path(os.path.dirname(im)).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(im, im_to_vis)

            if exif is not None:
                image_new = PIL.Image.open(im)
                image_new.save(im, exif=exif)
                image_new.close()

        # crop images
        if crp and len(bbox_info) > 0:
            counter = 1
            for bbox in bbox_info:
                if sep:
                    im_to_crp = PIL.Image.open(os.path.join(dst_dir, file))
                else:
                    im_to_crp = PIL.Image.open(im_to_crop_path)
                crp_im = im_to_crp.crop((bbox[3:7]))
                im_to_crp.close()
                filename, file_extension = os.path.splitext(file)
                im_path = os.path.join(dst_dir,
                                       filename + '_crop' + str(counter) + '_' +
                                       bbox[0] + file_extension)
                Path(os.path.dirname(im_path)).mkdir(parents=True, exist_ok=True)
                crp_im.save(im_path)
                counter += 1

                if exif is not None:
                    image_new = PIL.Image.open(im_path)
                    image_new.save(im_path, exif=exif)
                    image_new.close()

        # calculate stats and emit progress via event bus
        elapsed_time_sep = str(datetime.timedelta(
            seconds=round(time.time() - start_time)))
        time_left_sep = str(datetime.timedelta(
            seconds=round(((time.time() - start_time) * n_images / nloop) -
                           (time.time() - start_time))))
        percentage = (nloop / n_images) * 100
        event_bus.emit(POSTPROCESS_PROGRESS, pct=float(percentage),
                       message=f"Processing: {nloop}/{n_images}",
                       process=f"{data_type}_pst", status="running",
                       cur_it=nloop, tot_it=n_images,
                       time_ela=elapsed_time_sep, time_rem=time_left_sep,
                       cancel_func=_cancel_func)
        nloop += 1
        _update_ui()

    # create summary csv
    if exp:
        csv_for_summary = os.path.join(dst_dir, "results_summary.csv")
        if os.path.exists(csv_for_summary):
            os.remove(csv_for_summary)
        det_info = pd.DataFrame(pd.read_csv(csv_for_detections,
                                            dtype=_POSTPROCESS_DTYPES,
                                            low_memory=False))
        summary = pd.DataFrame(
            det_info.groupby(['label', 'data_type']).size()
            .sort_values(ascending=False)
            .reset_index(name='n_detections'))
        summary.to_csv(csv_for_summary, encoding='utf-8', mode='w',
                       index=False, header=True)

    # convert csv to xlsx if required
    if exp and exp_format == t('dpd_exp_format')[0]:  # XLSX
        xlsx_path = os.path.join(dst_dir, "results.xlsx")

        dfs = []
        for result_type in ['detections', 'files', 'summary']:
            csv_path = os.path.join(dst_dir, f"results_{result_type}.csv")
            if os.path.isfile(xlsx_path):
                df_xlsx = pd.read_excel(xlsx_path, sheet_name=result_type)
                df_csv = pd.read_csv(os.path.join(dst_dir,
                                                   f"results_{result_type}.csv"),
                                     dtype=_POSTPROCESS_DTYPES, low_memory=False)
                df = pd.concat([df_xlsx, df_csv], ignore_index=True)
            else:
                df = pd.read_csv(os.path.join(dst_dir, f"results_{result_type}.csv"),
                                 dtype=_POSTPROCESS_DTYPES, low_memory=False)
            dfs.append(df)

            if not plt:
                if os.path.isfile(csv_path):
                    os.remove(csv_path)

        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            for idx, result_type in enumerate(['detections', 'files', 'summary']):
                df = dfs[idx]
                if result_type in ['detections', 'files']:
                    df['DateTimeOriginal'] = pd.to_datetime(
                        df['DateTimeOriginal'], format='%d/%m/%y %H:%M:%S')
                    df['DateTime'] = pd.to_datetime(
                        df['DateTime'], format='%d/%m/%y %H:%M:%S')
                    df['DateTimeDigitized'] = pd.to_datetime(
                        df['DateTimeDigitized'], format='%d/%m/%y %H:%M:%S')
                df.to_excel(writer, sheet_name=result_type, index=None, header=True)

    # convert csv to tsv if required
    if exp and exp_format == t('dpd_exp_format')[3]:  # TSV
        csv_path = os.path.join(dst_dir, "results_detections.csv")
        tsv_path = os.path.join(dst_dir, "results_sensing_clues.tsv")

        if os.path.isfile(tsv_path):
            with open(csv_path, 'r', newline='') as csv_file, \
                    open(tsv_path, 'a', newline='') as tsv_file:
                csv_reader = csv.reader(x.replace('\0', '') for x in csv_file)
                tsv_writer = csv.writer(tsv_file, delimiter='\t')
                csv_header = next(csv_reader)
                idx_date, idx_lat, idx_lon = map(
                    csv_header.index,
                    ["DateTimeOriginal", "Latitude", "Longitude"])
                for row in csv_reader:
                    unique_id = generate_unique_id(row)
                    formatted_date = format_datetime(row[idx_date])
                    new_row = [unique_id, formatted_date, row[idx_lat],
                               row[idx_lon], "AddaxAI"] + row
                    tsv_writer.writerow(new_row)
        else:
            with open(csv_path, 'r', newline='') as csv_file, \
                    open(tsv_path, 'w', newline='') as tsv_file:
                csv_reader = csv.reader(x.replace('\0', '') for x in csv_file)
                tsv_writer = csv.writer(tsv_file, delimiter='\t')
                csv_header = next(csv_reader)
                idx_date, idx_lat, idx_lon = map(
                    csv_header.index,
                    ["DateTimeOriginal", "Latitude", "Longitude"])
                tsv_writer.writerow(["ID", "Date", "Lat", "Long", "Method"] +
                                    csv_header)
                for row in csv_reader:
                    unique_id = generate_unique_id(row)
                    formatted_date = format_datetime(row[idx_date])
                    new_row = [unique_id, formatted_date, row[idx_lat],
                               row[idx_lon], "AddaxAI"] + row
                    tsv_writer.writerow(new_row)

        if not plt:
            for result_type in ['detections', 'files', 'summary']:
                csv_path = os.path.join(dst_dir, f"results_{result_type}.csv")
                if os.path.isfile(csv_path):
                    os.remove(csv_path)

    # convert csv to coco format if required
    if exp and exp_format == t('dpd_exp_format')[2]:  # COCO
        import pandas as pd  # already imported above, but silences type checkers
        coco_path = os.path.join(dst_dir, f"results_coco_{data_type}.json")
        detections_df = pd.read_csv(
            os.path.join(dst_dir, "results_detections.csv"),
            dtype=_POSTPROCESS_DTYPES, low_memory=False)
        files_df = pd.read_csv(
            os.path.join(dst_dir, "results_files.csv"),
            dtype=_POSTPROCESS_DTYPES, low_memory=False)

        csv_to_coco(
            detections_df=detections_df,
            files_df=files_df,
            output_path=coco_path,
            version=str(current_version),
        )

        if not plt:
            for result_type in ['detections', 'files', 'summary']:
                csv_path = os.path.join(dst_dir, f"results_{result_type}.csv")
                if os.path.isfile(csv_path):
                    os.remove(csv_path)

    # change json paths back if converted earlier
    if json_paths_converted:
        make_json_absolute(recognition_file, src_dir)

    # let the user know it's done via event bus
    event_bus.emit(POSTPROCESS_PROGRESS, pct=100.0, message="Postprocessing complete",
                   process=f"{data_type}_pst", status="done")
    _update_ui()

    # create graphs
    if plt:
        if produce_plots_func is not None:
            produce_plots_func(dst_dir)

        if (exp and exp_format == t('dpd_exp_format')[0]) or \
                (exp and exp_format == t('dpd_exp_format')[2]) or \
                (exp and exp_format == t('dpd_exp_format')[3]) or \
                remove_csv:
            for result_type in ['detections', 'files', 'summary']:
                csv_path = os.path.join(dst_dir, f"results_{result_type}.csv")
                if os.path.isfile(csv_path):
                    os.remove(csv_path)


# ============================================================================
# run_postprocess  (B8b — extracted start_postprocess() wrapper)
# ============================================================================

def run_postprocess(
    config,              # type: PostprocessConfig
    callbacks,           # type: OrchestratorCallbacks
    cancel_func,         # type: Callable[[], None]
    produce_plots_func=None,   # type: Optional[Callable[[str], None]]
    base_path="",        # type: str
    cls_model_name="",   # type: str
):
    # type: (...) -> PostprocessResult
    """Run postprocessing on detection results.

    Validates inputs, emits POSTPROCESS_STARTED/FINISHED/ERROR events, and
    calls _postprocess_inner() for each data type present (img and/or vid).

    Args:
        config:             Postprocessing configuration.
        callbacks:          Injected callbacks for UI interaction.
        cancel_func:        Zero-argument callable that cancels postprocessing;
                            forwarded to progress events for the UI cancel button.
        produce_plots_func: Optional callable that generates plots, called as
                            produce_plots_func(results_dir).  Pass None to skip.
        base_path:          AddaxAI_files root, used for keep_series model lookup.
        cls_model_name:     Classification model name for keep_series filtering.

    Returns:
        PostprocessResult — check .success and .error_code for outcome.
    """
    logger.debug("EXECUTED: run_postprocess")

    event_bus.emit(POSTPROCESS_STARTED)

    src_dir = config.source_folder
    dst_dir = config.dest_folder

    # check which json files are present
    img_json = os.path.isfile(os.path.join(src_dir, "image_recognition_file.json"))
    vid_json = os.path.isfile(os.path.join(src_dir, "video_recognition_file.json"))

    if not img_json and not vid_json:
        event_bus.emit(POSTPROCESS_ERROR, message="No model output found")
        callbacks.on_error(t('error'), t('msg_no_model_output'))
        return PostprocessResult(success=False, error_code="no_json",
                                 error_message="No recognition files found")

    # check if destination dir is valid
    if dst_dir in ("", "/", "\\", ".", "~", ":") or not os.path.isdir(dst_dir):
        event_bus.emit(POSTPROCESS_ERROR, message="Destination folder not set or invalid")
        callbacks.on_error(
            t('msg_dest_folder_not_set'),
            ["Destination folder not set.\n\n You have not specified where the "
             "post-processing results should be placed or the set folder does not "
             "exist. This is required.",
             "Carpeta de destino no establecida. No ha especificado dónde deben "
             "colocarse los resultados del postprocesamiento o la carpeta establecida "
             "no existe. Esto opción es obligatoria.",
             "Le répertoire de sortie n'est pas spécifié. Vous n'avez pas spécifié "
             "l'emplacement où enregistrer les résultats du post-traitement ou le "
             "répertoire n'existe pas. Ceci est obligatoire."
             ][config.lang_idx],
        )
        return PostprocessResult(success=False, error_code="invalid_dest",
                                 error_message="Destination folder not set or invalid")

    # warn user if original files will be overwritten with visualized files
    if (os.path.normpath(dst_dir) == os.path.normpath(src_dir) and
            config.vis and not config.separate_files):
        if not callbacks.on_confirm(
            t('msg_original_images_overwritten'),
            [f"WARNING! The visualized images will be placed in the folder with the "
             f"original data: '{src_dir}'. By doing this, you will overwrite the "
             f"original images with the visualized ones. Visualizing is permanent and "
             f"cannot be undone. Are you sure you want to continue?",
             f"ATENCIÓN. Las imágenes visualizadas se colocarán en la carpeta con los "
             f"datos originales: '{src_dir}'. Al hacer esto, se sobrescribirán las "
             f"imágenes originales con las visualizadas. La visualización es permanente "
             f"y no se puede deshacer. ¿Está seguro de que desea continuar?",
             f"ATTENTION ! Les images visualisées seront placées dans le dossier "
             f"contenant les données d'origine : « {src_dir} ». Ce faisant, vous "
             f"écraserez les images d'origine par celles visualisées. La visualisation "
             f"est définitive et irréversible. Voulez-vous vraiment continuer ? "
             ][config.lang_idx],
        ):
            return PostprocessResult(success=False, error_code="cancelled",
                                     error_message="User cancelled: original images overwrite")

    # warn user if images will be moved and visualized
    if config.separate_files and config.file_placement == 1 and config.vis:
        if not callbacks.on_confirm(
            t('msg_original_images_overwritten'),
            [f"WARNING! You specified to visualize the original images. Visualizing "
             f"is permanent and cannot be undone. If you don't want to visualize the "
             f"original images, please select 'Copy' as '{t('lbl_file_placement')}'. "
             f"Are you sure you want to continue with the current settings?",
             f"ATENCIÓN. Ha especificado visualizar las imágenes originales. La "
             f"visualización es permanente y no puede deshacerse. Si no desea visualizar "
             f"las imágenes originales, seleccione 'Copiar' como "
             f"'{t('lbl_file_placement')}'. ¿Está seguro de que desea continuar con "
             f"la configuración actual?",
             f"ATTENTION ! Vous avez spécifié de visualiser les images originales. La "
             f"visualisation est définitive et irréversible. Si vous ne souhaitez pas "
             f"visualiser les images originales, sélectionnez « Copier » au format "
             f"« {t('lbl_file_placement')} ». Voulez-vous vraiment conserver les "
             f"paramètres actuels ?"
             ][config.lang_idx],
        ):
            return PostprocessResult(success=False, error_code="cancelled",
                                     error_message="User cancelled: move+visualize warning")

    # build shared kwargs for _postprocess_inner
    inner_kwargs = dict(
        dst_dir=dst_dir,
        thresh=config.thresh,
        sep=config.separate_files,
        keep_series=config.keep_series,
        keep_series_seconds=config.keep_series_seconds,
        file_placement=config.file_placement,
        sep_conf=config.sep_conf,
        vis=config.vis,
        crp=config.crp,
        exp=config.exp,
        plt=config.plt,
        exp_format=config.exp_format,
        keep_series_species=config.keep_series_species,
        vis_blur=config.vis_blur,
        vis_bbox=config.vis_bbox,
        vis_size_idx=config.vis_size_idx,
        cancel_check=callbacks.cancel_check,
        update_ui=callbacks.update_ui,
        cancel_func=cancel_func,
        produce_plots_func=produce_plots_func,
        on_confirm=callbacks.on_confirm,
        on_error=callbacks.on_error,
        current_version=config.current_version,
        lang_idx=config.lang_idx,
        base_path=base_path,
        cls_model_name=cls_model_name,
    )

    try:
        if img_json:
            _postprocess_inner(src_dir, data_type="img", **inner_kwargs)

        if vid_json and not callbacks.cancel_check():
            _postprocess_inner(src_dir, data_type="vid", **inner_kwargs)

        # check if there were postprocessing errors
        postprocessing_error_log = os.path.join(dst_dir, "postprocessing_error_log.txt")
        if os.path.isfile(postprocessing_error_log):
            callbacks.on_warning(
                t('warning'),
                [f"One or more files failed to be analysed by the model (e.g., corrupt "
                 f"files) and will be skipped by post-processing features. See\n\n"
                 f"'{postprocessing_error_log}'\n\nfor more info.",
                 f"Uno o más archivos no han podido ser analizados por el modelo (por "
                 f"ejemplo, ficheros corruptos) y serán omitidos por las funciones de "
                 f"post-procesamiento. Para más información, véase\n\n"
                 f"'{postprocessing_error_log}'",
                 f"Un ou plusieurs fichiers n'ont pas pu être analysés par le modèle "
                 f"(par exemple, des fichiers corrompus) et seront ignorés lors du "
                 f"post-traitement. Voir\n\n'{postprocessing_error_log}'\n\npour plus "
                 f"d'info."
                 ][config.lang_idx],
            )

        event_bus.emit(POSTPROCESS_FINISHED)
        return PostprocessResult(success=True, error_code=None, error_message=None)

    except Exception as error:
        logger.error("ERROR: %s", error, exc_info=True)
        event_bus.emit(POSTPROCESS_ERROR, message=str(error))
        callbacks.on_error(
            t('error'),
            t('an_error_occurred') + " (AddaxAI v" + config.current_version + "): '" +
            str(error) + "'.",
        )
        return PostprocessResult(success=False, error_code="exception",
                                 error_message=str(error))
