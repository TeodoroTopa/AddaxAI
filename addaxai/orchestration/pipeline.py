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

import dataclasses
import json
import logging
import os
import platform
import subprocess
from pathlib import Path
from subprocess import Popen
from typing import Any, Callable, Dict, List, Optional

from addaxai.core.config import load_model_vars_for
from addaxai.core.event_types import (
    DEPLOY_CANCELLED,
    DEPLOY_FINISHED,
    DEPLOY_PROGRESS,
    DEPLOY_STARTED,
    DEPLOY_ERROR,
)
from addaxai.core.events import event_bus
from addaxai.core.platform import get_python_interpreter
from addaxai.i18n import t
from addaxai.models.deploy import (
    extract_label_map_from_model,
    imitate_object_detection_for_full_image_classifier,
    switch_yolov5_version,
)
from addaxai.orchestration.callbacks import OrchestratorCallbacks
from addaxai.orchestration.context import DeployConfig
from addaxai.orchestration.stdout_parser import parse_detection_stdout
from addaxai.utils.json_ops import append_to_json, make_json_absolute

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
