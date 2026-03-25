"""Deployment UI tab — encapsulates deployment workflow UI components.

This module provides the DeployTab class which manages the deployment workflow UI,
including the deploy button, progress display, and related event handlers.
The class implements the DeployView protocol to abstract the UI from orchestration logic.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

try:
    from tkinter import Button
except ImportError:
    Button = None  # type: ignore

from addaxai.core.events import event_bus
from addaxai.core.event_types import (
    DEPLOY_PROGRESS,
    DEPLOY_FINISHED,
    DEPLOY_ERROR,
    CLASSIFY_PROGRESS,
    CLASSIFY_FINISHED,
    CLASSIFY_ERROR,
)
from addaxai.i18n import t

logger = logging.getLogger(__name__)


class DeployTab:
    """Manages the deployment workflow UI.

    Implements the DeployView protocol to provide a clean interface between
    the deployment orchestration logic (in app.py) and the UI widgets.

    Event handlers receive progress data from orchestrators via the event bus
    and forward it transparently to ProgressWindow.update_values(). The events
    carry the exact same kwargs that update_values() expects (process, status,
    cur_it, tot_it, time_ela, time_rem, speed, hware, cancel_func, etc.)
    so that no information is lost in translation.
    """

    def __init__(
        self,
        parent_frame: Any,  # tk.Frame
        start_deploy_callback: Callable[[], None],
        app_state: Optional[Any] = None,
    ) -> None:
        """Initialize the deploy tab UI.

        Args:
            parent_frame: The parent tkinter frame to add widgets to.
            start_deploy_callback: Callback function for the start deploy button.
            app_state: Reference to AppState instance for state management.
        """
        self.parent_frame = parent_frame
        self.start_deploy_callback = start_deploy_callback
        self.app_state = app_state
        self._button_ref: Optional[Any] = None
        self._progress_label_ref: Optional[Any] = None

        # Frame references (populated by build_widgets)
        self.cls_frame: Optional[Any] = None
        self.img_frame: Optional[Any] = None
        self.vid_frame: Optional[Any] = None

        # Subscribe to deployment and classification events
        event_bus.on(DEPLOY_PROGRESS, self._on_deploy_progress)
        event_bus.on(DEPLOY_ERROR, self._on_deploy_error)
        event_bus.on(DEPLOY_FINISHED, self._on_deploy_finished)
        event_bus.on(CLASSIFY_PROGRESS, self._on_classify_progress)
        event_bus.on(CLASSIFY_ERROR, self._on_classify_error)
        event_bus.on(CLASSIFY_FINISHED, self._on_classify_finished)

    def build_widgets(
        self,
        global_vars: Dict[str, Any],
        state: Any,
        dpd_options_model: List[List[str]],
        dpd_options_cls_model: List[List[str]],
        dpd_options_sppnet_location: List[str],
        var_det_model: Any,
        var_det_model_short: Any,
        var_det_model_path: Any,
        var_cls_model: Any,
        var_cls_detec_thresh: Any,
        var_cls_class_thresh: Any,
        var_smooth_cls_animal: Any,
        var_tax_fallback: Any,
        var_tax_levels: Any,
        var_sppnet_location: Any,
        var_exclude_subs: Any,
        var_use_custom_img_size_for_deploy: Any,
        var_image_size_for_deploy: Any,
        var_abs_paths: Any,
        var_disable_GPU: Any,
        var_process_img: Any,
        var_use_checkpnts: Any,
        var_checkpoint_freq: Any,
        var_cont_checkpnt: Any,
        var_process_vid: Any,
        var_not_all_frames: Any,
        var_nth_frame: Any,
        green_primary: str,
        text_font: str,
        label_width: float,
        widget_width: float,
        subframe_correction_factor: float,
        first_level_frame_font_size: int,
        second_level_frame_font_size: int,
        i18n_lang_idx: Callable[[], int],
        t_func: Callable[[str], Any],
        model_options_callback: Callable[..., None],
        model_cls_animal_options_callback: Callable[..., None],
        show_model_info_callback: Callable[[], None],
        open_species_selection_callback: Callable[[], None],
        on_chb_smooth_cls_animal_change_callback: Callable[..., None],
        toggle_tax_levels_callback: Callable[[], None],
        toggle_tax_levels_dpd_options_callback: Callable[[], None],
        taxon_mapping_csv_present_callback: Callable[[], bool],
        toggle_image_size_for_deploy_callback: Callable[[], None],
        image_size_for_deploy_focus_in_callback: Callable[..., None],
        abs_paths_warning_callback: Callable[[], None],
        toggle_img_frame_callback: Callable[[], None],
        checkpoint_freq_focus_in_callback: Callable[..., None],
        toggle_checkpoint_freq_callback: Callable[[], None],
        disable_chb_cont_checkpnt_callback: Callable[[], None],
        toggle_vid_frame_callback: Callable[[], None],
        nth_frame_focus_in_callback: Callable[..., None],
        fetch_known_models_callback: Callable[..., List[str]],
        load_model_vars_callback: Callable[[], Dict[str, Any]],
        write_model_vars_callback: Callable[..., None],
    ) -> None:
        """Build all deployment workflow widgets.

        This is a large method that constructs all detection and classification UI components.

        Args:
            global_vars: Global configuration dictionary.
            state: AppState instance for storing widget references.
            dpd_options_*: Lists of available options for dropdowns.
            var_*: Tkinter variables for widget state.
            green_primary: Primary green color for UI.
            text_font: Font family name.
            label_width: Width of label columns.
            widget_width: Width of widget columns.
            subframe_correction_factor: Adjustment for nested frame widths.
            first_level_frame_font_size: Font size for main frames.
            second_level_frame_font_size: Font size for nested frames.
            i18n_lang_idx: Function to get current language index.
            t_func: Translation function t(key).
            *_callback: Callback functions for various UI interactions.
            fetch_known_models_callback: Function to fetch available models.
            load_model_vars_callback: Function to load model variables.
            write_model_vars_callback: Function to write model variables.
        """
        # Avoid re-importing tkinter widgets at module level to support unit tests.
        try:
            import tkinter as tk
            from tkinter import Label, Button, Checkbutton, LabelFrame, Scale, OptionMenu, Entry, DISABLED, HORIZONTAL, NORMAL
        except ImportError:
            # Graceful fallback for environments without tkinter (unit tests)
            return

        # choose detector
        row_model = 0
        lbl_model = Label(master=self.parent_frame, text=t_func('lbl_model'), width=1, anchor="w")
        lbl_model.grid(row=row_model, sticky='nesw', pady=2)
        var_det_model.set(dpd_options_model[i18n_lang_idx()][global_vars["var_det_model_idx"]])
        var_det_model_short.set(global_vars["var_det_model_short"])
        var_det_model_path.set(global_vars["var_det_model_path"])
        dpd_model = OptionMenu(self.parent_frame, var_det_model, *dpd_options_model[i18n_lang_idx()], command=model_options_callback)
        dpd_model.configure(width=1)
        dpd_model.grid(row=row_model, column=1, sticky='nesw', padx=5)
        dsp_model = Label(master=self.parent_frame, textvariable=var_det_model_short, fg=green_primary)
        if var_det_model_short.get() != "":
            dsp_model.grid(column=0, row=row_model, sticky='e')

        # use classifier
        row_cls_model = 1
        lbl_cls_model = Label(self.parent_frame, text=t_func('lbl_cls_model'), width=1, anchor="w")
        lbl_cls_model.grid(row=row_cls_model, sticky='nesw', pady=2)
        var_cls_model.set(dpd_options_cls_model[i18n_lang_idx()][global_vars["var_cls_model_idx"]])
        dpd_cls_model = OptionMenu(self.parent_frame, var_cls_model, *dpd_options_cls_model[i18n_lang_idx()], command=model_cls_animal_options_callback)
        dpd_cls_model.configure(width=1, state=DISABLED)
        dpd_cls_model.grid(row=row_cls_model, column=1, sticky='nesw', padx=5, pady=2)

        # set global model vars for startup
        model_vars = load_model_vars_callback()

        # classification option frame (hidden by default)
        cls_frame_row = 2
        self.cls_frame = LabelFrame(self.parent_frame, text=" ↳ " + t_func('cls_frame') + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="black")  # type: ignore[arg-type]
        self.cls_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
        self.cls_frame.grid(row=cls_frame_row, column=0, columnspan=2, sticky='ew')
        self.cls_frame.columnconfigure(0, weight=1, minsize=label_width - subframe_correction_factor)
        self.cls_frame.columnconfigure(1, weight=1, minsize=widget_width - subframe_correction_factor)
        self.cls_frame.grid_forget()

        # show model info
        row_model_info = 0
        lbl_model_info = Label(master=self.cls_frame, text="     " + t_func('lbl_model_info'), width=1, anchor="w")
        lbl_model_info.grid(row=row_model_info, sticky='nesw', pady=2)
        btn_model_info = Button(master=self.cls_frame, text=t_func('show'), width=1, command=show_model_info_callback)
        btn_model_info.grid(row=row_model_info, column=1, sticky='nesw', padx=5)

        # choose classes
        row_choose_classes = 1
        lbl_choose_classes = Label(master=self.cls_frame, text="     " + t_func('lbl_choose_classes'), width=1, anchor="w")
        lbl_choose_classes.grid(row=row_choose_classes, sticky='nesw', pady=2)
        btn_choose_classes = Button(master=self.cls_frame, text=t_func('select'), width=1, command=open_species_selection_callback)
        btn_choose_classes.grid(row=row_choose_classes, column=1, sticky='nesw', padx=5)
        if var_cls_model.get() != t_func('none'):
            dsp_choose_classes = Label(self.cls_frame, text=f"{len(model_vars.get('selected_classes', []))} of {len(model_vars.get('all_classes', []))}")
        else:
            dsp_choose_classes = Label(self.cls_frame, text="")
        dsp_choose_classes.grid(row=row_choose_classes, column=0, sticky='e', padx=0)
        dsp_choose_classes.configure(fg=green_primary)

        # threshold to classify detections
        row_cls_detec_thresh = 2
        lbl_cls_detec_thresh = Label(self.cls_frame, text="     " + t_func('lbl_cls_detec_thresh'), width=1, anchor="w")
        lbl_cls_detec_thresh.grid(row=row_cls_detec_thresh, sticky='nesw', pady=2)
        var_cls_detec_thresh.set(model_vars.get('var_cls_detec_thresh', 0.6))
        scl_cls_detec_thresh = Scale(self.cls_frame, from_=0.01, to=1, resolution=0.01, orient=HORIZONTAL, variable=var_cls_detec_thresh, showvalue=0, width=10, length=1, state=DISABLED, command=lambda value: write_model_vars_callback(new_values={"var_cls_detec_thresh": str(value)}))  # type: ignore[arg-type]
        scl_cls_detec_thresh.grid(row=row_cls_detec_thresh, column=1, sticky='ew', padx=10)
        dsp_cls_detec_thresh = Label(self.cls_frame, textvariable=var_cls_detec_thresh)
        dsp_cls_detec_thresh.grid(row=row_cls_detec_thresh, column=0, sticky='e', padx=0)
        dsp_cls_detec_thresh.configure(fg=green_primary)

        # threshold accept identifications
        row_cls_class_thresh = 3
        lbl_cls_class_thresh = Label(self.cls_frame, text="     " + t_func('lbl_cls_class_thresh'), width=1, anchor="w")
        lbl_cls_class_thresh.grid(row=row_cls_class_thresh, sticky='nesw', pady=2)
        var_cls_class_thresh.set(model_vars.get('var_cls_class_thresh', 0.5))
        scl_cls_class_thresh = Scale(self.cls_frame, from_=0.01, to=1, resolution=0.01, orient=HORIZONTAL, variable=var_cls_class_thresh, showvalue=0, width=10, length=1, state=DISABLED, command=lambda value: write_model_vars_callback(new_values={"var_cls_class_thresh": value}))  # type: ignore[arg-type]
        scl_cls_class_thresh.grid(row=row_cls_class_thresh, column=1, sticky='ew', padx=10)
        dsp_cls_class_thresh = Label(self.cls_frame, textvariable=var_cls_class_thresh)
        dsp_cls_class_thresh.grid(row=row_cls_class_thresh, column=0, sticky='e', padx=0)
        dsp_cls_class_thresh.configure(fg=green_primary)

        # Smoothen results
        row_smooth_cls_animal = 4
        lbl_smooth_cls_animal = Label(self.cls_frame, text="     " + t_func('lbl_smooth_cls_animal'), width=1, anchor="w")
        lbl_smooth_cls_animal.grid(row=row_smooth_cls_animal, sticky='nesw', pady=2)
        var_smooth_cls_animal.set(model_vars.get('var_smooth_cls_animal', False))
        chb_smooth_cls_animal = Checkbutton(self.cls_frame, variable=var_smooth_cls_animal, anchor="w", command=on_chb_smooth_cls_animal_change_callback)
        chb_smooth_cls_animal.grid(row=row_smooth_cls_animal, column=1, sticky='nesw', padx=5)

        # taxonomic fallback checkbox (only visible if taxon mapping is present)
        row_tax_fallback = 5
        lbl_tax_fallback = Label(self.cls_frame, text="     " + t_func('lbl_tax_fallback'), width=1, anchor="w")
        var_tax_fallback.set(model_vars.get('var_tax_fallback', False))
        chb_tax_fallback = Checkbutton(self.cls_frame, variable=var_tax_fallback, anchor="w", command=toggle_tax_levels_callback)

        # taxonomic fallback dropdown (only visible if taxon mapping is present)
        row_tax_levels = 6
        lbl_tax_levels = Label(self.cls_frame, text="     " + t_func('lbl_tax_levels'), width=1, anchor="w")
        var_tax_levels.set('dummy')  # set dummy value to avoid error
        dpd_tax_levels = OptionMenu(self.cls_frame, var_tax_levels, ["dummy"])  # type: ignore[arg-type]
        dpd_tax_levels.configure(width=1, state=DISABLED)

        # make taxonomic fallback widgets visible if taxon mapping is present
        if taxon_mapping_csv_present_callback():
            toggle_tax_levels_dpd_options_callback()
            lbl_tax_fallback.grid(row=row_tax_fallback, sticky='nesw', pady=2)
            chb_tax_fallback.grid(row=row_tax_fallback, column=1, sticky='nesw', padx=5)
            toggle_tax_levels_callback()

        # choose location for species net
        row_sppnet_location = 1
        lbl_sppnet_location = Label(master=self.cls_frame, text="     " + t_func('lbl_sppnet_location'), width=1, anchor="w")
        lbl_sppnet_location.grid(row=row_sppnet_location, sticky='nesw', pady=2)
        var_sppnet_location.set(dpd_options_sppnet_location[global_vars["var_sppnet_location_idx"]])
        dpd_sppnet_location = OptionMenu(self.cls_frame, var_sppnet_location, *dpd_options_sppnet_location)
        dpd_sppnet_location.configure(width=1, state=DISABLED)

        # include subdirectories
        row_exclude_subs = 3
        lbl_exclude_subs = Label(self.parent_frame, text=t_func('lbl_exclude_subs'), width=1, anchor="w")
        lbl_exclude_subs.grid(row=row_exclude_subs, sticky='nesw', pady=2)
        var_exclude_subs.set(global_vars['var_exclude_subs'])
        chb_exclude_subs = Checkbutton(self.parent_frame, variable=var_exclude_subs, anchor="w")
        chb_exclude_subs.grid(row=row_exclude_subs, column=1, sticky='nesw', padx=5)

        # use custom image size
        row_use_custom_img_size_for_deploy = 4
        lbl_use_custom_img_size_for_deploy = Label(self.parent_frame, text=t_func('lbl_use_custom_img_size_for_deploy'), width=1, anchor="w")
        lbl_use_custom_img_size_for_deploy.grid(row=row_use_custom_img_size_for_deploy, sticky='nesw', pady=2)
        var_use_custom_img_size_for_deploy.set(global_vars['var_use_custom_img_size_for_deploy'])
        chb_use_custom_img_size_for_deploy = Checkbutton(self.parent_frame, variable=var_use_custom_img_size_for_deploy, command=toggle_image_size_for_deploy_callback, anchor="w")
        chb_use_custom_img_size_for_deploy.grid(row=row_use_custom_img_size_for_deploy, column=1, sticky='nesw', padx=5)

        # specify custom image size (not grid by default)
        row_image_size_for_deploy = 5
        lbl_image_size_for_deploy = Label(self.parent_frame, text=" ↳ " + t_func('lbl_image_size_for_deploy'), width=1, anchor="w")
        var_image_size_for_deploy.set(global_vars['var_image_size_for_deploy'])
        ent_image_size_for_deploy = Entry(self.parent_frame, textvariable=var_image_size_for_deploy, fg='grey', state=NORMAL, width=1)
        if var_image_size_for_deploy.get() == "":
            ent_image_size_for_deploy.insert(0, t_func('eg') + ": 640")
        else:
            ent_image_size_for_deploy.configure(fg='black')
        ent_image_size_for_deploy.bind("<FocusIn>", image_size_for_deploy_focus_in_callback)
        ent_image_size_for_deploy.configure(state=DISABLED)

        # use absolute paths
        row_abs_path = 6
        lbl_abs_paths = Label(self.parent_frame, text=t_func('lbl_abs_paths'), width=1, anchor="w")
        lbl_abs_paths.grid(row=row_abs_path, sticky='nesw', pady=2)
        var_abs_paths.set(global_vars['var_abs_paths'])
        chb_abs_paths = Checkbutton(self.parent_frame, variable=var_abs_paths, command=abs_paths_warning_callback, anchor="w")
        chb_abs_paths.grid(row=row_abs_path, column=1, sticky='nesw', padx=5)

        # use absolute paths (misnamed in original, actually GPU setting)
        row_disable_GPU = 7
        lbl_disable_GPU = Label(self.parent_frame, text=t_func('lbl_disable_GPU'), width=1, anchor="w")
        lbl_disable_GPU.grid(row=row_disable_GPU, sticky='nesw', pady=2)
        var_disable_GPU.set(global_vars['var_disable_GPU'])
        chb_disable_GPU = Checkbutton(self.parent_frame, variable=var_disable_GPU, anchor="w")
        chb_disable_GPU.grid(row=row_disable_GPU, column=1, sticky='nesw', padx=5)

        # process images
        row_process_img = 8
        lbl_process_img = Label(self.parent_frame, text=t_func('lbl_process_img'), width=1, anchor="w")
        lbl_process_img.grid(row=row_process_img, sticky='nesw', pady=2)
        var_process_img.set(global_vars['var_process_img'])
        chb_process_img = Checkbutton(self.parent_frame, variable=var_process_img, command=toggle_img_frame_callback, anchor="w")
        chb_process_img.grid(row=row_process_img, column=1, sticky='nesw', padx=5)

        # image option frame (hidden by default)
        img_frame_row = 9
        self.img_frame = LabelFrame(self.parent_frame, text=" ↳ " + t_func('img_frame') + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="grey80")  # type: ignore[arg-type]
        self.img_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
        self.img_frame.grid(row=img_frame_row, column=0, columnspan=2, sticky='ew')
        self.img_frame.columnconfigure(0, weight=1, minsize=label_width - subframe_correction_factor)
        self.img_frame.columnconfigure(1, weight=1, minsize=widget_width - subframe_correction_factor)
        self.img_frame.grid_forget()

        # use checkpoints
        row_use_checkpnts = 0
        lbl_use_checkpnts = Label(self.img_frame, text="     " + t_func('lbl_use_checkpnts'), pady=2, state=DISABLED, width=1, anchor="w")
        lbl_use_checkpnts.grid(row=row_use_checkpnts, sticky='nesw')
        var_use_checkpnts.set(global_vars['var_use_checkpnts'])
        chb_use_checkpnts = Checkbutton(self.img_frame, variable=var_use_checkpnts, command=toggle_checkpoint_freq_callback, state=DISABLED, anchor="w")
        chb_use_checkpnts.grid(row=row_use_checkpnts, column=1, sticky='nesw', padx=5)

        # checkpoint frequency
        row_checkpoint_freq = 1
        lbl_checkpoint_freq = Label(self.img_frame, text="        ↳ " + t_func('lbl_checkpoint_freq'), pady=2, state=DISABLED, width=1, anchor="w")
        lbl_checkpoint_freq.grid(row=row_checkpoint_freq, sticky='nesw')
        var_checkpoint_freq.set(global_vars['var_checkpoint_freq'])
        ent_checkpoint_freq = Entry(self.img_frame, textvariable=var_checkpoint_freq, fg='grey', state=NORMAL, width=1)
        ent_checkpoint_freq.grid(row=row_checkpoint_freq, column=1, sticky='nesw', padx=5)
        if var_checkpoint_freq.get() == "":
            ent_checkpoint_freq.insert(0, t_func('eg') + ": 10000")
        else:
            ent_checkpoint_freq.configure(fg='black')
        ent_checkpoint_freq.bind("<FocusIn>", checkpoint_freq_focus_in_callback)
        ent_checkpoint_freq.configure(state=DISABLED)

        # continue from checkpoint file
        row_cont_checkpnt = 2
        lbl_cont_checkpnt = Label(self.img_frame, text="     " + t_func('lbl_cont_checkpnt'), pady=2, state=DISABLED, width=1, anchor="w")
        lbl_cont_checkpnt.grid(row=row_cont_checkpnt, sticky='nesw')
        var_cont_checkpnt.set(global_vars['var_cont_checkpnt'])
        chb_cont_checkpnt = Checkbutton(self.img_frame, variable=var_cont_checkpnt, state=DISABLED, command=disable_chb_cont_checkpnt_callback, anchor="w")
        chb_cont_checkpnt.grid(row=row_cont_checkpnt, column=1, sticky='nesw', padx=5)

        # process videos
        row_process_vid = 10
        lbl_process_vid = Label(self.parent_frame, text=t_func('lbl_process_vid'), width=1, anchor="w")
        lbl_process_vid.grid(row=row_process_vid, sticky='nesw', pady=2)
        var_process_vid.set(global_vars['var_process_vid'])
        chb_process_vid = Checkbutton(self.parent_frame, variable=var_process_vid, command=toggle_vid_frame_callback, anchor="w")
        chb_process_vid.grid(row=row_process_vid, column=1, sticky='nesw', padx=5)

        # video option frame (hidden by default)
        vid_frame_row = 11
        self.vid_frame = LabelFrame(self.parent_frame, text=" ↳ " + t_func('vid_frame') + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="grey80")  # type: ignore[arg-type]
        self.vid_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
        self.vid_frame.grid(row=vid_frame_row, column=0, columnspan=2, sticky='ew')
        self.vid_frame.columnconfigure(0, weight=1, minsize=label_width - subframe_correction_factor)
        self.vid_frame.columnconfigure(1, weight=1, minsize=widget_width - subframe_correction_factor)
        self.vid_frame.grid_forget()

        # dont process all frames
        row_not_all_frames = 0
        lbl_not_all_frames = Label(self.vid_frame, text="     " + t_func('lbl_not_all_frames'), pady=2, state=DISABLED, width=1, anchor="w")
        lbl_not_all_frames.grid(row=row_not_all_frames, sticky='nesw')
        var_not_all_frames.set(global_vars['var_not_all_frames'])
        chb_not_all_frames = Checkbutton(self.vid_frame, variable=var_not_all_frames, command=toggle_nth_frame_callback, state=DISABLED, anchor="w")
        chb_not_all_frames.grid(row=row_not_all_frames, column=1, sticky='nesw', padx=5)

        # process every nth frame
        row_nth_frame = 1
        lbl_nth_frame = Label(self.vid_frame, text="        ↳ " + t_func('lbl_nth_frame'), pady=2, state=DISABLED, width=1, anchor="w")
        lbl_nth_frame.grid(row=row_nth_frame, sticky='nesw')
        var_nth_frame.set(global_vars['var_nth_frame'])
        ent_nth_frame = Entry(self.vid_frame, textvariable=var_nth_frame, fg='grey' if var_nth_frame.get().isdecimal() else 'black', state=NORMAL, width=1)
        ent_nth_frame.grid(row=row_nth_frame, column=1, sticky='nesw', padx=5)
        if var_nth_frame.get() == "":
            ent_nth_frame.insert(0, t_func('eg') + ": 1")
            ent_nth_frame.configure(fg='grey')
        else:
            ent_nth_frame.configure(fg='black')
        ent_nth_frame.bind("<FocusIn>", nth_frame_focus_in_callback)
        ent_nth_frame.configure(state=DISABLED)

    def _forward_to_progress_window(self, **kwargs: Any) -> None:
        """Forward event kwargs to progress_window.update_values().

        Extracts the kwargs that update_values() accepts and passes them
        through unchanged. Any extra kwargs (like 'pct', 'message') are
        ignored — they're for other consumers.
        """
        if not self.app_state or not hasattr(self.app_state, 'progress_window'):
            return
        pw = self.app_state.progress_window
        if pw is None:
            return

        # Build the kwargs dict with only keys update_values() accepts
        uv_kwargs = {}  # type: dict
        for key in ('process', 'status', 'cur_it', 'tot_it', 'time_ela',
                     'time_rem', 'speed', 'hware', 'cancel_func',
                     'extracting_frames_txt', 'frame_video_choice'):
            if key in kwargs:
                uv_kwargs[key] = kwargs[key]

        if 'process' not in uv_kwargs or 'status' not in uv_kwargs:
            return  # Can't call update_values without these two

        try:
            pw.update_values(**uv_kwargs)
        except (AttributeError, TypeError):
            # ProgressWindow may not have the widgets for this process type
            # (e.g. full-image classifier has no img_det frame)
            logger.debug("Could not forward to progress_window", exc_info=True)

    def _on_deploy_progress(self, **kwargs: Any) -> None:
        """Handle deployment progress event.

        Forwards all update_values()-compatible kwargs to ProgressWindow.
        """
        pct = kwargs.get('pct', 0.0)
        message = kwargs.get('message', '')
        self.show_progress(pct, message)
        self._forward_to_progress_window(**kwargs)

    def _on_deploy_error(self, **kwargs: Any) -> None:
        """Handle deployment error event."""
        self.show_error(kwargs.get('message', ''))

    def _on_deploy_finished(self, **kwargs: Any) -> None:
        """Handle deployment finished event."""
        self.show_completion(kwargs.get('results_path', ''))

    def _on_classify_progress(self, **kwargs: Any) -> None:
        """Handle classification progress event.

        Forwards all update_values()-compatible kwargs to ProgressWindow.
        """
        pct = kwargs.get('pct', 0.0)
        message = kwargs.get('message', '')
        self.show_progress(pct, message)
        self._forward_to_progress_window(**kwargs)

    def _on_classify_error(self, **kwargs: Any) -> None:
        """Handle classification error event."""
        self.show_error(kwargs.get('message', ''))

    def _on_classify_finished(self, **kwargs: Any) -> None:
        """Handle classification finished event."""
        self.show_completion(kwargs.get('results_path', ''))

    def create_button(self) -> Any:
        """Create and return the start deploy button.

        Returns:
            The created button widget.
        """
        if Button is None:
            raise ImportError("tkinter.Button not available")
        row = 12
        btn = Button(
            self.parent_frame,
            text=t("btn_start_deploy"),
            command=self.start_deploy_callback,
        )
        btn.grid(row=row, column=0, columnspan=2, sticky="ew")
        self._button_ref = btn
        return btn

    def set_button_ref(self, button_ref: Any) -> None:
        """Set reference to the deploy button (for integration with existing code).

        Args:
            button_ref: Reference to the button widget created externally.
        """
        self._button_ref = button_ref

    def get_button_ref(self) -> Optional[Any]:
        """Get reference to the deploy button.

        Returns:
            The button widget reference.
        """
        return self._button_ref

    def show_progress(self, pct: float, message: str) -> None:
        """Display deployment progress.

        Args:
            pct: Progress percentage (0-100).
            message: Progress message to display.
        """
        if self._progress_label_ref:
            self._progress_label_ref.configure(text=f"{pct}%: {message}")

    def show_error(self, message: str) -> None:
        """Display an error message.

        Args:
            message: Error message text.
        """
        # Error display is typically handled by messageboxes in the orchestration logic
        # This method is here for protocol compliance
        pass

    def show_completion(self, results_path: str) -> None:
        """Display completion message with results path.

        Args:
            results_path: Path to the results JSON file.
        """
        # Completion is typically handled in orchestration logic
        # This method is here for protocol compliance
        pass

    def set_model_list(self, models: List[str]) -> None:
        """Update the list of available detection models.

        Args:
            models: List of available model names.
        """
        # Model list updates are typically handled elsewhere
        # This method is here for protocol compliance
        pass

    def on_cancel(self, callback: Callable[[], None]) -> None:
        """Register a callback for cancel button press.

        Args:
            callback: Function to call when cancel is pressed.
        """
        # Cancel handling is integrated with the button command
        # This method is here for protocol compliance
        pass

    def reset(self) -> None:
        """Reset the UI to initial state."""
        if self._progress_label_ref:
            self._progress_label_ref.configure(text="")
        if self._button_ref:
            self._button_ref.configure(state="normal")
