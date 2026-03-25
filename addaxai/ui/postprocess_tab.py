"""Postprocessing UI tab — encapsulates postprocessing workflow UI components.

This module provides the PostprocessTab class which manages the postprocessing workflow UI,
including button controls and progress display. The class implements the PostprocessView
protocol to abstract the UI from orchestration logic.
"""

import logging
from typing import Any, Callable, Dict, Optional

from addaxai.core.events import event_bus
from addaxai.core.event_types import (
    POSTPROCESS_PROGRESS,
    POSTPROCESS_FINISHED,
    POSTPROCESS_ERROR,
)

logger = logging.getLogger(__name__)


class PostprocessTab:
    """Manages the postprocessing workflow UI.

    Implements the PostprocessView protocol to provide a clean interface between
    the postprocessing orchestration logic (in app.py) and the UI widgets.

    Event handlers receive progress data from orchestrators via the event bus
    and forward it transparently to ProgressWindow.update_values(). The events
    carry the exact same kwargs that update_values() expects (process, status,
    cur_it, tot_it, time_ela, time_rem, cancel_func) so that no information
    is lost in translation.
    """

    def __init__(
        self,
        parent_frame: Any,
        app_state: Optional[Any] = None,
    ) -> None:
        """Initialize the postprocess tab UI.

        Args:
            parent_frame: The parent tkinter frame to add widgets to.
            app_state: Reference to AppState instance for state management.
        """
        self.parent_frame = parent_frame
        self.app_state = app_state
        self._progress_label_ref: Optional[Any] = None

        # Frame references (populated by build_widgets)
        self.sep_frame: Optional[Any] = None
        self.keep_series_frame: Optional[Any] = None
        self.vis_frame: Optional[Any] = None
        self.exp_frame: Optional[Any] = None
        self.lbl_vis_files: Optional[Any] = None
        self.lbl_exp: Optional[Any] = None

        # Subscribe to postprocessing events
        event_bus.on(POSTPROCESS_PROGRESS, self._on_postprocess_progress)
        event_bus.on(POSTPROCESS_ERROR, self._on_postprocess_error)
        event_bus.on(POSTPROCESS_FINISHED, self._on_postprocess_finished)

    def build_widgets(
        self,
        global_vars: Dict[str, Any],
        var_output_dir: Any,
        var_output_dir_short: Any,
        var_separate_files: Any,
        var_file_placement: Any,
        var_sep_conf: Any,
        var_keep_series: Any,
        var_keep_series_seconds: Any,
        var_vis_files: Any,
        var_vis_bbox: Any,
        var_vis_size: Any,
        var_vis_blur: Any,
        var_crp_files: Any,
        var_plt: Any,
        var_exp: Any,
        var_exp_format: Any,
        var_thresh: Any,
        green_primary: str,
        text_font: str,
        label_width: float,
        widget_width: float,
        subframe_correction_factor: float,
        first_level_frame_font_size: int,
        second_level_frame_font_size: int,
        t_func: Callable[[str], Any],
        browse_dir_callback: Callable[..., None],
        toggle_sep_frame_callback: Callable[[], None],
        toggle_keep_series_frame_callback: Callable[[], None],
        toggle_vis_frame_callback: Callable[[], None],
        toggle_exp_frame_callback: Callable[[], None],
        open_keep_series_species_callback: Callable[[], None],
        start_postprocess_callback: Callable[[], None],
    ) -> None:
        """Build all postprocessing workflow widgets.

        Args:
            global_vars: Global configuration dictionary.
            var_*: Tkinter variables for widget state.
            green_primary: Primary green color for UI.
            text_font: Font family name.
            label_width: Width of label columns.
            widget_width: Width of widget columns.
            subframe_correction_factor: Adjustment for nested frame widths.
            first_level_frame_font_size: Font size for main frames.
            second_level_frame_font_size: Font size for nested frames.
            t_func: Translation function t(key).
            browse_dir_callback: Callback for browse button.
            toggle_*_callback: Callbacks for checkbox toggles.
            start_postprocess_callback: Callback for start button.
        """
        # Avoid re-importing tkinter widgets at module level to support unit tests.
        # Use getattr to dynamically access tkinter classes from parent frame's module.
        try:
            import tkinter as tk
            from tkinter import Label, Button, Checkbutton, Radiobutton, LabelFrame, Scale, OptionMenu
            from tkinter import HORIZONTAL
        except ImportError:
            # Graceful fallback for environments without tkinter (unit tests)
            return

        # folder for results
        row_output_dir = 0
        lbl_output_dir = Label(master=self.parent_frame, text=t_func('lbl_output_dir'), width=1, anchor="w")
        lbl_output_dir.grid(row=row_output_dir, sticky='nesw', pady=2)
        var_output_dir.set("")
        dsp_output_dir = Label(master=self.parent_frame, textvariable=var_output_dir_short, fg=green_primary)
        btn_output_dir = Button(
            master=self.parent_frame,
            text=t_func('browse'),
            width=1,
            command=lambda: browse_dir_callback(var_output_dir, var_output_dir_short, dsp_output_dir, 25, row_output_dir, 0, 'e')
        )
        btn_output_dir.grid(row=row_output_dir, column=1, sticky='nesw', padx=5)

        # separate files
        row_separate_files = 1
        lbl_separate_files = Label(self.parent_frame, text=t_func('lbl_separate_files'), width=1, anchor="w")
        lbl_separate_files.grid(row=row_separate_files, sticky='nesw', pady=2)
        var_separate_files.set(global_vars['var_separate_files'])
        chb_separate_files = Checkbutton(self.parent_frame, variable=var_separate_files, command=toggle_sep_frame_callback, anchor="w")
        chb_separate_files.grid(row=row_separate_files, column=1, sticky='nesw', padx=5)

        # separation frame
        sep_frame_row = 2
        self.sep_frame = LabelFrame(
            self.parent_frame,
            text=" ↳ " + t_func('sep_frame') + " ",
            pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="grey80"
        )  # type: ignore[arg-type]
        self.sep_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
        self.sep_frame.grid(row=sep_frame_row, column=0, columnspan=2, sticky='ew')
        self.sep_frame.columnconfigure(0, weight=1, minsize=label_width - subframe_correction_factor)
        self.sep_frame.columnconfigure(1, weight=1, minsize=widget_width - subframe_correction_factor)
        self.sep_frame.grid_forget()

        # method of file placement
        row_file_placement = 0
        lbl_file_placement = Label(self.sep_frame, text="     " + t_func('lbl_file_placement'), pady=2, width=1, anchor="w")
        lbl_file_placement.grid(row=row_file_placement, sticky='nesw')
        var_file_placement.set(global_vars['var_file_placement'])
        rad_file_placement_move = Radiobutton(self.sep_frame, text=t_func('copy'), variable=var_file_placement, value=2)
        rad_file_placement_move.grid(row=row_file_placement, column=1, sticky='w', padx=5)
        rad_file_placement_copy = Radiobutton(self.sep_frame, text=t_func('move'), variable=var_file_placement, value=1)
        rad_file_placement_copy.grid(row=row_file_placement, column=1, sticky='e', padx=5)

        # separate per confidence
        row_sep_conf = 1
        lbl_sep_conf = Label(self.sep_frame, text="     " + t_func('lbl_sep_conf'), width=1, anchor="w")
        lbl_sep_conf.grid(row=row_sep_conf, sticky='nesw', pady=2)
        var_sep_conf.set(global_vars['var_sep_conf'])
        chb_sep_conf = Checkbutton(self.sep_frame, variable=var_sep_conf, anchor="w")
        chb_sep_conf.grid(row=row_sep_conf, column=1, sticky='nesw', padx=5)

        # keep series files (only affects separation)
        row_keep_series = 2
        lbl_keep_series = Label(self.sep_frame, text="     " + t_func('lbl_keep_series'), width=1, anchor="w")
        lbl_keep_series.grid(row=row_keep_series, sticky='nesw', pady=2)
        var_keep_series.set(global_vars['var_keep_series'])
        chb_keep_series = Checkbutton(self.sep_frame, variable=var_keep_series, command=toggle_keep_series_frame_callback, anchor="w")
        chb_keep_series.grid(row=row_keep_series, column=1, sticky='nesw', padx=5)

        # keep_series frame (nested under separation options)
        keep_series_frame_row = 3
        self.keep_series_frame = LabelFrame(
            self.sep_frame,
            text="        ↳ " + t_func('keep_series_frame') + " ",
            pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="grey80"
        )  # type: ignore[arg-type]
        self.keep_series_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
        self.keep_series_frame.grid(row=keep_series_frame_row, column=0, columnspan=2, sticky='ew')
        self.keep_series_frame.columnconfigure(0, weight=1, minsize=label_width - subframe_correction_factor)
        self.keep_series_frame.columnconfigure(1, weight=1, minsize=widget_width - subframe_correction_factor)
        self.keep_series_frame.grid_forget()

        # keep series seconds - only visible if keep series is checked
        row_keep_series_seconds = 0
        lbl_keep_series_seconds = Label(self.keep_series_frame, text="     " + t_func('lbl_keep_series_seconds'), width=1, anchor="w")
        lbl_keep_series_seconds.grid(row=row_keep_series_seconds, sticky='nesw', pady=2)
        var_keep_series_seconds.set(global_vars['var_keep_series_seconds'])
        chb_keep_series_seconds = Scale(
            self.keep_series_frame, from_=0.1, to=10, resolution=0.1, orient=HORIZONTAL,
            variable=var_keep_series_seconds, showvalue=0, width=10, length=1
        )  # type: ignore[arg-type]
        chb_keep_series_seconds.grid(row=row_keep_series_seconds, column=1, sticky='ew', padx=10)
        dsp_keep_series_seconds = Label(self.keep_series_frame, textvariable=var_keep_series_seconds)
        dsp_keep_series_seconds.configure(fg=green_primary)
        dsp_keep_series_seconds.grid(row=row_keep_series_seconds, column=0, sticky='e', padx=0)

        # keep series species - optional trigger filter
        row_keep_series_species = 1
        lbl_keep_series_species = Label(self.keep_series_frame, text="     " + t_func('lbl_keep_series_species'), width=1, anchor="w")
        lbl_keep_series_species.grid(row=row_keep_series_species, sticky='nesw', pady=2)

        # display: show how many triggers are selected (empty = any)
        dsp_keep_series_species = Label(self.keep_series_frame, fg=green_primary)
        if len(global_vars.get('var_keep_series_species', []) or []) == 0:
            dsp_keep_series_species.configure(text=t_func('any'))
        else:
            dsp_keep_series_species.configure(text=str(len(global_vars.get('var_keep_series_species', []))))
        dsp_keep_series_species.grid(row=row_keep_series_species, column=0, sticky='e', padx=0)

        btn_keep_series_species = Button(self.keep_series_frame, text=t_func('select'), command=open_keep_series_species_callback)
        btn_keep_series_species.grid(row=row_keep_series_species, column=1, sticky='w', padx=10)

        # visualize images
        row_vis_files = 3
        self.lbl_vis_files = Label(self.parent_frame, text=t_func('lbl_vis_files'), width=1, anchor="w")
        self.lbl_vis_files.grid(row=row_vis_files, sticky='nesw', pady=2)
        var_vis_files.set(global_vars['var_vis_files'])
        chb_vis_files = Checkbutton(self.parent_frame, variable=var_vis_files, anchor="w", command=toggle_vis_frame_callback)
        chb_vis_files.grid(row=row_vis_files, column=1, sticky='nesw', padx=5)

        # visualization options
        vis_frame_row = 4
        self.vis_frame = LabelFrame(
            self.parent_frame,
            text=" ↳ " + t_func('vis_frame') + " ",
            pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="grey80"
        )  # type: ignore[arg-type]
        self.vis_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
        self.vis_frame.grid(row=vis_frame_row, column=0, columnspan=2, sticky='ew')
        self.vis_frame.columnconfigure(0, weight=1, minsize=label_width - subframe_correction_factor)
        self.vis_frame.columnconfigure(1, weight=1, minsize=widget_width - subframe_correction_factor)
        self.vis_frame.grid_forget()

        # draw bboxes
        row_vis_bbox = 0
        lbl_vis_bbox = Label(self.vis_frame, text="     " + t_func('lbl_vis_bbox'), width=1, anchor="w")
        lbl_vis_bbox.grid(row=row_vis_bbox, sticky='nesw', pady=2)
        var_vis_bbox.set(global_vars['var_vis_bbox'])
        chb_vis_bbox = Checkbutton(self.vis_frame, variable=var_vis_bbox, anchor="w")
        chb_vis_bbox.grid(row=row_vis_bbox, column=1, sticky='nesw', padx=5)

        # line size
        row_vis_size = 1
        lbl_vis_size = Label(self.vis_frame, text="        ↳ " + t_func('lbl_vis_size'), pady=2, width=1, anchor="w")
        lbl_vis_size.grid(row=row_vis_size, sticky='nesw')
        var_vis_size.set(t_func('dpd_vis_size')[global_vars['var_vis_size_idx']])
        dpd_vis_size = OptionMenu(self.vis_frame, var_vis_size, *t_func('dpd_vis_size'))
        dpd_vis_size.configure(width=1)
        dpd_vis_size.grid(row=row_vis_size, column=1, sticky='nesw', padx=5)

        # blur people
        row_vis_blur = 2
        lbl_vis_blur = Label(self.vis_frame, text="     " + t_func('lbl_vis_blur'), width=1, anchor="w")
        lbl_vis_blur.grid(row=row_vis_blur, sticky='nesw', pady=2)
        var_vis_blur.set(global_vars['var_vis_blur'])
        chb_vis_blur = Checkbutton(self.vis_frame, variable=var_vis_blur, anchor="w")
        chb_vis_blur.grid(row=row_vis_blur, column=1, sticky='nesw', padx=5)

        # crop images
        row_crp_files = 5
        lbl_crp_files = Label(self.parent_frame, text=t_func('lbl_crp_files'), width=1, anchor="w")
        lbl_crp_files.grid(row=row_crp_files, sticky='nesw', pady=2)
        var_crp_files.set(global_vars['var_crp_files'])
        chb_crp_files = Checkbutton(self.parent_frame, variable=var_crp_files, anchor="w")
        chb_crp_files.grid(row=row_crp_files, column=1, sticky='nesw', padx=5)

        # plot
        row_plt = 6
        lbl_plt = Label(self.parent_frame, text=t_func('lbl_plt'), width=1, anchor="w")
        lbl_plt.grid(row=row_plt, sticky='nesw', pady=2)
        var_plt.set(global_vars['var_plt'])
        chb_plt = Checkbutton(self.parent_frame, variable=var_plt, anchor="w")
        chb_plt.grid(row=row_plt, column=1, sticky='nesw', padx=5)

        # export results
        row_exp = 7
        self.lbl_exp = Label(self.parent_frame, text=t_func('lbl_exp'), width=1, anchor="w")
        self.lbl_exp.grid(row=row_exp, sticky='nesw', pady=2)
        var_exp.set(global_vars['var_exp'])
        chb_exp = Checkbutton(self.parent_frame, variable=var_exp, anchor="w", command=toggle_exp_frame_callback)
        chb_exp.grid(row=row_exp, column=1, sticky='nesw', padx=5)

        # exportation options
        exp_frame_row = 8
        self.exp_frame = LabelFrame(
            self.parent_frame,
            text=" ↳ " + t_func('exp_frame') + " ",
            pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="grey80"
        )  # type: ignore[arg-type]
        self.exp_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
        self.exp_frame.grid(row=exp_frame_row, column=0, columnspan=2, sticky='ew')
        self.exp_frame.columnconfigure(0, weight=1, minsize=label_width - subframe_correction_factor)
        self.exp_frame.columnconfigure(1, weight=1, minsize=widget_width - subframe_correction_factor)
        self.exp_frame.grid_forget()

        # export format
        row_exp_format = 0
        lbl_exp_format = Label(self.exp_frame, text="     " + t_func('lbl_exp_format'), pady=2, width=1, anchor="w")
        lbl_exp_format.grid(row=row_exp_format, sticky='nesw')
        var_exp_format.set(t_func('dpd_exp_format')[global_vars['var_exp_format_idx']])
        dpd_exp_format = OptionMenu(self.exp_frame, var_exp_format, *t_func('dpd_exp_format'))
        dpd_exp_format.configure(width=1)
        dpd_exp_format.grid(row=row_exp_format, column=1, sticky='nesw', padx=5)

        # threshold
        row_lbl_thresh = 9
        lbl_thresh = Label(self.parent_frame, text=t_func('lbl_thresh'), width=1, anchor="w")
        lbl_thresh.grid(row=row_lbl_thresh, sticky='nesw', pady=2)
        var_thresh.set(global_vars['var_thresh'])
        scl_thresh = Scale(
            self.parent_frame, from_=0.01, to=1, resolution=0.01, orient=HORIZONTAL,
            variable=var_thresh, showvalue=0, width=10, length=1
        )  # type: ignore[arg-type]
        scl_thresh.grid(row=row_lbl_thresh, column=1, sticky='ew', padx=10)
        dsp_thresh = Label(self.parent_frame, textvariable=var_thresh)
        dsp_thresh.configure(fg=green_primary)
        dsp_thresh.grid(row=row_lbl_thresh, column=0, sticky='e', padx=0)

        # postprocessing button
        row_start_postprocess = 10
        btn_start_postprocess = Button(self.parent_frame, text=t_func('btn_start_postprocess'), command=start_postprocess_callback)
        btn_start_postprocess.grid(row=row_start_postprocess, column=0, columnspan=2, sticky='ew')

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
            logger.debug("Could not forward to progress_window", exc_info=True)

    def _on_postprocess_progress(self, **kwargs: Any) -> None:
        """Handle postprocessing progress event.

        Forwards all update_values()-compatible kwargs to ProgressWindow.
        """
        pct = kwargs.get('pct', 0.0)
        message = kwargs.get('message', '')
        self.show_progress(pct, message)
        self._forward_to_progress_window(**kwargs)

    def _on_postprocess_error(self, **kwargs: Any) -> None:
        """Handle postprocessing error event."""
        self.show_error(kwargs.get('message', ''))

    def _on_postprocess_finished(self, **kwargs: Any) -> None:
        """Handle postprocessing finished event."""
        self.show_completion({})

    def show_progress(self, pct: float, message: str) -> None:
        """Display postprocessing progress.

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
        # Error display is typically handled by messageboxes in orchestration logic
        pass

    def show_completion(self, summary: Dict[str, Any]) -> None:
        """Display completion with summary stats.

        Args:
            summary: Dictionary with summary statistics.
        """
        # Completion display is typically handled in orchestration logic
        pass

    def reset(self) -> None:
        """Reset the UI to initial state."""
        if self._progress_label_ref:
            self._progress_label_ref.configure(text="")
