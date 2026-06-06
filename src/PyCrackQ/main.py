import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import datetime
import math
import threading
import queue
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
import matplotlib
matplotlib.use("TkAgg")

from PyCrackQ.config import THEME_NAME, MIN_CRACK_AREA, JUNCTION_THRESH, DEFAULT_AUTO_MODE
from PyCrackQ.image_processing import (
    apply_binarization, apply_denoising, get_skeleton, get_distance_map,
    prune_spurs, detect_junctions,
    create_circular_mask, apply_circular_mask, recommend_parameters
)
from PyCrackQ.analysis import (
    trace_segment_euclidean_length, compute_precise_skeleton_length,
    calculate_accurate_metrics, calculate_segment_metrics,
    get_fractal_dim, calculate_branch_angles, analyze_soil_clods,
    analyze_crack_connectivity, classify_junctions
)
from PyCrackQ.visualization import (
    show_rose_plot, show_fractal_plot, show_histograms,
    show_clod_analysis, show_connectivity_analysis,
    show_junction_classification, open_enlarged_result_window,
    ZoomableImageFrame
)
from PyCrackQ.export import export_excel, export_csv, export_pdf, export_binary_image
from PyCrackQ.calibration import CalibrationWindow
from PyCrackQ.manual_edit import ManualEditWindow
from PyCrackQ.circular_region import CircularRegionWindow
from PyCrackQ.batch_processing import BatchProcessor
from PyCrackQ.batch_setup import BatchSetupWindow


class CreateToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() + 4
        y = self.widget.winfo_rooty()
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, font=("Microsoft YaHei", 9),
                         background="#ffffcc", relief="solid", borderwidth=1, padx=6, pady=2)
        label.pack()

    def _hide(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class BinarizationApp(ttk.Window):
    def __init__(self):
        super().__init__(themename=THEME_NAME)
        self.title("PyCrackQ")
        self.geometry("1920x950")

        self.cv_image = None
        self.binary_image = None
        self.final_image = None
        self.current_filename = "Unknown"
        self.current_filepath = ""
        self.analysis_data = {
            "File Name": "", "Analysis Time": "", "Algorithm": "", "Threshold/Window": "",
            "Area Ratio (%)": 0, "Total Crack Length": 0, "Average Width": 0, "Maximum Width": 0, "Junction Count": 0,
            "Fractal Dimension": 0, "Fractal Fit R2": 0
        }

        self.segments_data = []
        self.clods_data = []

        self.junction_points = []
        self.cached_angle_skel = None
        self.cached_angle_vis_base = None
        self.cached_dist_map = None
        self._analysis_cache_image_id = None
        self._analysis_cache_skeleton = None
        self._analysis_cache_dist_map = None

        self.scale_factor = 1.0
        self.unit = "px"

        self.circular_mask = None
        self.is_circular_mode = False
        self.circle_center = None
        self.circle_radius = 0
        self.last_saved_circle_center = None
        self.last_saved_circle_radius = None

        # Debounce mechanism: prevent cascading re-processing on slider drag
        self._debounce_binarize_id = None
        self._debounce_denoise_id = None
        self._processing_busy = False

        self.init_variables()

        self.setup_custom_styles()
        self.setup_ui()

    def init_variables(self):
        self.algo_var = ttk.StringVar(value="Sauvola")
        self.scale1_var = ttk.IntVar(value=25)
        self.scale2_var = ttk.DoubleVar(value=0.2)
        self.denoise_var = ttk.StringVar(value="Median Filter")
        self.denoise_k_var = ttk.IntVar(value=3)
        self.auto_mode_var = tk.IntVar(value=1 if DEFAULT_AUTO_MODE else 0)
        self._setting_auto_params = False  # Flag to prevent auto-disable during programmatic updates
        self._auto_recommendation_token = 0
        self._fractal_progress_win = None
        self._fractal_progress_bar = None
        self._fractal_progress_label = None

    def setup_custom_styles(self):
        style = ttk.Style()
        self.ui_colors = {
            "bg": "#f5f7fa",
            "panel": "#ffffff",
            "panel_alt": "#eef3f7",
            "panel_soft": "#f8fafc",
            "border": "#d8e0e8",
            "border_soft": "#e7edf3",
            "text": "#17202a",
            "muted": "#5d6b78",
            "subtle": "#7d8b99",
            "accent": "#18c878",
            "accent_dark": "#dff7ea",
            "accent_text": "#063d25",
            "danger": "#d64545",
            "status_bg": "#eef3f7",
        }
        c = self.ui_colors

        try:
            self.configure(background=c["bg"])
        except tk.TclError:
            pass

        style.configure("Shell.TFrame", background=c["bg"])
        style.configure("Panel.TFrame", background=c["panel"])
        style.configure("Soft.TFrame", background=c["panel_soft"])
        style.configure("Rail.TFrame", background=c["bg"])
        style.configure("Topbar.TFrame", background=c["bg"])
        style.configure("Status.TFrame", background=c["status_bg"])

        style.configure("Title.TLabel", background=c["bg"], foreground=c["text"],
                        font=("Microsoft YaHei", 15, "bold"))
        style.configure("BrandAccent.TLabel", background=c["bg"], foreground=c["accent"],
                        font=("Microsoft YaHei", 15, "bold"))
        style.configure("Muted.TLabel", background=c["panel_soft"], foreground=c["muted"],
                        font=("Microsoft YaHei", 10))
        style.configure("PanelMuted.TLabel", background=c["panel"], foreground=c["muted"],
                        font=("Microsoft YaHei", 9))
        style.configure("Section.TLabel", background=c["panel"], foreground=c["text"],
                        font=("Microsoft YaHei", 9, "bold"))
        style.configure("Status.TLabel", background=c["status_bg"], foreground=c["subtle"],
                        font=("Consolas", 9))

        style.configure("Panel.TLabelframe", background=c["panel"], bordercolor=c["border"],
                        lightcolor=c["border"], darkcolor=c["border"], relief="flat")
        style.configure("Panel.TLabelframe.Label", background=c["panel"], foreground=c["text"],
                        font=("Microsoft YaHei", 9, "bold"))

        for style_name, bg, fg, border, padding in [
            ("Command.TButton", c["panel_alt"], c["text"], c["border"], (10, 7)),
            ("Accent.TButton", c["accent"], c["accent_text"], c["accent"], (12, 8)),
            ("Tool.TButton", c["panel_alt"], c["text"], c["border"], (8, 7)),
            ("Rail.TButton", c["panel"], c["muted"], c["border_soft"], (6, 8)),
            ("RailActive.TButton", c["accent_dark"], c["accent"], c["accent"], (6, 8)),
            ("View.TButton", c["panel"], c["muted"], c["border_soft"], (10, 6)),
            ("ViewActive.TButton", c["panel_alt"], c["text"], c["accent"], (10, 6)),
        ]:
            style.configure(style_name, background=bg, foreground=fg,
                            bordercolor=border, lightcolor=border, darkcolor=border,
                            focusthickness=0, focuscolor=bg, padding=padding,
                            font=("Microsoft YaHei", 9))
            style.map(style_name,
                      background=[("active", c["panel_alt"]), ("pressed", c["accent_dark"])],
                      foreground=[("active", c["text"])],
                      bordercolor=[("active", c["accent"])])

        base_lf_style = "TLabelframe"
        custom_lf_style = "Card.TLabelframe"
        try:
            default_layout = style.layout(base_lf_style)
            if default_layout: style.layout(custom_lf_style, default_layout)
        except tk.TclError:
            pass
        style.configure(custom_lf_style, font=("Microsoft YaHei", 10, "bold"), borderwidth=2, relief="groove")

        for tag in ["Primary", "Success", "Warning", "Secondary"]:
            custom_btn_style = f"3D.{tag}.TButton"
            style.configure(custom_btn_style, font=("Microsoft YaHei", 9, "bold"), borderwidth=3, relief="raised", padding=8)

    def setup_ui(self):
        self.title("PyCrackQ")
        c = self.ui_colors
        # Log buffer (hidden; open with Ctrl+L)
        self._log_buffer = []
        self._log_window = None
        self._log_preview_text = None
        # Thumbnail store
        self._thumb_images = {}       # name -> cv2 image (BGR or grayscale)
        self._thumb_buttons = {}      # name -> ttk.Button
        self._current_view = None     # name of currently displayed thumbnail
        self._results_data = {}       # key -> value for results panel

        self.option_add("*Menu.Background", c["panel"])
        self.option_add("*Menu.Foreground", c["text"])
        self.option_add("*Menu.ActiveBackground", c["accent"])
        self.option_add("*Menu.ActiveForeground", c["accent_text"])
        self.option_add("*Menu.BorderWidth", 0)

        def make_menu():
            return tk.Menu(self, tearoff=0, bg=c["panel"], fg=c["text"],
                           activebackground=c["accent"], activeforeground=c["accent_text"],
                           relief="flat", borderwidth=0)

        def section(parent, title):
            ttk.Label(parent, text=title, style="Section.TLabel").pack(anchor=W, pady=(12, 6))

        def side_button(parent, text, command, accent=False):
            style_name = "Accent.TButton" if accent else "Tool.TButton"
            btn = ttk.Button(parent, text=text, style=style_name, command=command)
            btn.pack(fill=X, pady=3)
            return btn

        # === TOP COMMAND BAR ===
        topbar = ttk.Frame(self, style="Topbar.TFrame", padding=(10, 8, 10, 6))
        topbar.pack(fill=X, side=TOP)

        brand = ttk.Frame(topbar, style="Topbar.TFrame")
        brand.pack(side=LEFT, padx=(0, 18))
        ttk.Label(brand, text="PyCrack", style="Title.TLabel").pack(side=LEFT)
        ttk.Label(brand, text="Q", style="BrandAccent.TLabel").pack(side=LEFT)

        ttk.Button(topbar, text="Open Image", style="Accent.TButton",
                   command=self.load_image).pack(side=LEFT, padx=(0, 6))

        # Export dropdown
        def _show_export_menu(e):
            export_menu = make_menu()
            export_menu.add_command(label="Export Excel", command=self.export_excel)
            export_menu.add_command(label="Export CSV", command=self.export_csv)
            export_menu.add_command(label="Export PDF", command=self.export_pdf)
            export_menu.add_separator()
            export_menu.add_command(label="Export Binary Image", command=self.export_binary_image)
            export_menu.post(e.x_root, e.y_root)
        export_btn = ttk.Button(topbar, text="Export ▾", style="Command.TButton")
        export_btn.pack(side=LEFT, padx=3)
        export_btn.bind("<Button-1>", _show_export_menu)

        ttk.Button(topbar, text="Batch Process", style="Command.TButton",
                   command=self.batch_process).pack(side=LEFT, padx=3)

        ttk.Separator(topbar, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=10, pady=2)

        self._tb_calib = ttk.Button(topbar, text="Calibrate", style="Command.TButton",
                                     command=self.start_calibration)
        self._tb_calib.pack(side=LEFT, padx=3)

        self._tb_roi = ttk.Button(topbar, text="ROI Select", style="Command.TButton",
                                   command=self.select_circular_region)
        self._tb_roi.pack(side=LEFT, padx=3)

        ttk.Button(topbar, text="Manual Edit", style="Command.TButton",
                   command=self.start_manual_edit).pack(side=LEFT, padx=3)

        ttk.Separator(topbar, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=10, pady=2)

        # Analyze dropdown
        def _show_analyze_menu(e):
            ana_menu = make_menu()
            ana_menu.add_command(label="Area Ratio", command=self.calc_area)
            ana_menu.add_command(label="Skeleton Length", command=self.calc_length)
            ana_menu.add_command(label="Average Width", command=self.calc_width)
            ana_menu.add_separator()
            ana_menu.add_command(label="Crack Segmentation", command=self.calc_segmentation)
            ana_menu.add_command(label="Orientation Rose Diagram", command=self.calc_rose_diagram)
            ana_menu.add_command(label="Fractal Dimension", command=self.calc_fractal_dimension)
            ana_menu.add_command(label="Junction Angles", command=self.calc_angles)
            ana_menu.add_separator()
            ana_menu.add_command(label="Soil Clod Analysis", command=self.calc_clods)
            ana_menu.add_command(label="Crack Connectivity", command=self.calc_connectivity)
            ana_menu.add_command(label="Junction Classification", command=self.calc_junction_types)
            ana_menu.post(e.x_root, e.y_root)
        ana_btn = ttk.Button(topbar, text="Analyze ▾", style="Command.TButton")
        ana_btn.pack(side=LEFT, padx=3)
        ana_btn.bind("<Button-1>", _show_analyze_menu)

        # Auto toggle
        self.auto_cb = ttk.Checkbutton(
            topbar, text="Auto", variable=self.auto_mode_var,
            bootstyle="success-round-toggle", command=self._on_auto_mode_toggle
        )
        self.auto_cb.pack(side=LEFT, padx=(8, 2))

        # Log button
        ttk.Button(topbar, text="Log", style="Command.TButton",
                   command=self._open_log_window).pack(side=RIGHT, padx=3)

        # === MAIN WORKBENCH ===
        main_area = ttk.Frame(self, style="Shell.TFrame", padding=(8, 0, 8, 6))
        main_area.pack(fill=BOTH, expand=YES)

        # === LEFT ACTIVITY RAIL ===
        tool_panel = ttk.Frame(main_area, width=52, style="Rail.TFrame")
        tool_panel.pack(side=LEFT, fill=Y, padx=(0, 8))
        tool_panel.pack_propagate(False)

        tools_cfg = [
            ("A", "Area Ratio", self.calc_area),
            ("L", "Skeleton Length", self.calc_length),
            ("W", "Average Width", self.calc_width),
            ("S", "Crack Segmentation", self.calc_segmentation),
            ("R", "Rose Diagram", self.calc_rose_diagram),
            ("F", "Fractal Dimension", self.calc_fractal_dimension),
            ("∠", "Junction Angles", self.calc_angles),
            ("C", "Soil Clod Analysis", self.calc_clods),
            ("N", "Connectivity Graph", self.calc_connectivity),
            ("J", "Junction Types", self.calc_junction_types),
        ]
        for idx, (label, tip, cmd) in enumerate(tools_cfg):
            btn = ttk.Button(tool_panel, text=label, command=cmd,
                             style="RailActive.TButton" if idx == 0 else "Rail.TButton")
            btn.pack(pady=3, padx=4, fill=X)
            CreateToolTip(btn, tip)

        # === LEFT SIDE PANEL ===
        side_panel = ttk.Frame(main_area, width=250, style="Panel.TFrame", padding=(12, 10))
        side_panel.pack(side=LEFT, fill=Y, padx=(0, 8))
        side_panel.pack_propagate(False)

        section(side_panel, "Workflow")
        side_button(side_panel, "Open Image", self.load_image, accent=True)
        side_button(side_panel, "Calibrate Scale", self.start_calibration)
        side_button(side_panel, "Select ROI", self.select_circular_region)
        side_button(side_panel, "Manual Edit Mask", self.start_manual_edit)
        side_button(side_panel, "Batch Process", self.batch_process)

        # --- Parameters ---
        param_frame = ttk.Labelframe(side_panel, text=" Parameters ", padding=8, style="Panel.TLabelframe")
        param_frame.pack(fill=X, pady=(14, 4))

        self.algo_combo = ttk.Combobox(param_frame, textvariable=self.algo_var, state="readonly",
                                        font=("Microsoft YaHei", 9), bootstyle="success")
        self.algo_combo['values'] = (
            "Global Threshold", "Otsu", "Triangle", "Adaptive Mean", "Adaptive Gaussian",
            "Sauvola", "Niblack")
        self.algo_combo.bind("<<ComboboxSelected>>", self.on_algo_change)
        self.algo_combo.pack(fill=X, pady=(0, 8))

        self.param_container = ttk.Frame(param_frame, style="Panel.TFrame")
        self.param_container.pack(fill=X, pady=1)

        self.p1_label = ttk.Label(self.param_container, text="Window:", style="PanelMuted.TLabel")
        self.p1_label.pack(anchor=W)
        self.scale1 = ttk.Scale(self.param_container, from_=0, to=255, variable=self.scale1_var,
                                command=self.update_binary_image, bootstyle="success")
        self.scale1.pack(fill=X)
        self.scale1_val = ttk.Label(self.param_container, text="25", style="PanelMuted.TLabel")
        self.scale1_val.pack(anchor=E)

        self.p2_frame = ttk.Frame(self.param_container, style="Panel.TFrame")
        self.p2_label = ttk.Label(self.p2_frame, text="k:", style="PanelMuted.TLabel")
        self.p2_label.pack(anchor=W)
        self.scale2 = ttk.Scale(self.p2_frame, from_=0, to=10, variable=self.scale2_var,
                                command=self.update_binary_image, bootstyle="success")
        self.scale2.pack(fill=X)
        self.scale2_val = ttk.Label(self.p2_frame, text="0.20", style="PanelMuted.TLabel")
        self.scale2_val.pack(anchor=E)
        self.p2_frame.pack(fill=X, pady=2)

        ttk.Separator(param_frame).pack(fill=X, pady=3)

        ttk.Label(param_frame, text="Denoise:", style="PanelMuted.TLabel").pack(anchor=W)
        self.denoise_combo = ttk.Combobox(param_frame, textvariable=self.denoise_var, state="readonly",
                                          font=("Microsoft YaHei", 9))
        self.denoise_combo['values'] = ("None", "Gaussian Filter", "Mean Filter", "Median Filter")
        self.denoise_combo.bind("<<ComboboxSelected>>", self.update_denoise_image_event)
        self.denoise_combo.pack(fill=X, pady=1)

        self.denoise_k_frame = ttk.Frame(param_frame, style="Panel.TFrame")
        self.denoise_k_frame.pack(fill=X, pady=1)
        ttk.Label(self.denoise_k_frame, text="Kernel:", style="PanelMuted.TLabel").pack(anchor=W)
        self.denoise_scale = ttk.Scale(self.denoise_k_frame, from_=1, to=15, variable=self.denoise_k_var,
                                       command=self.update_denoise_image_event, bootstyle="success")
        self.denoise_scale.pack(fill=X)
        self.denoise_k_val = ttk.Label(self.denoise_k_frame, text="3", style="PanelMuted.TLabel")
        self.denoise_k_val.pack(anchor=E)

        ttk.Separator(param_frame).pack(fill=X, pady=3)

        self.auto_info_label = ttk.Label(
            param_frame, text="Auto: analyzes image, sets optimal params",
            style="PanelMuted.TLabel"
        )
        self.auto_info_label.pack(anchor=W)

        # === CENTER IMAGE WORKSPACE ===
        center_area = ttk.Frame(main_area, style="Soft.TFrame")
        center_area.pack(side=LEFT, fill=BOTH, expand=YES, padx=(0, 8))

        view_tabs = ttk.Frame(center_area, style="Soft.TFrame", padding=(8, 8, 8, 0))
        view_tabs.pack(fill=X)
        ttk.Label(view_tabs, text="Workspace", style="Muted.TLabel").pack(side=LEFT, padx=(0, 8))
        for name in ["Source", "Binary", "Result"]:
            btn = ttk.Button(view_tabs, text=name, style="View.TButton",
                             command=lambda n=name: self._switch_view(n))
            btn.pack(side=LEFT, padx=2)
            self._thumb_buttons[name] = btn
            self._thumb_images[name] = None

        ttk.Button(view_tabs, text="+", style="View.TButton",
                   command=self._add_current_as_thumbnail).pack(side=LEFT, padx=(6, 1))

        canvas_frame = ttk.Frame(center_area, style="Soft.TFrame", padding=10)
        canvas_frame.pack(fill=BOTH, expand=YES)
        self._canvas_frame = canvas_frame
        self.canvas_view = None
        self._no_image_label = ttk.Label(canvas_frame,
            text="No image loaded\nOpen an image to begin crack analysis",
            style="Muted.TLabel", anchor="center", justify="center")
        self._no_image_label.pack(fill=BOTH, expand=YES)

        # === RIGHT INSPECTOR PANEL ===
        right_panel = ttk.Frame(main_area, width=280, style="Panel.TFrame", padding=(10, 10))
        right_panel.pack(side=RIGHT, fill=Y)
        right_panel.pack_propagate(False)

        result_frame = ttk.Labelframe(right_panel, text=" Results ", padding=8, style="Panel.TLabelframe")
        result_frame.pack(fill=BOTH, expand=YES, pady=(0, 8))

        self._results_text = tk.Text(result_frame, height=14, font=("Consolas", 9),
                                     bg=c["panel_soft"], fg=c["text"], insertbackground=c["accent"],
                                     relief="flat", highlightthickness=1,
                                     highlightbackground=c["border"], highlightcolor=c["accent"],
                                     borderwidth=0, state="disabled", wrap="word")
        self._results_text.pack(fill=BOTH, expand=YES)

        log_frame = ttk.Labelframe(right_panel, text=" Log Preview ", padding=8, style="Panel.TLabelframe")
        log_frame.pack(fill=X)
        self._log_preview_text = tk.Text(log_frame, height=7, font=("Consolas", 8),
                                         bg=c["panel_soft"], fg=c["muted"], insertbackground=c["accent"],
                                         relief="flat", highlightthickness=1,
                                         highlightbackground=c["border"], highlightcolor=c["accent"],
                                         borderwidth=0, state="disabled", wrap="word")
        self._log_preview_text.pack(fill=X)

        # === STATUS BAR ===
        self._status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Frame(self, style="Status.TFrame")
        status_bar.pack(fill=X, side=BOTTOM)
        ttk.Label(status_bar, textvariable=self._status_var,
                  style="Status.TLabel", padding=(10, 5)).pack(fill=X)

        self.bind("<Control-l>", lambda e: self._open_log_window())

        self.log("System initialization complete... ")
        self._update_status()
        self.update_ui_controls()

    # === New UI helpers ===

    def _switch_view(self, name):
        if name in self._thumb_images and self._thumb_images[name] is not None:
            self._current_view = name
            if self.canvas_view is not None:
                self.canvas_view.set_image(self._thumb_images[name])
            for n, btn in self._thumb_buttons.items():
                try:
                    btn.configure(style="ViewActive.TButton" if n == name else "View.TButton")
                except tk.TclError:
                    btn.configure(bootstyle="success-outline" if n == name else "secondary-outline")
            self._update_status()

    def _add_current_as_thumbnail(self):
        if self.canvas_view is None:
            return
        import datetime
        ts = datetime.datetime.now().strftime("%H%M%S")
        name = f"View_{ts}"
        self._thumb_images[name] = self.canvas_view._cv_image.copy()
        btn = ttk.Button(self._thumb_buttons["Source"].master, text=name, style="View.TButton",
                         command=lambda n=name: self._switch_view(n))
        all_children = list(self._thumb_buttons["Source"].master.pack_slaves())
        plus_idx = [i for i, c in enumerate(all_children) if c.cget("text") == "+"]
        if plus_idx:
            btn.pack(side=LEFT, padx=1, before=all_children[plus_idx[0]])
        else:
            btn.pack(side=LEFT, padx=1)
        self._thumb_buttons[name] = btn
        self._switch_view(name)

    def _update_status(self):
        parts = []
        if self._current_view:
            parts.append(f"View: {self._current_view}")
        parts.append(self.current_filename)
        if self.cv_image is not None:
            h, w = self.cv_image.shape[:2]
            parts.append(f"{w}x{h}")
        scale_text = f"{self.scale_factor:.3f} px/{self.unit}" if self.unit != "px" else "1.000 px/px"
        parts.append(scale_text)
        mode = "Circle" if self.is_circular_mode else "Rect"
        parts.append(mode)
        parts.append(self.algo_var.get().split(" ")[0])
        denoise = self.denoise_var.get()
        if denoise != "None":
            denoise += f"({int(self.denoise_k_var.get())})"
        parts.append(denoise)
        self._status_var.set(" | ".join(parts))

    def _update_results_panel(self):
        self._results_text.config(state="normal")
        self._results_text.delete("1.0", "end")
        if not self._results_data:
            self._results_text.insert("1.0", "Run an analysis to see results here.")
        else:
            lines = []
            for k, v in self._results_data.items():
                lines.append(f"{k}:  {v}")
            self._results_text.insert("1.0", "\n".join(lines))
        self._results_text.config(state="disabled")

    def _set_result(self, key, value):
        self._results_data[key] = value
        self._update_results_panel()

    def _open_log_window(self):
        if self._log_window is not None:
            try:
                if self._log_window.winfo_exists():
                    self._log_window.lift()
                    return
            except tk.TclError:
                pass
            self._log_window = None
            self._log_window_text = None
        self._log_window = tk.Toplevel(self)
        self._log_window.title("System Log — Ctrl+L to toggle")
        self._log_window.geometry("700x300")
        c = self.ui_colors
        self._log_window.configure(background=c["bg"])
        self._log_window_text = tk.Text(self._log_window, font=("Consolas", 9),
                                        bg=c["panel_soft"], fg=c["text"],
                                        insertbackground=c["accent"], relief="flat",
                                        highlightthickness=1,
                                        highlightbackground=c["border"],
                                        highlightcolor=c["accent"])
        self._log_window_text.pack(fill=BOTH, expand=YES, padx=8, pady=8)
        for msg in self._log_buffer:
            self._log_window_text.insert("end", msg + "\n")
        self._log_window_text.see("end")
        self._log_window_text.config(state="disabled")

        def on_close():
            if self._log_window is not None:
                self._log_window.destroy()
            self._log_window_text = None
            self._log_window = None
        self._log_window.protocol("WM_DELETE_WINDOW", on_close)

    def log(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self._log_buffer.append(entry)
        if len(self._log_buffer) > 500:
            self._log_buffer = self._log_buffer[-300:]
        if self._log_window is not None and self._log_window_text is not None:
            try:
                if self._log_window.winfo_exists():
                    self._log_window_text.config(state="normal")
                    self._log_window_text.insert("end", entry + "\n")
                    self._log_window_text.see("end")
                    self._log_window_text.config(state="disabled")
            except (tk.TclError, AttributeError):
                pass
        if self._log_preview_text is not None:
            try:
                if self._log_preview_text.winfo_exists():
                    self._log_preview_text.config(state="normal")
                    self._log_preview_text.insert("end", entry + "\n")
                    lines = int(float(self._log_preview_text.index("end-1c").split(".")[0]))
                    if lines > 80:
                        self._log_preview_text.delete("1.0", "20.0")
                    self._log_preview_text.see("end")
                    self._log_preview_text.config(state="disabled")
            except (tk.TclError, AttributeError):
                pass

    def read_image(self, file_path, flags=cv2.IMREAD_COLOR):
        try:
            return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)
        except Exception as e:
            print(f"Error reading image {file_path}: {e}")
            return None

    def save_image_cv2(self, file_path, img):
        try:
            ext = os.path.splitext(file_path)[1]
            if not ext: ext = ".png"
            is_success, im_buf = cv2.imencode(ext, img)
            if is_success:
                im_buf.tofile(file_path)
                return True
        except Exception as e:
            print(f"Error saving image {file_path}: {e}")
        return False


    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")])
        if file_path:
            img = self.read_image(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                self.log("Error: Cannot read file")
                return
            self.cv_image = img
            self.current_filename = os.path.basename(file_path)
            self.current_filepath = file_path
            self.analysis_data["File Name"] = self.current_filename
            self.analysis_data["Analysis Time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.circular_mask = None
            self.is_circular_mode = False
            self.circle_center = None
            self.circle_radius = 0
            self.scale_factor = 1.0
            self.unit = "px"
            self.display_image(self.cv_image, "Source")
            self._update_status()
            self._schedule_binarization(immediate=True)
            self.log(f"Loaded image: {self.current_filename}")

    # Debounce and busy helpers

    def _schedule_binarization(self, immediate=False):
        """Debounced binarization - coalesces rapid slider events into one call.

        Heavy processing runs on a background thread so the UI stays responsive."""
        if self._debounce_binarize_id is not None:
            self.after_cancel(self._debounce_binarize_id)
            self._debounce_binarize_id = None
        if immediate:
            self._start_binarization_thread()
        else:
            self._set_busy(True)
            self._debounce_binarize_id = self.after(300, self._do_scheduled_binarization)

    def _do_scheduled_binarization(self):
        self._debounce_binarize_id = None
        self._start_binarization_thread()

    def _start_binarization_thread(self, use_auto=True):
        """Collect parameters on main thread, then launch background worker."""
        if self.cv_image is None:
            self._set_busy(False)
            return

        self._set_busy(True)

        if use_auto and self.auto_mode_var.get():
            self._start_auto_recommendation_thread()
            return

        # Read all tkinter variables on the main thread.
        params = {
            'img': self.cv_image,
            'method': self.algo_var.get(),
            'v1': int(self.scale1_var.get()),
            'v2': self.scale2_var.get(),
            'denoise_mode': self.denoise_var.get(),
            'denoise_k': int(self.denoise_k_var.get()),
            'is_circular_mode': self.is_circular_mode,
            'circular_mask':
                self.circular_mask.copy() if self.circular_mask is not None else None,
        }

        if params['v1'] < 3:
            params['v1'] = 3
        if params['v1'] % 2 == 0:
            params['v1'] += 1
        if params['denoise_k'] < 1:
            params['denoise_k'] = 1
        if params['denoise_k'] % 2 == 0:
            params['denoise_k'] += 1

        t = threading.Thread(target=self._run_binarization_thread,
                             args=(params,), daemon=True)
        t.start()

    def _start_auto_recommendation_thread(self):
        """Run automatic parameter recommendation off the UI thread."""
        if self.cv_image is None:
            self._set_busy(False)
            return

        self._auto_recommendation_token += 1
        token = self._auto_recommendation_token
        image_id = id(self.cv_image)
        img = self.cv_image.copy()
        self._status_var.set("Analyzing image parameters...")

        t = threading.Thread(
            target=self._run_auto_recommendation_thread,
            args=(img, image_id, token),
            daemon=True,
        )
        t.start()

    def _run_auto_recommendation_thread(self, img, image_id, token):
        try:
            params = recommend_parameters(img)
            self.after(0, lambda: self._on_auto_recommendation_complete(
                params, image_id, token))
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: self._on_auto_recommendation_error(error_msg, token))

    def _on_auto_recommendation_complete(self, params, image_id, token):
        if token != self._auto_recommendation_token:
            return
        if self.cv_image is None or id(self.cv_image) != image_id:
            return
        if not self.auto_mode_var.get():
            self._set_busy(False)
            return

        self._apply_auto_params(params)
        self._start_binarization_thread(use_auto=False)

    def _on_auto_recommendation_error(self, error_msg, token):
        if token != self._auto_recommendation_token:
            return
        self.log(f"Auto param analysis failed: {error_msg}")
        self._set_busy(False)

    def _run_binarization_thread(self, params):
        """Background processing — no tkinter access allowed here."""
        log_msgs = []

        def thread_log(msg):
            log_msgs.append(msg)

        try:
            img = params['img']
            method = params['method']
            v1 = params['v1']
            v2 = params['v2']

            # Binarization.
            binary = apply_binarization(img, method, v1, v2)

            # Circular mask.
            if params['is_circular_mode'] and params['circular_mask'] is not None:
                binary = apply_circular_mask(binary, params['circular_mask'])

            # Denoising.
            final = apply_denoising(binary, params['denoise_mode'],
                                    params['denoise_k'])
            if params['is_circular_mode'] and params['circular_mask'] is not None:
                final = apply_circular_mask(final, params['circular_mask'])

            # Schedule UI update on the main thread.
            self.after(0, lambda: self._on_binarization_complete(
                binary, final, method, v1, v2, log_msgs))

        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: self._on_binarization_error(error_msg))

    def _on_binarization_complete(self, binary, final, method, v1, v2, log_msgs):
        """Called on main thread after background processing succeeds."""
        self.binary_image = binary
        self.final_image = final
        self._invalidate_analysis_cache()
        self.analysis_data["Algorithm"] = method
        self.analysis_data["Threshold/Window"] = f"Win:{v1}, k:{v2:.2f}"
        self.display_image(binary, "Binary")
        self.display_image(final, "Result")
        for msg in log_msgs:
            self.log(msg)
        self._set_busy(False)

    def _on_binarization_error(self, error_msg):
        """Called on main thread when background processing fails."""
        self.log(f"Binarization error: {error_msg}")
        self._set_busy(False)

    def _schedule_denoising(self):
        """Debounced denoising - coalesces rapid slider events into one call."""
        if self._debounce_denoise_id is not None:
            self.after_cancel(self._debounce_denoise_id)
            self._debounce_denoise_id = None
        self._debounce_denoise_id = self.after(200, self._do_scheduled_denoising)

    def _do_scheduled_denoising(self):
        self._debounce_denoise_id = None
        self.process_denoising()

    def _set_busy(self, busy):
        self._processing_busy = busy
        if busy:
            self._status_var.set("Processing... please wait")
        else:
            self._update_status()


    def process_denoising(self):
        if self.binary_image is None or not hasattr(self, 'denoise_k_var'): return
        mode = self.denoise_var.get()
        k = int(self.denoise_k_var.get())
        if k < 1: k = 1
        if k % 2 == 0: k += 1
        self.final_image = apply_denoising(self.binary_image, mode, k)
        if self.is_circular_mode and self.circular_mask is not None:
            self.final_image = apply_circular_mask(self.final_image, self.circular_mask)
        self._invalidate_analysis_cache()
        self.display_image(self.final_image, "Result")

    # Auto parameter recommendation

    def _apply_auto_params(self, params):
        """Apply precomputed automatic binarization parameters to the UI."""
        self._setting_auto_params = True
        try:
            self.algo_var.set(params['method'])
            self.update_ui_controls()
            self.scale1_var.set(params['window'])
            self.scale2_var.set(params['k'])
        finally:
            self._setting_auto_params = False

        # Update slider labels
        self.scale1_val.config(text=str(params['window']))
        self.scale2_val.config(text=f"{params['k']:.2f}")

        # Build concise info label
        illumination = params.get('illumination_level', '?')
        complexity = params.get('complexity_level', '?')
        crack_w = params.get('crack_width_est', '?')
        noise_s = params.get('noise_sigma', '?')
        self.auto_info_label.config(
            text=f"Auto: {params['method']}, win={params['window']}, "
                 f"k={params['k']:.2f} | crack~{crack_w}px, "
                 f"noise~{noise_s} sigma, {illumination} light, {complexity} cracks"
        )
        reason = params.get('reason', '')
        if reason:
            self.log(f"Auto recommendation: {reason}")

    def _on_auto_mode_toggle(self):
        if self.auto_mode_var.get():
            self.log("Auto parameter recommendation enabled")
            # Re-process with auto params
            if self.cv_image is not None:
                self._schedule_binarization(immediate=True)
        else:
            self.log("Auto parameter recommendation disabled — manual mode")
            self.auto_info_label.config(text="Manual mode: use sliders to adjust params")

    def _on_manual_param_change(self):
        """Called when user moves a slider — disable auto mode if it was on."""
        if self._setting_auto_params:
            return  # Programmatic update, don't disable
        if hasattr(self, 'auto_mode_var') and self.auto_mode_var.get():
            self.auto_mode_var.set(0)
            self.auto_info_label.config(text="Manual mode: user adjusted parameters")
            self.log("Auto mode disabled — parameter manually adjusted")


    def _invalidate_analysis_cache(self):
        self._analysis_cache_image_id = None
        self._analysis_cache_skeleton = None
        self._analysis_cache_dist_map = None

    def _get_skeleton(self):
        if self.final_image is None: return None
        image_id = id(self.final_image)
        if self._analysis_cache_image_id == image_id and self._analysis_cache_skeleton is not None:
            return self._analysis_cache_skeleton.copy()
        skel_img = get_skeleton(self.final_image)
        self._analysis_cache_image_id = image_id
        self._analysis_cache_skeleton = skel_img
        return skel_img.copy()

    def _get_distance_map(self):
        if self.final_image is None: return None
        image_id = id(self.final_image)
        if self._analysis_cache_image_id == image_id and self._analysis_cache_dist_map is not None:
            return self._analysis_cache_dist_map.copy()
        dist_map = get_distance_map(self.final_image)
        self._analysis_cache_image_id = image_id
        self._analysis_cache_dist_map = dist_map
        return dist_map.copy()

    # Analysis methods

    def calc_area(self):
        if self.final_image is None: return

        if self.is_circular_mode and self.circular_mask is not None:
            total = cv2.countNonZero(self.circular_mask)
            crack = cv2.countNonZero(cv2.bitwise_and(self.final_image, self.final_image, mask=self.circular_mask))
        else:
            total = self.final_image.size
            crack = cv2.countNonZero(self.final_image)

        if total > 0:
            ratio = (crack / total) * 100
        else:
            ratio = 0

        area_phy = crack / (self.scale_factor ** 2)
        unit_sq = f"{self.unit}2"
        self.analysis_data["Area Ratio (%)"] = round(ratio, 4)
        msg = f"Image Size: {self.final_image.shape}\nCrack Pixels: {crack} px\nPhysical Area: {area_phy:.2f} {unit_sq}\nArea Ratio: {ratio:.4f}%"
        self.log(f"Area analysis: {area_phy:.2f} {unit_sq} ({ratio:.2f}%)")
        self._set_result("Area Ratio", f"{ratio:.2f}%")
        messagebox.showinfo("Area Analysis", msg)

    def calc_length(self):
        if self.final_image is None: return
        skel_bool = self._get_skeleton() > 0
        metrics = calculate_accurate_metrics(
            self.final_image, skel_bool, None,
            scale_factor=self.scale_factor,
            is_circular_mode=self.is_circular_mode,
            circular_mask=self.circular_mask
        )
        length_px = metrics['length_px']
        length_phy = metrics['length_phy']
        self.analysis_data["Total Crack Length"] = f"{length_phy:.2f} {self.unit}"
        self.display_image((skel_bool * 255).astype(np.uint8), "Result")
        self.log(f"Skeleton total length (accurate): {length_phy:.2f} {self.unit}")
        self._set_result("Crack Length", f"{length_phy:.2f} {self.unit}")
        messagebox.showinfo("Length Analysis (Accurate)", f"Total crack skeleton length: {length_phy:.2f} {self.unit}\n(Pixel estimate: {length_px:.1f})")

    def calc_width(self):
        if self.final_image is None: return
        dist_map = self._get_distance_map()
        skel_bool = self._get_skeleton() > 0

        metrics = calculate_accurate_metrics(
            self.final_image, skel_bool, dist_map,
            scale_factor=self.scale_factor,
            is_circular_mode=self.is_circular_mode,
            circular_mask=self.circular_mask
        )
        avg_w = metrics['avg_width_phy']
        max_w = metrics['max_width_phy']

        self.analysis_data["Average Width"] = f"{avg_w:.2f} {self.unit}"
        self.analysis_data["Maximum Width"] = f"{max_w:.2f} {self.unit}"
        dist_display = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        dist_color = cv2.applyColorMap(dist_display, cv2.COLORMAP_JET)
        self.display_image(dist_color, "Result")
        self.log(f"Width analysis: Avg={avg_w:.2f} {self.unit}, Max={max_w:.2f} {self.unit}")
        self._set_result("Avg Width", f"{avg_w:.2f} {self.unit}")
        self._set_result("Max Width", f"{max_w:.2f} {self.unit}")
        messagebox.showinfo("Width Analysis (Accurate)", f"Average Width (Area/Length): {avg_w:.2f} {self.unit}\nMaximum Width: {max_w:.2f} {self.unit}")


    def calc_rose_diagram(self):
        if self.final_image is None:
            messagebox.showwarning("Notice", "Please process the image first")
            return
        self.log("Drawing crack orientation rose diagram...")

        skel_bool = self._get_skeleton() > 0
        if skel_bool is None: return

        junction_mask, _, _ = detect_junctions(skel_bool)
        skel_no_junctions = skel_bool.copy()
        skel_no_junctions[junction_mask] = False
        labeled_skeleton = label(skel_no_junctions, connectivity=2)
        props = regionprops(labeled_skeleton)
        orientations = []
        lengths = []

        neighbor_kernel = np.array([[1, 1, 1],
                                     [1, 0, 1],
                                     [1, 1, 1]], dtype=np.uint8)

        for prop in props:
            if prop.area < max(1, MIN_CRACK_AREA // 2): continue

            coords = prop.coords
            if len(coords) < 2: continue

            min_r, min_c = np.min(coords, axis=0)
            max_r, max_c = np.max(coords, axis=0)
            local_h = max_r - min_r + 1
            local_w = max_c - min_c + 1
            local_skel = np.zeros((local_h, local_w), dtype=np.uint8)
            for coord in coords:
                local_skel[coord[0] - min_r, coord[1] - min_c] = 1

            neighbor_count = cv2.filter2D(local_skel, -1, neighbor_kernel)
            endpoints_local = np.argwhere((local_skel == 1) & (neighbor_count == 1))

            if len(endpoints_local) >= 2:
                max_dist = 0
                best_pair = (endpoints_local[0], endpoints_local[1])
                for i in range(len(endpoints_local)):
                    for j in range(i + 1, len(endpoints_local)):
                        p1, p2 = endpoints_local[i], endpoints_local[j]
                        d = np.linalg.norm(p1 - p2)
                        if d > max_dist:
                            max_dist = d
                            best_pair = (p1, p2)

                p1_global = best_pair[0] + [min_r, min_c]
                p2_global = best_pair[1] + [min_r, min_c]

                dy = p2_global[0] - p1_global[0]
                dx = p2_global[1] - p1_global[1]
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)

                if angle_deg < 0: angle_deg += 180
                if angle_deg >= 180: angle_deg -= 180
            else:
                if len(coords) >= 2:
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    if np.std(cols) > np.std(rows):
                        coeffs = np.polyfit(cols, rows, 1)
                        angle_deg = 90 - math.degrees(math.atan(coeffs[0]))
                    else:
                        coeffs = np.polyfit(rows, cols, 1)
                        angle_deg = math.degrees(math.atan(coeffs[0]))
                    if angle_deg < 0: angle_deg += 180
                    if angle_deg >= 180: angle_deg -= 180
                else:
                    continue

            orientations.append(angle_deg)
            seg_length = trace_segment_euclidean_length(skel_no_junctions, coords)
            lengths.append(seg_length / self.scale_factor)

        if not orientations:
            self.log("No valid crack segments detected")
            return
        show_rose_plot(self, orientations, lengths, unit=self.unit)


    def calc_fractal_dimension(self):
        if self.final_image is None:
            messagebox.showwarning("Notice", "Please process the image first")
            return
        self.log("Calculating fractal dimension (Box-Counting Method)...")
        source = self.binary_image if self.binary_image is not None else self.final_image
        pixels = (source > 0).copy()
        self._open_fractal_progress_window()
        self._start_fractal_dimension_thread(pixels)

    def _open_fractal_progress_window(self):
        if self._fractal_progress_win is not None:
            try:
                if self._fractal_progress_win.winfo_exists():
                    self._fractal_progress_win.destroy()
            except tk.TclError:
                pass

        win = tk.Toplevel(self)
        win.title("Fractal Dimension")
        win.geometry("360x120")
        win.resizable(False, False)
        win.protocol("WM_DELETE_WINDOW", lambda: None)
        ttk.Label(win, text="Fractal dimension progress", padding=(12, 10, 12, 4)).pack(fill=X)
        self._fractal_progress_label = ttk.Label(
            win, text="Preparing box-counting...", padding=(12, 0, 12, 4)
        )
        self._fractal_progress_label.pack(fill=X)
        self._fractal_progress_bar = ttk.Progressbar(
            win, maximum=1, value=0, mode="determinate"
        )
        self._fractal_progress_bar.pack(fill=X, padx=12, pady=(2, 12))
        self._fractal_progress_win = win

    def _start_fractal_dimension_thread(self, pixels):
        t = threading.Thread(
            target=self._run_fractal_dimension_thread,
            args=(pixels,),
            daemon=True,
        )
        t.start()

    def _run_fractal_dimension_thread(self, pixels):
        def progress(step, total, box_size, count):
            self.after(0, lambda: self._on_fractal_progress(
                step, total, box_size, count))

        try:
            sizes, counts = get_fractal_dim(pixels, progress_callback=progress)
            if not sizes or not counts or len(sizes) < 2:
                result = {"error": "too_small"}
            elif any(c == 0 for c in counts):
                result = {"error": "invalid_distribution"}
            else:
                coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
                fractal_dim = -coeffs[0]
                fit_log_counts = np.polyval(coeffs, np.log(sizes))
                y_bar = np.mean(np.log(counts))
                ss_tot = np.sum((np.log(counts) - y_bar) ** 2)
                ss_res = np.sum((np.log(counts) - fit_log_counts) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                result = {
                    "sizes": sizes,
                    "counts": counts,
                    "coeffs": coeffs,
                    "fractal_dim": fractal_dim,
                    "r_squared": r_squared,
                }
            self.after(0, lambda: self._on_fractal_dimension_complete(result))
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: self._on_fractal_dimension_error(error_msg))

    def _on_fractal_progress(self, step, total, box_size, count):
        if self._fractal_progress_bar is not None:
            self._fractal_progress_bar.configure(maximum=max(total, 1))
            self._fractal_progress_bar["value"] = step
        if self._fractal_progress_label is not None:
            self._fractal_progress_label.config(
                text=f"Fractal dimension progress: {step}/{total}, box={box_size}, count={count}"
            )
        self._status_var.set(f"Fractal dimension progress: {step}/{total}")

    def _close_fractal_progress_window(self):
        if self._fractal_progress_win is not None:
            try:
                if self._fractal_progress_win.winfo_exists():
                    self._fractal_progress_win.destroy()
            except tk.TclError:
                pass
        self._fractal_progress_win = None
        self._fractal_progress_bar = None
        self._fractal_progress_label = None
        self._update_status()

    def _on_fractal_dimension_complete(self, result):
        self._close_fractal_progress_window()
        if result.get("error") == "too_small":
            self.log("Image is too small to calculate fractal dimension")
            messagebox.showwarning("Notice", "Image is too small to calculate fractal dimension\nAt least a 4x4 pixel area is required")
            return
        if result.get("error") == "invalid_distribution":
            self.log("Crack area is too small or irregularly distributed; cannot calculate fractal dimension")
            messagebox.showwarning("Notice", "Crack area is too small or irregularly distributed\nCannot calculate a valid fractal dimension")
            return

        sizes = result["sizes"]
        counts = result["counts"]
        coeffs = result["coeffs"]
        fractal_dim = result["fractal_dim"]
        r_squared = result["r_squared"]
        self.analysis_data["Fractal Dimension"] = round(fractal_dim, 4)
        self.analysis_data["Fractal Fit R2"] = round(r_squared, 4)
        self.log(f"Fractal Dimension D = {fractal_dim:.4f} (R2 = {r_squared:.4f})")
        self._set_result("Fractal Dim", f"{fractal_dim:.4f}")
        self._set_result("Fractal R2", f"{r_squared:.4f}")
        show_fractal_plot(self, sizes, counts, coeffs, fractal_dim, r_squared)

    def _on_fractal_dimension_error(self, error_msg):
        self._close_fractal_progress_window()
        self.log(f"Fractal dimension error: {error_msg}")
        messagebox.showerror("Error", error_msg)

    # Junction angle analysis

    def calc_angles(self):
        if self.final_image is None:
            messagebox.showwarning("Notice", "Please process the image first")
            return
        self.log("Starting junction angle analysis...")

        skel_bool = self._get_skeleton() > 0
        if skel_bool is None: return
        self.cached_dist_map = self._get_distance_map()
        junction_mask, _, _ = detect_junctions(skel_bool)
        j_labels, j_num = label(junction_mask, return_num=True, connectivity=2)
        j_props = regionprops(j_labels)
        self.junction_points = []

        if self.is_circular_mode and self.final_image is not None:
            vis_img = cv2.cvtColor(self.final_image, cv2.COLOR_GRAY2BGR)
        elif len(self.cv_image.shape) == 2:
            vis_img = cv2.cvtColor(self.cv_image, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = self.cv_image.copy()
        vis_img = (vis_img * 0.4).astype(np.uint8)
        vis_img[skel_bool == 1] = [0, 255, 0]
        for jp in j_props:
            jc = jp.centroid
            r, c = int(jc[0]), int(jc[1])
            self.junction_points.append((r, c))
            cv2.circle(vis_img, (c, r), 4, (0, 0, 255), -1)
            cv2.circle(vis_img, (c, r), 6, (255, 255, 255), 1)
        self.cached_angle_skel = skel_bool
        self.cached_angle_vis_base = vis_img.copy()
        self.analysis_data["Junction Count"] = len(self.junction_points)
        self._set_result("Junctions", str(len(self.junction_points)))
        self.log(f"Detected {len(self.junction_points)} junction points.")
        self.open_angle_interaction_window(vis_img)


    def open_angle_interaction_window(self, base_img):
        top = tk.Toplevel(self)
        top.title("Junction Angle Analysis (Left-click to select, wheel to zoom, right-drag to pan)")

        sw = top.winfo_screenwidth()
        sh = top.winfo_screenheight()
        win_w, win_h = min(1200, sw - 100), min(800, sh - 100)
        top.geometry(f"{win_w}x{win_h}+{(sw - win_w) // 2}+{(sh - win_h) // 2}")

        h, w = base_img.shape[:2]

        state = {
            'scale': min((win_w - 40) / w, (win_h - 40) / h),
            'img': base_img.copy(),
            'cx': 0,
            'cy': 0,
            'new_w': 0,
            'new_h': 0
        }

        frame = ttk.Frame(top)
        frame.pack(expand=YES, fill=BOTH)

        v_scroll = ttk.Scrollbar(frame, orient=VERTICAL)
        h_scroll = ttk.Scrollbar(frame, orient=HORIZONTAL)

        canvas = tk.Canvas(frame, bg=self.ui_colors["panel_soft"], yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        v_scroll.config(command=canvas.yview)
        h_scroll.config(command=canvas.xview)

        v_scroll.pack(side=RIGHT, fill=Y)
        h_scroll.pack(side=BOTTOM, fill=X)
        canvas.pack(side=LEFT, expand=YES, fill=BOTH)

        bottom_bar = ttk.Frame(top)
        bottom_bar.pack(fill=X, side=BOTTOM)

        info_label = ttk.Label(bottom_bar, text="Click a red junction point in the image... (wheel zoom and right-drag supported)",
                               font=("Microsoft YaHei", 12), bootstyle="inverse-info", padding=10)
        info_label.pack(side=LEFT, fill=X, expand=True)

        def save_current_view():
            file_path = filedialog.asksaveasfilename(
                parent=top, title="Save Image", defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"),
                           ("TIFF Image", "*.tiff"), ("Bitmap Image", "*.bmp"), ("All Files", "*.*")],
            )
            if not file_path:
                return
            img_rgb = cv2.cvtColor(state['img'], cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(file_path)

        ttk.Button(bottom_bar, text="Save Image", bootstyle="success",
                   command=save_current_view).pack(side=RIGHT, padx=5, pady=5)

        def update_display():
            scale = state['scale']
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            state['new_w'] = new_w
            state['new_h'] = new_h

            resized = cv2.resize(state['img'], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            tk_img = ImageTk.PhotoImage(pil_img)

            canvas.delete("all")
            canvas_w = canvas.winfo_width()
            canvas_h = canvas.winfo_height()

            if canvas_w < 10: canvas_w = win_w - 20
            if canvas_h < 10: canvas_h = win_h - 20

            cx = max(new_w // 2, canvas_w // 2)
            cy = max(new_h // 2, canvas_h // 2)
            state['cx'] = cx
            state['cy'] = cy

            canvas.create_image(cx, cy, anchor=CENTER, image=tk_img)
            canvas.image = tk_img
            canvas.config(scrollregion=(0, 0, new_w, new_h))

        def on_mousewheel(event):
            if getattr(event, 'num', 0) == 4 or getattr(event, 'delta', 0) > 0:
                state['scale'] *= 1.15
            elif getattr(event, 'num', 0) == 5 or getattr(event, 'delta', 0) < 0:
                state['scale'] /= 1.15

            state['scale'] = max(0.05, min(state['scale'], 20.0))
            update_display()

        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Button-4>", on_mousewheel)
        canvas.bind("<Button-5>", on_mousewheel)

        def on_resize(event):
            if event.widget == canvas:
                update_display()
        canvas.bind("<Configure>", on_resize)

        def on_click(event):
            canvas_x = canvas.canvasx(event.x)
            canvas_y = canvas.canvasy(event.y)

            img_top_left_x = state['cx'] - state['new_w'] // 2
            img_top_left_y = state['cy'] - state['new_h'] // 2

            rel_x = canvas_x - img_top_left_x
            rel_y = canvas_y - img_top_left_y

            click_x = int(rel_x / state['scale'])
            click_y = int(rel_y / state['scale'])

            if click_x < 0 or click_x >= w or click_y < 0 or click_y >= h:
                return

            min_dist = 999
            target_pt = None
            for r, c in self.junction_points:
                dist = math.hypot(r - click_y, c - click_x)
                if dist < 20:
                    if dist < min_dist:
                        min_dist = dist
                        target_pt = (r, c)
            if target_pt:
                r, c = target_pt
                angles = calculate_branch_angles(self.cached_angle_skel, r, c)
                local_width = 5
                if self.cached_dist_map is not None:
                    try: local_width = self.cached_dist_map[r, c]
                    except: pass
                result_img = self.draw_angle_arcs(self.cached_angle_vis_base, (r, c), angles, local_width)
                state['img'] = result_img
                update_display()
                info_label.config(text=f"Selected junction point ({c}, {r})", bootstyle="inverse-success")
            else:
                state['img'] = self.cached_angle_vis_base
                update_display()
                info_label.config(text="No junction point selected (click a red point)", bootstyle="inverse-secondary")

        canvas.bind("<Button-1>", on_click)

        def start_pan(event):
            canvas.scan_mark(event.x, event.y)
        def do_pan(event):
            canvas.scan_dragto(event.x, event.y, gain=1)

        canvas.bind("<ButtonPress-3>", start_pan)
        canvas.bind("<B3-Motion>", do_pan)
        canvas.bind("<ButtonPress-2>", start_pan)
        canvas.bind("<B2-Motion>", do_pan)

    def draw_angle_arcs(self, img, center, angles, local_width=5):
        if len(angles) < 2: return img
        r, c = center
        vis_img = img.copy()
        radius_base = max(20, local_width * 2 + 10)
        arc_radius = int(radius_base)
        img_pil = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        n = len(angles)
        for i in range(n):
            a1 = angles[i]
            a2 = angles[(i + 1) % n]
            diff = a2 - a1
            if diff < 0: diff += 360
            gap = 8
            if diff <= gap * 2: continue
            draw_start = a1 + gap
            draw_end = a1 + diff - gap
            bbox = [(c - arc_radius, r - arc_radius), (c + arc_radius, r + arc_radius)]
            draw.arc(bbox, start=draw_start, end=draw_end, fill="yellow", width=2)
            mid_angle_rad = math.radians(a1 + diff / 2)
            text_r = arc_radius + 15
            tx = c + text_r * math.cos(mid_angle_rad)
            ty = r + text_r * math.sin(mid_angle_rad)
            text = f"{diff:.1f}"
            try:
                left, top, right, bottom = draw.textbbox((tx, ty), text, font=font, anchor="mm")
                draw.rectangle((left-2, top-2, right+2, bottom+2), fill="black")
            except: pass
            draw.text((tx, ty), text, font=font, fill="white", anchor="mm")
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


    def calc_segmentation(self):
        if self.final_image is None:
            messagebox.showwarning("Notice", "Please process the image first")
            return

        self.log("Starting accurate crack segmentation and statistics...")

        n_lbl, l_img, stats, _ = cv2.connectedComponentsWithStats(self.final_image, connectivity=8)
        clean_bin = np.zeros_like(self.final_image)
        for i in range(1, n_lbl):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_CRACK_AREA:
                clean_bin[l_img == i] = 255

        non_zero_count = cv2.countNonZero(clean_bin)
        self.log(f"Detected nonzero pixel count: {non_zero_count}")
        if non_zero_count == 0:
            self.log("No valid crack area detected (area too small)")
            messagebox.showwarning("Notice", "No valid crack area detected\nAll crack areas may be smaller than the minimum threshold")
            return

        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        smooth_bin = cv2.morphologyEx(clean_bin, cv2.MORPH_CLOSE, smooth_kernel)

        skel_bool = skeletonize(smooth_bin > 0)

        j_mask, _, _ = detect_junctions(skel_bool)
        skel_bool[j_mask] = False

        l_skel = label(skel_bool, connectivity=2)
        valid_props = [p for p in regionprops(l_skel) if p.area >= max(1, MIN_CRACK_AREA)]
        self.log(f"Detected valid skeleton segment count: {len(valid_props)}")
        if not valid_props:
            self.log("No Valid Cracks")
            messagebox.showwarning("Notice", "No valid crack segments detected\nPlease check whether the image has clear crack structures")
            return

        valid_props.sort(key=lambda p: (int(p.centroid[0] / 50), p.centroid[1]))

        clean_skel_lbl = np.zeros_like(l_skel)
        for new_id, p in enumerate(valid_props, start=1):
            for r, c in p.coords:
                clean_skel_lbl[r, c] = new_id

        dist_map = cv2.distanceTransform(smooth_bin, cv2.DIST_L2, 5)
        mask_smooth = smooth_bin > 0
        expanded_labels = watershed(-dist_map, clean_skel_lbl, mask=mask_smooth)

        if self.current_filepath and os.path.exists(self.current_filepath):
            img_color = self.read_image(self.current_filepath, cv2.IMREAD_COLOR)
            if img_color is None:
                img_color = cv2.cvtColor(self.cv_image, cv2.COLOR_GRAY2BGR)
        else:
            img_color = cv2.cvtColor(self.cv_image, cv2.COLOR_GRAY2BGR)
        vis_img = (img_color * 0.4).astype(np.uint8)

        colors = np.random.randint(50, 255, size=(len(valid_props) + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]
        vis_img[expanded_labels > 0] = colors[expanded_labels[expanded_labels > 0]]

        if self.is_circular_mode and self.circular_mask is not None:
            vis_img[self.circular_mask == 0] = [180, 180, 180]

        segment_lengths = []
        segment_widths = []
        segment_ids = []
        self.segments_data = []

        drawn_boxes = []
        font_scale = max(0.3, min(0.6, vis_img.shape[1] / 2000))
        thickness = 1

        for new_id, p in enumerate(valid_props, start=1):
            coords = p.coords
            segment_skel = (clean_skel_lbl == new_id)
            seg_mask = ((expanded_labels == new_id).astype(np.uint8) * 255)
            seg_metrics = calculate_segment_metrics(
                seg_mask, segment_skel, dist_map,
                scale_factor=self.scale_factor,
                is_circular_mode=self.is_circular_mode,
                circular_mask=self.circular_mask
            )

            length_phy = seg_metrics['length_phy']
            avg_w_phy = seg_metrics['avg_width_phy']

            segment_lengths.append(length_phy)
            segment_widths.append(avg_w_phy)
            segment_ids.append(new_id)

            self.segments_data.append({
                "ID": new_id,
                f"Length({self.unit})": round(length_phy, 2),
                f"AvgWidth({self.unit})": round(avg_w_phy, 2)
            })

            cy, cx = p.centroid
            text = str(new_id)
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            pad = 2
            best_tx = int(cx - tw / 2)
            best_ty = int(cy)

            placed = False
            search_radius = 2.0
            angle = 0.0

            while not placed and search_radius < 150:
                box1 = [best_tx - pad, best_ty - th - pad, best_tx + tw + pad, best_ty + base + pad]
                collision = False
                for box2 in drawn_boxes:
                    if not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3]):
                        collision = True
                        break

                if not collision:
                    placed = True
                else:
                    angle += 0.8
                    search_radius += 1.5
                    best_tx = int(cx - tw / 2 + search_radius * math.cos(angle))
                    best_ty = int(cy + search_radius * math.sin(angle))

            best_tx = max(0, min(vis_img.shape[1] - tw, best_tx))
            best_ty = max(th, min(vis_img.shape[0] - base, best_ty))
            drawn_boxes.append([best_tx - pad, best_ty - th - pad, best_tx + tw + pad, best_ty + base + pad])

            dist_moved = math.hypot(best_tx + tw/2 - cx, best_ty - th/2 - cy)
            if dist_moved > 15:
                cv2.line(vis_img, (int(cx), int(cy)), (int(best_tx + tw/2), int(best_ty - th/2)), (150, 255, 150), 1, cv2.LINE_AA)
                cv2.circle(vis_img, (int(cx), int(cy)), 2, (150, 255, 150), -1)

            cv2.putText(vis_img, text, (best_tx, best_ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(vis_img, text, (best_tx, best_ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

        self.display_image(vis_img, "Result")
        self._set_result("Segments", str(len(valid_props)))
        open_enlarged_result_window(self, vis_img, f"Crack Segmentation Result - {len(valid_props)} segments (Unit: {self.unit})")

        show_histograms(self, segment_ids, segment_lengths, segment_widths, unit=self.unit)

    def calc_clods(self):
        """Identify and analyze soil clods (polygons) enclosed by crack networks."""
        if self.final_image is None:
            messagebox.showwarning("Notice", "Please process the image first")
            return

        # Use binary_image (pre-denoised) for clod analysis to preserve
        # thin crack lines that median filtering would erase. The morphological
        # closing step inside analyze_soil_clods bridges any remaining gaps.
        src_image = self.binary_image if self.binary_image is not None else self.final_image

        self.log("Starting soil clod analysis...")

        result = analyze_soil_clods(
            src_image,
            scale_factor=self.scale_factor,
            is_circular_mode=self.is_circular_mode,
            circular_mask=self.circular_mask
        )

        clods = result['clods']
        summary = result['summary']
        vis_image = result['vis_image']

        self.clods_data = clods

        self.log(
            f"Soil clod analysis: {summary['clod_count']} clods identified, "
            f"area ratio: {summary['clod_area_ratio']}%, "
            f"mean area: {summary['mean_area']} {summary['unit']}2"
        )
        self._set_result("Clod Count", str(summary['clod_count']))
        self._set_result("Clod Area Ratio", f"{summary['clod_area_ratio']}%")

        show_clod_analysis(self, clods, summary, vis_image, unit=summary['unit'])

    def calc_connectivity(self):
        """Analyze crack network connectivity using graph-theoretic metrics."""
        if self.final_image is None:
            messagebox.showwarning("Notice", "Please process the image first")
            return

        skel_bool = self._get_skeleton() > 0
        if skel_bool is None or not np.any(skel_bool):
            messagebox.showwarning("Notice", "No skeleton available. Process the image first.")
            return

        # Apply circular ROI if active
        if self.is_circular_mode and self.circular_mask is not None:
            skel_bool = np.logical_and(skel_bool, self.circular_mask > 0)
            if not np.any(skel_bool):
                messagebox.showwarning("Notice", "No skeleton within the circular ROI.")
                return

        self.log("Analyzing crack network connectivity...")

        result = analyze_crack_connectivity(skel_bool)

        self.log(
            f"Connectivity: J={result['junction_count']}, E={result['endpoint_count']}, "
            f"S={result['segment_count']}, C={result['component_count']}, "
            f"CI={result['connectivity_index']}, x={result['euler_number']}"
        )
        self._set_result("Conn Index", str(result['connectivity_index']))
        self._set_result("Euler Num", str(result['euler_number']))

        show_connectivity_analysis(
            self, result, skel_bool, result['vis_info']
        )

    def calc_junction_types(self):
        """Classify crack junctions as T, Y, X, or Multi type."""
        if self.final_image is None:
            messagebox.showwarning("Notice", "Please process the image first")
            return

        skel_bool = self._get_skeleton() > 0
        if skel_bool is None or not np.any(skel_bool):
            messagebox.showwarning("Notice", "No skeleton available. Process the image first.")
            return

        if self.is_circular_mode and self.circular_mask is not None:
            skel_bool = np.logical_and(skel_bool, self.circular_mask > 0)
            if not np.any(skel_bool):
                messagebox.showwarning("Notice", "No skeleton within the circular ROI.")
                return

        self.log("Classifying junction types (T / Y / X / Multi)...")

        result = classify_junctions(skel_bool)

        self._set_result("T/Y/X/Multi", f"{result['T_count']}/{result['Y_count']}/{result['X_count']}/{result['Multi_count']}")
        self.log(
            f"Junction types: T={result['T_count']} ({result['T_pct']}%), "
            f"Y={result['Y_count']} ({result['Y_pct']}%), "
            f"X={result['X_count']} ({result['X_pct']}%), "
            f"Multi={result['Multi_count']} ({result['Multi_pct']}%), "
            f"Total={result['total']}"
        )

        show_junction_classification(self, result, skel_bool)


    def start_calibration(self):
        if self.cv_image is None:
            messagebox.showwarning("Notice", "Please load an image first")
            return
        self.log("Opening calibration window...")
        CalibrationWindow(self, self.cv_image, on_calibrated=self._on_calibrated)

    def _on_calibrated(self, scale_factor, unit):
        self.scale_factor = scale_factor
        self.unit = unit
        self._update_status()
        self.log(f"Calibration successful! 1 {unit} = {self.scale_factor:.2f} px")

    def start_manual_edit(self):
        if self.final_image is None:
            messagebox.showwarning("Notice", "Please run binarization first")
            return
        self.log("Opening manual correction window...")
        ManualEditWindow(self, self.final_image, on_edit_done=self._on_edit_done)

    def _on_edit_done(self, edited_image):
        self.final_image = edited_image
        self._invalidate_analysis_cache()
        self.display_image(self.final_image, "Result")
        self.log("Manual correction saved!")
        messagebox.showinfo("Success", "Correction applied!")

    def select_circular_region(self):
        if self.cv_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        prev_center = self.last_saved_circle_center if hasattr(self, 'last_saved_circle_center') else None
        prev_radius = self.last_saved_circle_radius if hasattr(self, 'last_saved_circle_radius') else None
        CircularRegionWindow(
            self, self.cv_image,
            previous_center=prev_center,
            previous_radius=prev_radius,
            on_confirm=self._on_circle_confirmed,
            on_cancel=self._on_circle_cancelled
        )

    def _on_circle_confirmed(self, center, radius, mask):
        self.circle_center = center
        self.circle_radius = radius
        self.last_saved_circle_center = center
        self.last_saved_circle_radius = radius
        self.circular_mask = mask
        self.is_circular_mode = True
        self._update_status()
        self._schedule_binarization(immediate=True)
        self.log(f"Circular region set: center {center}, radius {radius}px")

    def _on_circle_cancelled(self):
        self.is_circular_mode = False
        self.circular_mask = None
        self._update_status()

    def disable_circular_mode(self):
        self.is_circular_mode = False
        self.circular_mask = None
        self.circle_center = None
        self.circle_radius = 0
        self._update_status()
        self._schedule_binarization(immediate=True)
        self.log("Circular mode disabled")


    def batch_process(self):
        sample_path = filedialog.askopenfilename(
            title="Select a Sample Image for Batch Setup",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not sample_path:
            return

        in_dir = os.path.dirname(sample_path)

        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(exts))
        if not files:
            messagebox.showinfo("Notice", "No images were found in the folder")
            return

        params = {
            'method': self.algo_var.get(),
            'v1': int(self.scale1_var.get()),
            'v2': self.scale2_var.get(),
            'denoise_mode': self.denoise_var.get(),
            'k_denoise': int(self.denoise_k_var.get()),
            'scale': self.scale_factor,
            'unit': self.unit,
        }

        def start_configured_batch(configured_params):
            out_dir = configured_params.pop('out_dir')
            crop_rect = configured_params.get('crop_rect')
            circle_roi = configured_params.get('circle_roi')
            roi_desc = (
                f"circle={circle_roi}" if circle_roi
                else f"crop={crop_rect}" if crop_rect
                else "full image"
            )
            self.log(
                f"Starting batch processing of {len(files)} images from "
                f"{os.path.basename(in_dir)}; {roi_desc}"
            )
            self._batch_processor = BatchProcessor(
                image_reader=self.read_image,
                image_saver=self.save_image_cv2,
                log_callback=self.log
            )
            self._batch_processor.start(files, in_dir, out_dir, configured_params)

        BatchSetupWindow(
            self, sample_path, self.read_image, params,
            on_start=start_configured_batch
        )


    def export_excel(self):
        if self.final_image is None:
            messagebox.showwarning("Warning", "No data to export")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")],
                                                 initialfile=f"Analysis_Result_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}")
        if file_path:
            try:
                export_excel(self.analysis_data, self.segments_data, file_path, log_callback=self.log)
                messagebox.showinfo("Export Succeeded", f"File saved: {file_path}")
            except Exception as e:
                self.log(f"Export failed: {e}")
                messagebox.showerror("Error", str(e))

    def export_csv(self):
        if self.final_image is None:
            messagebox.showwarning("Warning", "No data to export")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"Analysis_Result_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        if file_path:
            try:
                export_csv(self.analysis_data, self.segments_data, file_path, log_callback=self.log)
                messagebox.showinfo("Export Succeeded", f"File saved: {file_path}")
            except Exception as e:
                self.log(f"Export failed: {e}")
                messagebox.showerror("Error", str(e))

    def export_pdf(self):
        if self.final_image is None:
            messagebox.showwarning("Warning", "No data to export")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=f"Analysis_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        if not file_path:
            return
        try:
            export_pdf(
                self.analysis_data, self.segments_data,
                self.cv_image, self.binary_image, self.final_image,
                file_path, log_callback=self.log
            )
            messagebox.showinfo("Export Succeeded", f"File saved: {file_path}")
        except Exception as e:
            self.log(f"PDF export failed: {e}")
            messagebox.showerror("Error", str(e))

    def export_binary_image(self):
        if self.final_image is None:
            messagebox.showwarning("Warning", "No data to export")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=f"Binary_Result_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        if not file_path:
            return

        try:
            export_binary_image(
                self.final_image, file_path,
                is_circular_mode=self.is_circular_mode,
                circular_mask=self.circular_mask,
                log_callback=self.log
            )
            messagebox.showinfo("Export Succeeded", f"Image saved: {file_path}")
        except Exception as e:
            self.log(f"Export failed: {e}")
            messagebox.showerror("Error", str(e))

    # Display helper

    def display_image(self, cv_img, target):
        if cv_img is None:
            return
        # Lazy-create the ZoomableImageFrame on first real image
        if self.canvas_view is None and self._no_image_label is not None:
            self._no_image_label.pack_forget()
            self.canvas_view = ZoomableImageFrame(self._canvas_frame, cv_img, max_display_size=800)
            self.canvas_view.pack(fill=BOTH, expand=YES)
        if target in self._thumb_buttons:
            self._thumb_images[target] = cv_img
            if self._current_view is None:
                self._switch_view(target)
            elif self._current_view == target:
                self._switch_view(target)  # refresh canvas

    # UI event handlers

    def on_algo_change(self, event):
        self._on_manual_param_change()
        self.update_ui_controls()
        self._schedule_binarization(immediate=True)

    def update_ui_controls(self):
        method = self.algo_var.get()
        if "Sauvola" in method or "Niblack" in method:
            self.p2_frame.pack(fill=X, pady=5)
            self.p1_label.config(text="Window Size:")
            self.scale2.config(from_=0.0, to=1.0)
            self.scale2_var.set(0.2 if "Sauvola" in method else 0.12)
            self.scale2_val.config(text=f"{self.scale2_var.get():.2f}")
        elif "Adaptive" in method:
            self.p2_frame.pack(fill=X, pady=5)
            self.p1_label.config(text="Block Size:")
            self.scale2.config(from_=0, to=50)
            self.scale2_var.set(10)
            self.scale2_val.config(text=f"{int(self.scale2_var.get())}")
        else:
            self.p2_frame.pack_forget()
            self.p1_label.config(text="Global Threshold (0-255):")
            self.scale1_var.set(127)

    def update_binary_image(self, val=None):
        # If user manually moved a slider, disable auto mode
        self._on_manual_param_change()
        # Immediately update labels, but debounce the heavy processing
        v = int(self.scale1_var.get())
        if v % 2 == 0: v += 1
        self.scale1_val.config(text=str(v))
        v2 = self.scale2_var.get()
        if self.scale2.cget("to") == 50:
            self.scale2_val.config(text=str(int(v2)))
        else:
            self.scale2_val.config(text=f"{v2:.2f}")
        self._schedule_binarization()

    def update_denoise_image_event(self, val=None):
        # Immediately update label, but debounce the processing
        k = int(self.denoise_k_var.get())
        if k % 2 == 0: k += 1
        self.denoise_k_val.config(text=str(k))
        self._schedule_denoising()

def main():
    app = BinarizationApp()
    app.mainloop()

if __name__ == "__main__":
    main()
