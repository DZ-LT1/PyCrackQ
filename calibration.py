"""Physical-distance calibration window."""

import tkinter as tk
from tkinter import simpledialog, messagebox
import math
import cv2
import numpy as np
from PIL import Image, ImageTk
import ttkbootstrap as ttk


class CalibrationWindow(tk.Toplevel):
    """Physical distance calibration with zoom and pan."""

    def __init__(self, parent, source_image, on_calibrated=None):
        """Open calibration window.

        Args:
            parent: Parent tk widget
            source_image: The source cv2 image (grayscale or BGR)
            on_calibrated: Callback(scale_factor, unit) when calibration is confirmed
        """
        super().__init__(parent)
        self._source_image = source_image
        self._on_calibrated = on_calibrated

        # --- prepare display image ---
        if len(source_image.shape) == 2:
            img_disp = cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR)
        else:
            img_disp = source_image.copy()

        h, w = img_disp.shape[:2]
        self._base_img = img_disp

        win_w, win_h = 1000, 800
        img_scale = min((win_w - 80) / w, (win_h - 130) / h)
        self._img_scale = max(0.05, min(img_scale, 1.0))

        self._start_img = None
        self._end_img = None
        self._display_width = 0
        self._display_height = 0
        self._img_offset_x = 0
        self._img_offset_y = 0

        # --- window setup ---
        self.title("Physical Calibration - Zoom with Wheel, Drag Reference Line")
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{win_w}x{win_h}+{(sw - win_w) // 2}+{(sh - win_h) // 2}")
        self.focus_force()

        # --- UI ---
        frame = ttk.Frame(self)
        frame.pack(expand=True, fill="both")

        v_scroll = ttk.Scrollbar(frame, orient="vertical")
        h_scroll = ttk.Scrollbar(frame, orient="horizontal")

        self._canvas = tk.Canvas(
            frame,
            bg="#2b2b2b",
            cursor="crosshair",
            highlightthickness=0,
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set,
        )
        v_scroll.config(command=self._canvas.yview)
        h_scroll.config(command=self._canvas.xview)

        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        self._canvas.pack(side="left", expand=True, fill="both")

        self._img_id = self._canvas.create_image(0, 0, anchor="nw")
        self._redraw_calibration_canvas()

        tip_lbl = ttk.Label(
            self,
            text="Instructions: use the mouse wheel to zoom, right/middle mouse to pan, "
                 "and left-drag the reference line from start to end",
            bootstyle="inverse-info",
            padding=10,
        )
        tip_lbl.pack(fill="x", side="bottom")

        # --- bindings ---
        self._canvas.bind("<Button-1>", self.on_calib_press_new)
        self._canvas.bind("<B1-Motion>", self.on_calib_drag_new)
        self._canvas.bind("<ButtonRelease-1>", self.on_calib_release_new)
        self._canvas.bind("<MouseWheel>", self._on_calib_mousewheel)
        self._canvas.bind("<Button-4>", self._on_calib_mousewheel)
        self._canvas.bind("<Button-5>", self._on_calib_mousewheel)
        self._canvas.bind("<ButtonPress-3>", self._start_calib_pan)
        self._canvas.bind("<B3-Motion>", self._do_calib_pan)
        self._canvas.bind("<ButtonPress-2>", self._start_calib_pan)
        self._canvas.bind("<B2-Motion>", self._do_calib_pan)
        self._canvas.bind("<Configure>", self._update_calibration_canvas_position)

    # ------------------------------------------------------------------
    # Drawing / layout helpers
    # ------------------------------------------------------------------

    def _redraw_calibration_canvas(self):
        h, w = self._base_img.shape[:2]
        new_w = max(1, int(w * self._img_scale))
        new_h = max(1, int(h * self._img_scale))
        self._display_width = new_w
        self._display_height = new_h
        resized = cv2.resize(self._base_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        self._tk_img = ImageTk.PhotoImage(img_pil)
        self._canvas.itemconfig(self._img_id, image=self._tk_img)
        self._update_calibration_canvas_position()
        self._draw_calibration_line()

    def _update_calibration_canvas_position(self, event=None):
        if self._canvas is None or not hasattr(self, '_img_id'):
            return
        canvas_w = max(self._canvas.winfo_width(), 1)
        canvas_h = max(self._canvas.winfo_height(), 1)
        img_w = getattr(self, '_display_width', 0)
        img_h = getattr(self, '_display_height', 0)
        self._img_offset_x = max((canvas_w - img_w) // 2, 0)
        self._img_offset_y = max((canvas_h - img_h) // 2, 0)
        self._canvas.coords(self._img_id, self._img_offset_x, self._img_offset_y)
        self._canvas.config(
            scrollregion=(
                0,
                0,
                max(img_w + self._img_offset_x, canvas_w),
                max(img_h + self._img_offset_y, canvas_h),
            )
        )
        self._draw_calibration_line()

    def _draw_calibration_line(self):
        if self._canvas is None:
            return
        self._canvas.delete("calib_line")
        if self._start_img is None or self._end_img is None:
            return
        sx, sy = self._start_img
        ex, ey = self._end_img
        self._canvas.create_line(
            self._img_offset_x + sx * self._img_scale,
            self._img_offset_y + sy * self._img_scale,
            self._img_offset_x + ex * self._img_scale,
            self._img_offset_y + ey * self._img_scale,
            fill="red",
            width=3,
            tags="calib_line",
        )

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _calib_event_to_image_point(self, event):
        canvas_x = self._canvas.canvasx(event.x)
        canvas_y = self._canvas.canvasy(event.y)
        h, w = self._base_img.shape[:2]
        x = max(0.0, min(w - 1.0, (canvas_x - self._img_offset_x) / self._img_scale))
        y = max(0.0, min(h - 1.0, (canvas_y - self._img_offset_y) / self._img_scale))
        return x, y

    # ------------------------------------------------------------------
    # Mouse wheel (zoom)
    # ------------------------------------------------------------------

    def _on_calib_mousewheel(self, event):
        old_scale = self._img_scale
        if getattr(event, 'num', 0) == 4 or getattr(event, 'delta', 0) > 0:
            self._img_scale *= 1.15
        elif getattr(event, 'num', 0) == 5 or getattr(event, 'delta', 0) < 0:
            self._img_scale /= 1.15
        self._img_scale = max(0.05, min(self._img_scale, 20.0))
        if abs(self._img_scale - old_scale) < 1e-6:
            return
        self._redraw_calibration_canvas()

    # ------------------------------------------------------------------
    # Pan (right / middle button)
    # ------------------------------------------------------------------

    def _start_calib_pan(self, event):
        self._canvas.scan_mark(event.x, event.y)

    def _do_calib_pan(self, event):
        self._canvas.scan_dragto(event.x, event.y, gain=1)

    # ------------------------------------------------------------------
    # Left-button draw (reference line)
    # ------------------------------------------------------------------

    def on_calib_press_new(self, event):
        self._start_img = self._calib_event_to_image_point(event)
        self._end_img = self._start_img
        self._draw_calibration_line()

    def on_calib_drag_new(self, event):
        if self._start_img is None:
            return
        self._end_img = self._calib_event_to_image_point(event)
        self._draw_calibration_line()

    def on_calib_release_new(self, event):
        if self._start_img is None:
            return
        self._end_img = self._calib_event_to_image_point(event)
        self._draw_calibration_line()
        sx, sy = self._start_img
        ex, ey = self._end_img
        dist_px_original = math.sqrt((sx - ex) ** 2 + (sy - ey) ** 2)
        if dist_px_original < 5:
            return
        real_len = simpledialog.askfloat(
            "Physical Calibration",
            f"Original image distance: {dist_px_original:.2f} px\n\n"
            f"Enter the actual physical length of this segment:",
            parent=self,
        )
        if real_len and real_len > 0:
            unit = simpledialog.askstring(
                "Unit", "Enter unit (mm, cm, um...):",
                parent=self, initialvalue="mm",
            )
            if not unit:
                unit = "unit"
            scale_factor = dist_px_original / real_len
            if self._on_calibrated is not None:
                self._on_calibrated(scale_factor, unit)
            messagebox.showinfo(
                "Success",
                f"Calibration complete! Unit for later analysis: {unit}",
            )
            self.destroy()
        else:
            # User cancelled – window stays open so they can try again
            pass
