"""Circular ROI selection window."""

import tkinter as tk
from tkinter import messagebox
import math
import numpy as np
import cv2
from PIL import Image, ImageTk
import ttkbootstrap as ttk

from PyCrackQ.image_processing import create_circular_mask


class CircularRegionWindow(tk.Toplevel):
    """Circular ROI selection window."""

    def __init__(self, parent, source_image, previous_center=None,
                 previous_radius=None, on_confirm=None, on_cancel=None):
        """Open circular region selection window.

        Args:
            parent: Parent tk widget.
            source_image: The source cv2 image (grayscale or BGR).
            previous_center: (x, y) tuple of previously saved center, or None.
            previous_radius: Previously saved radius, or None.
            on_confirm: Callback(center_xy, radius, circular_mask) when the
                user confirms a selection.
            on_cancel: Callback() when the user cancels.
        """
        super().__init__(parent)
        self.title("Select Circular Region - Click Center, Drag Radius")
        self.geometry("900x700")

        # ---- store inputs ----
        self._source_image = source_image
        self._previous_center = previous_center
        self._previous_radius = previous_radius
        self._on_confirm = on_confirm
        self._on_cancel = on_cancel

        # ---- build UI ----
        self._canvas = tk.Canvas(self, bg='#2b2b2b', highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Convert source image for display (PIL).
        img_display = source_image.copy()
        if len(img_display.shape) == 2:
            self._pil_img = Image.fromarray(img_display, mode='L')
        else:
            self._pil_img = Image.fromarray(
                cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))

        self._img_id = self._canvas.create_image(0, 0, anchor=tk.CENTER)

        # ---- selection state ----
        self._center = None           # scaled-image coords
        self._display_center = None   # canvas coords
        self._temp_radius = 0         # scaled-image radius
        self._orig_center = None      # original-image coords
        self._orig_radius = 0         # original-image radius

        # ---- mouse bindings ----
        self._canvas.bind("<Configure>", self._on_resize)
        self._canvas.bind("<Button-1>", self._on_center_click)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)

        # ---- button bar ----
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Confirm Selection",
                   style="3D.Success.TButton",
                   command=self._confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel",
                   style="3D.Secondary.TButton",
                   command=self._cancel).pack(side=tk.LEFT, padx=5)
        if previous_center is not None and previous_radius is not None:
            ttk.Button(btn_frame, text="Use Previous Settings",
                       style="3D.Warning.TButton",
                       command=self._use_previous).pack(side=tk.LEFT, padx=5)

        self.update_idletasks()
        self.focus_force()

    # ------------------------------------------------------------------
    #  Event handlers
    # ------------------------------------------------------------------

    def _on_resize(self, event):
        """Re-scale the displayed image when the canvas is resized."""
        canvas_w = event.width
        canvas_h = event.height
        if canvas_w < 10 or canvas_h < 10:
            return

        img_w, img_h = self._pil_img.size
        margin = 40
        avail_w = max(10, canvas_w - margin)
        avail_h = max(10, canvas_h - margin)

        scale = min(avail_w / img_w, avail_h / img_h)
        new_w = max(int(img_w * scale), 10)
        new_h = max(int(img_h * scale), 10)

        resized_img = self._pil_img.resize((new_w, new_h),
                                           Image.Resampling.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(resized_img)

        cx, cy = canvas_w // 2, canvas_h // 2
        self._canvas.coords(self._img_id, cx, cy)
        self._canvas.itemconfig(self._img_id, image=self._tk_img)

        # Store layout metadata on the canvas for later hit-testing.
        self._canvas.img_scale = scale
        self._canvas.img_width = new_w
        self._canvas.img_height = new_h
        self._canvas.img_cx = cx
        self._canvas.img_cy = cy

        # Re-position existing preview if the user already picked a center.
        if self._orig_center is not None:
            img_left = cx - new_w // 2
            img_top = cy - new_h // 2
            rel_x = self._orig_center[0] * scale
            rel_y = self._orig_center[1] * scale
            self._center = (rel_x, rel_y)
            self._display_center = (img_left + rel_x, img_top + rel_y)
            self._temp_radius = self._orig_radius * scale
            self._redraw_preview()

    def _redraw_preview(self):
        """Redraw the center dot and radius circle on the canvas."""
        self._canvas.delete("circle_preview")
        if self._display_center is None:
            return
        cx, cy = self._display_center
        self._canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                                 fill='red', tags="circle_preview")
        if getattr(self, '_temp_radius', 0) > 0:
            r = self._temp_radius
            self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                     outline='blue', width=2,
                                     tags="circle_preview")

    def _on_center_click(self, event):
        """Record the center point on left-click."""
        if not hasattr(self._canvas, 'img_cx'):
            return
        img_left = self._canvas.img_cx - self._canvas.img_width // 2
        img_top = self._canvas.img_cy - self._canvas.img_height // 2

        rel_x = event.x - img_left
        rel_y = event.y - img_top

        # Ignore clicks outside the image.
        if (rel_x < 0 or rel_y < 0
                or rel_x > self._canvas.img_width
                or rel_y > self._canvas.img_height):
            return

        self._display_center = (event.x, event.y)
        self._center = (rel_x, rel_y)
        self._temp_radius = 0

        scale = self._canvas.img_scale
        self._orig_center = (rel_x / scale, rel_y / scale)
        self._orig_radius = 0

        self._redraw_preview()

    def _on_drag(self, event):
        """Update the radius while the user drags."""
        if self._center is None:
            return
        cx, cy = self._display_center
        self._temp_radius = math.sqrt((event.x - cx) ** 2
                                      + (event.y - cy) ** 2)

        scale = self._canvas.img_scale
        self._orig_radius = self._temp_radius / scale

        self._redraw_preview()

    def _on_release(self, event):
        """Forward release event to the drag handler for final radius."""
        self._on_drag(event)

    # ------------------------------------------------------------------
    #  Button commands
    # ------------------------------------------------------------------

    def _confirm(self):
        """Validate the selection and fire the confirm callback."""
        if (getattr(self, '_orig_center', None) is None
                or getattr(self, '_orig_radius', 0) <= 0):
            return  # silently ignore -- caller may show its own warning

        center_x = int(self._orig_center[0])
        center_y = int(self._orig_center[1])
        radius = int(self._orig_radius)

        mask = create_circular_mask(self._source_image.shape,
                                    (center_x, center_y), radius)

        if self._on_confirm is not None:
            self._on_confirm((center_x, center_y), radius, mask)

        self.destroy()

    def _cancel(self):
        """Fire the cancel callback and close the window."""
        if self._on_cancel is not None:
            self._on_cancel()
        self.destroy()

    def _use_previous(self):
        """Apply the previously-saved circle settings (with bounds checks)."""
        if self._previous_center is None or self._previous_radius is None:
            return

        h, w = self._source_image.shape[:2]
        cx, cy = self._previous_center
        r = self._previous_radius

        # Silently reject out-of-bounds values.
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return
        if r <= 0 or r > min(w, h):
            return

        mask = create_circular_mask(self._source_image.shape,
                                    (cx, cy), r)

        if self._on_confirm is not None:
            self._on_confirm((cx, cy), r, mask)

        self.destroy()
