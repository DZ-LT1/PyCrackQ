import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
import ttkbootstrap as ttk
from ttkbootstrap.constants import *


class ManualEditWindow(tk.Toplevel):
    """Manual brush-based binary image editing window."""

    def __init__(self, parent, binary_image, on_edit_done=None):
        """Open manual editing window.

        Args:
            parent: Parent tk widget
            binary_image: The binary image to edit (uint8 numpy array)
            on_edit_done: Callback(edited_binary_image) when user saves
        """
        super().__init__(parent)
        self._on_edit_done = on_edit_done

        # --- Window setup ---
        self.title("Manual Correction - Left Click to Draw/Erase")
        win_w, win_h = 1100, 850
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{win_w}x{win_h}+{(sw - win_w) // 2}+{(sh - win_h) // 2}")
        self.focus_force()

        # --- Toolbar ---
        toolbar = ttk.Frame(self, padding=5)
        toolbar.pack(fill=X, side=TOP)

        ttk.Label(toolbar, text="Tool:", font=("Microsoft YaHei", 10, "bold")).pack(
            side=LEFT, padx=5
        )

        self._brush_mode_var = tk.IntVar(value=255)
        ttk.Radiobutton(
            toolbar,
            text="✏️ Repair (White)",
            variable=self._brush_mode_var,
            value=255,
            bootstyle="toolbutton-success",
        ).pack(side=LEFT, padx=5)
        ttk.Radiobutton(
            toolbar,
            text="🧼 Erase (Black)",
            variable=self._brush_mode_var,
            value=0,
            bootstyle="toolbutton-dark",
        ).pack(side=LEFT, padx=5)

        ttk.Label(
            toolbar, text=" | Brush Size:", font=("Microsoft YaHei", 10)
        ).pack(side=LEFT, padx=5)

        self._brush_size = ttk.Scale(
            toolbar, from_=1, to=20, value=3, orient=HORIZONTAL, length=150
        )
        self._brush_size.pack(side=LEFT, padx=5)

        ttk.Button(
            toolbar,
            text="💾 Save and Apply",
            style="3D.Primary.TButton",
            bootstyle="primary",
            command=self._save,
        ).pack(side=RIGHT, padx=10)

        # --- PIL image for editing ---
        self._pil_img = Image.fromarray(binary_image)
        if self._pil_img.mode != "L":
            self._pil_img = self._pil_img.convert("L")
        self._draw = ImageDraw.Draw(self._pil_img)

        # --- Canvas ---
        h, w = binary_image.shape[:2]
        self._scale = min((win_w - 40) / w, (win_h - 100) / h)
        display_w, display_h = int(w * self._scale), int(h * self._scale)
        self._canvas = tk.Canvas(
            self, width=display_w, height=display_h, bg="#333", cursor="spraycan"
        )
        self._canvas.pack(anchor=CENTER, expand=YES, pady=10)

        # --- Bindings ---
        self._last_pos = None
        self._update_display()
        self._canvas.bind("<Button-1>", self._on_start)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_end)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _update_display(self):
        w, h = self._pil_img.size
        new_w, new_h = int(w * self._scale), int(h * self._scale)
        display_img = self._pil_img.resize((new_w, new_h), Image.NEAREST)
        self._tk_img = ImageTk.PhotoImage(display_img)
        self._canvas.create_image(0, 0, anchor=NW, image=self._tk_img)

    # ------------------------------------------------------------------
    # Mouse handlers
    # ------------------------------------------------------------------

    def _on_start(self, event):
        self._last_pos = (event.x, event.y)
        self._paint(event)

    def _on_drag(self, event):
        self._paint(event)
        self._last_pos = (event.x, event.y)

    def _on_end(self, event):
        self._last_pos = None
        self._update_display()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def _paint(self, event):
        x = int(event.x / self._scale)
        y = int(event.y / self._scale)
        size = int(self._brush_size.get())
        color = self._brush_mode_var.get()

        if self._last_pos:
            last_x = int(self._last_pos[0] / self._scale)
            last_y = int(self._last_pos[1] / self._scale)
            self._draw.line([last_x, last_y, x, y], fill=color, width=size)
            r = size // 2
            self._draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        else:
            r = size // 2
            self._draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save(self):
        final_image = np.array(self._pil_img)
        if final_image.dtype != np.uint8:
            final_image = final_image.astype(np.uint8)
        _, final_image = cv2.threshold(final_image, 127, 255, cv2.THRESH_BINARY)

        if self._on_edit_done is not None:
            self._on_edit_done(final_image)

        self.destroy()
