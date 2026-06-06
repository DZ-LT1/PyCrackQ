"""Batch-processing setup window."""

import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk

from PyCrackQ.batch_options import (
    create_circle_mask,
    crop_image,
    mask_to_circle_roi,
    normalize_circle_roi,
    normalize_crop_rect,
    normalize_export_options,
)
from PyCrackQ.image_processing import apply_binarization, apply_denoising, recommend_parameters


class BatchSetupWindow(tk.Toplevel):
    """Configure crop, processing parameters, and output types for a batch."""

    def __init__(self, master, sample_path, image_reader, initial_params, on_start):
        super().__init__(master)
        self.master = master
        self.sample_path = sample_path
        self.image_reader = image_reader
        self.on_start = on_start
        self.sample_dir = os.path.dirname(sample_path)
        self.initial_params = initial_params

        self.title("Batch Processing Setup")
        self.geometry("1160x720")
        self.minsize(980, 620)

        self.sample_color = self.image_reader(sample_path, cv2.IMREAD_COLOR)
        self.sample_gray = self.image_reader(sample_path, cv2.IMREAD_GRAYSCALE)
        if self.sample_gray is None:
            messagebox.showerror("Error", "Cannot read the selected sample image")
            self.destroy()
            return
        if self.sample_color is None:
            self.sample_color = cv2.cvtColor(self.sample_gray, cv2.COLOR_GRAY2BGR)

        self.crop_rect = None
        self.circle_roi = None
        self.roi_mode_var = tk.StringVar(value="rectangle")
        self._drag_start = None
        self._preview_scale = 1.0
        self._preview_offset = (0, 0)
        self._preview_photo = None
        self._canvas_w = 720
        self._canvas_h = 480

        self._build_variables()
        self._build_ui()
        self._render_preview()
        self.grab_set()
        self.focus_set()

    def _build_variables(self):
        self.method_var = tk.StringVar(value=self.initial_params.get('method', 'Sauvola'))
        self.window_var = tk.IntVar(value=int(self.initial_params.get('v1', 25)))
        self.k_var = tk.DoubleVar(value=float(self.initial_params.get('v2', 0.2)))
        self.denoise_var = tk.StringVar(value=self.initial_params.get('denoise_mode', 'Median Filter'))
        self.denoise_k_var = tk.IntVar(value=int(self.initial_params.get('k_denoise', 3)))
        self.output_dir_var = tk.StringVar(value=os.path.join(self.sample_dir, "batch_results"))

        options = normalize_export_options()
        self.output_vars = {
            "result_images": tk.IntVar(value=1 if options["result_images"] else 0),
            "binary_images": tk.IntVar(value=1 if options["binary_images"] else 0),
            "source_crops": tk.IntVar(value=1 if options["source_crops"] else 0),
            "summary_excel": tk.IntVar(value=1 if options["summary_excel"] else 0),
            "segment_excel": tk.IntVar(value=1 if options["segment_excel"] else 0),
        }
        self.metric_vars = {
            key: tk.IntVar(value=1 if value else 0)
            for key, value in options["metrics"].items()
        }
        self.segment_field_vars = {
            key: tk.IntVar(value=1 if value else 0)
            for key, value in options["segment_fields"].items()
        }

    def _build_ui(self):
        shell = ttk.Frame(self, padding=10)
        shell.pack(fill=BOTH, expand=YES)

        left = ttk.Frame(shell)
        left.pack(side=LEFT, fill=BOTH, expand=YES, padx=(0, 10))
        right = ttk.Frame(shell, width=350)
        right.pack(side=RIGHT, fill=Y)
        right.pack_propagate(False)

        title = ttk.Label(
            left,
            text=f"Sample: {os.path.basename(self.sample_path)}",
            font=("Microsoft YaHei", 10, "bold"),
        )
        title.pack(anchor=W, pady=(0, 6))
        ttk.Label(
            left,
            text="Drag on the image to select a crop region. The same crop coordinates will be applied to all images in this folder.",
            font=("Microsoft YaHei", 9),
        ).pack(anchor=W, pady=(0, 8))

        self.canvas = tk.Canvas(left, width=self._canvas_w, height=self._canvas_h,
                                bg="#f8fafc", highlightthickness=1,
                                highlightbackground="#d8e0e8")
        self.canvas.pack(fill=BOTH, expand=YES)
        self.canvas.bind("<ButtonPress-1>", self._on_crop_press)
        self.canvas.bind("<B1-Motion>", self._on_crop_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_crop_release)

        crop_actions = ttk.Frame(left)
        crop_actions.pack(fill=X, pady=(8, 0))
        ttk.Radiobutton(crop_actions, text="Rectangle Crop", value="rectangle",
                        variable=self.roi_mode_var,
                        command=self._clear_roi).pack(side=LEFT, padx=(0, 8))
        ttk.Radiobutton(crop_actions, text="Circular ROI", value="circle",
                        variable=self.roi_mode_var,
                        command=self._clear_roi).pack(side=LEFT, padx=(0, 8))
        ttk.Button(crop_actions, text="Use Full Image", command=self._clear_crop,
                   bootstyle="secondary").pack(side=LEFT)
        self.crop_label = ttk.Label(crop_actions, text="Crop: full image")
        self.crop_label.pack(side=LEFT, padx=10)

        self._build_processing_panel(right)
        self._build_export_panel(right)

        footer = ttk.Frame(right)
        footer.pack(fill=X, pady=(12, 0))
        ttk.Button(footer, text="Start Batch", command=self._start_batch,
                   bootstyle="success").pack(side=RIGHT)
        ttk.Button(footer, text="Cancel", command=self.destroy,
                   bootstyle="secondary").pack(side=RIGHT, padx=(0, 8))

    def _build_processing_panel(self, parent):
        frame = ttk.Labelframe(parent, text=" Processing Parameters ", padding=10)
        frame.pack(fill=X, pady=(0, 10))

        ttk.Label(frame, text="Algorithm").pack(anchor=W)
        self.method_combo = ttk.Combobox(frame, textvariable=self.method_var, state="readonly")
        self.method_combo['values'] = (
            "Global Threshold", "Otsu", "Triangle", "Adaptive Mean",
            "Adaptive Gaussian", "Sauvola", "Niblack"
        )
        self.method_combo.bind("<<ComboboxSelected>>", lambda event: self._on_method_change())
        self.method_combo.pack(fill=X, pady=(2, 8))

        ttk.Label(frame, text="Window / Threshold").pack(anchor=W)
        self.window_scale = ttk.Scale(frame, from_=3, to=255, variable=self.window_var,
                                      command=lambda value: self._update_value_labels())
        self.window_scale.pack(fill=X)
        self.window_label = ttk.Label(frame, text="")
        self.window_label.pack(anchor=E, pady=(0, 6))

        ttk.Label(frame, text="k / C value").pack(anchor=W)
        self.k_scale = ttk.Scale(frame, from_=0, to=1, variable=self.k_var,
                                 command=lambda value: self._update_value_labels())
        self.k_scale.pack(fill=X)
        self.k_label = ttk.Label(frame, text="")
        self.k_label.pack(anchor=E, pady=(0, 6))

        ttk.Label(frame, text="Denoise").pack(anchor=W)
        self.denoise_combo = ttk.Combobox(frame, textvariable=self.denoise_var, state="readonly")
        self.denoise_combo['values'] = ("None", "Gaussian Filter", "Mean Filter", "Median Filter")
        self.denoise_combo.pack(fill=X, pady=(2, 8))

        ttk.Label(frame, text="Denoise Kernel").pack(anchor=W)
        self.denoise_scale = ttk.Scale(frame, from_=1, to=15, variable=self.denoise_k_var,
                                       command=lambda value: self._update_value_labels())
        self.denoise_scale.pack(fill=X)
        self.denoise_label = ttk.Label(frame, text="")
        self.denoise_label.pack(anchor=E, pady=(0, 8))

        ttk.Button(frame, text="Auto Recommend From Crop", command=self._auto_recommend,
                   bootstyle="info").pack(fill=X)
        ttk.Button(frame, text="Preview Current Settings", command=self._preview_current_settings,
                   bootstyle="success").pack(fill=X, pady=(6, 0))
        self.auto_label = ttk.Label(frame, text="", wraplength=310)
        self.auto_label.pack(fill=X, pady=(6, 0))
        self._on_method_change()
        self._update_value_labels()

    def _on_method_change(self):
        method = self.method_var.get()
        if "Adaptive" in method:
            self.k_scale.configure(from_=0, to=50)
            if self.k_var.get() <= 1:
                self.k_var.set(10)
        else:
            self.k_scale.configure(from_=0, to=1)
            if self.k_var.get() > 1:
                self.k_var.set(0.2)
        self._update_value_labels()

    def _build_export_panel(self, parent):
        frame = ttk.Labelframe(parent, text=" Export Options ", padding=10)
        frame.pack(fill=BOTH, expand=YES)

        ttk.Label(frame, text="Output folder").pack(anchor=W)
        row = ttk.Frame(frame)
        row.pack(fill=X, pady=(2, 8))
        ttk.Entry(row, textvariable=self.output_dir_var).pack(side=LEFT, fill=X, expand=YES)
        ttk.Button(row, text="Browse", command=self._browse_output,
                   bootstyle="secondary").pack(side=RIGHT, padx=(6, 0))

        for key, label in [
            ("result_images", "Result images with segment labels"),
            ("binary_images", "Binary crack masks"),
            ("source_crops", "Cropped source images"),
            ("summary_excel", "Excel summary indicators"),
            ("segment_excel", "Excel segment details"),
        ]:
            ttk.Checkbutton(frame, text=label, variable=self.output_vars[key],
                            bootstyle="success-round-toggle").pack(anchor=W, pady=2)

        ttk.Separator(frame).pack(fill=X, pady=8)
        ttk.Label(frame, text="Summary indicators").pack(anchor=W)
        for key, label in [
            ("area", "Crack area"),
            ("area_ratio", "Area ratio"),
            ("length", "Total length"),
            ("average_width", "Average width"),
            ("maximum_width", "Maximum width"),
            ("junction_count", "Junction count"),
            ("fractal_dimension", "Fractal dimension"),
            ("segment_count", "Segment count"),
        ]:
            ttk.Checkbutton(frame, text=label, variable=self.metric_vars[key]).pack(anchor=W)

        ttk.Separator(frame).pack(fill=X, pady=8)
        ttk.Label(frame, text="Segment detail fields").pack(anchor=W)
        ttk.Checkbutton(frame, text="Length", variable=self.segment_field_vars["length"]).pack(anchor=W)
        ttk.Checkbutton(frame, text="Width", variable=self.segment_field_vars["width"]).pack(anchor=W)

    def _update_value_labels(self):
        window = int(self.window_var.get())
        if window % 2 == 0:
            window += 1
        denoise_k = int(self.denoise_k_var.get())
        if denoise_k % 2 == 0:
            denoise_k += 1
        self.window_label.config(text=str(window))
        self.k_label.config(text=f"{float(self.k_var.get()):.2f}")
        self.denoise_label.config(text=str(denoise_k))

    def _render_preview(self):
        h, w = self.sample_color.shape[:2]
        scale = min(self._canvas_w / w, self._canvas_h / h)
        scale = min(scale, 1.0)
        display_w = max(1, int(w * scale))
        display_h = max(1, int(h * scale))
        offset_x = (self._canvas_w - display_w) // 2
        offset_y = (self._canvas_h - display_h) // 2
        self._preview_scale = scale
        self._preview_offset = (offset_x, offset_y)

        rgb = cv2.cvtColor(self.sample_color, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((display_w, display_h), Image.Resampling.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, anchor=NW, image=self._preview_photo)
        self._draw_crop_rect()

    def _event_to_image(self, event):
        x0, y0 = self._preview_offset
        x = int((event.x - x0) / self._preview_scale)
        y = int((event.y - y0) / self._preview_scale)
        h, w = self.sample_gray.shape[:2]
        return max(0, min(w, x)), max(0, min(h, y))

    def _image_rect_to_canvas(self, rect):
        if rect is None:
            return None
        x0, y0 = self._preview_offset
        scale = self._preview_scale
        left, top, right, bottom = rect
        return (
            x0 + left * scale,
            y0 + top * scale,
            x0 + right * scale,
            y0 + bottom * scale,
        )

    def _draw_crop_rect(self):
        self.canvas.delete("crop")
        rect = normalize_crop_rect(self.crop_rect, self.sample_gray.shape)
        circle = normalize_circle_roi(self.circle_roi, self.sample_gray.shape)
        if rect is None and circle is None:
            self.crop_label.config(text="Crop: full image")
            return
        if rect is not None:
            canvas_rect = self._image_rect_to_canvas(rect)
            self.canvas.create_rectangle(*canvas_rect, outline="#18c878", width=2, tags="crop")
            left, top, right, bottom = rect
            self.crop_label.config(
                text=f"Crop: x={left}, y={top}, w={right-left}, h={bottom-top}"
            )
        if circle is not None:
            cx, cy, radius = circle
            canvas_circle = self._image_rect_to_canvas(
                (cx - radius, cy - radius, cx + radius, cy + radius)
            )
            self.canvas.create_oval(*canvas_circle, outline="#18c878", width=2, tags="crop")
            self.crop_label.config(
                text=f"Circle ROI: cx={cx}, cy={cy}, r={radius}"
            )

    def _on_crop_press(self, event):
        self._drag_start = self._event_to_image(event)

    def _on_crop_drag(self, event):
        if self._drag_start is None:
            return
        x, y = self._event_to_image(event)
        x0, y0 = self._drag_start
        if self.roi_mode_var.get() == "circle":
            radius = int(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5)
            self.circle_roi = (x0, y0, radius)
            self.crop_rect = None
        else:
            self.crop_rect = (x0, y0, x, y)
            self.circle_roi = None
        self._draw_crop_rect()

    def _on_crop_release(self, event):
        if self._drag_start is None:
            return
        x, y = self._event_to_image(event)
        x0, y0 = self._drag_start
        if self.roi_mode_var.get() == "circle":
            radius = int(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5)
            self.circle_roi = normalize_circle_roi((x0, y0, radius), self.sample_gray.shape)
            self.crop_rect = None
        else:
            self.crop_rect = normalize_crop_rect((x0, y0, x, y), self.sample_gray.shape)
            self.circle_roi = None
        self._drag_start = None
        self._draw_crop_rect()

    def _clear_crop(self):
        self._clear_roi()

    def _clear_roi(self):
        self.crop_rect = None
        self.circle_roi = None
        self._draw_crop_rect()

    def _current_sample_for_recommendation(self):
        rect = normalize_crop_rect(self.crop_rect, self.sample_gray.shape)
        if rect is None:
            circle = normalize_circle_roi(self.circle_roi, self.sample_gray.shape)
            if circle is None:
                return self.sample_gray
            return mask_to_circle_roi(self.sample_gray, circle)
        left, top, right, bottom = rect
        return self.sample_gray[top:bottom, left:right]

    def _auto_recommend(self):
        try:
            params = recommend_parameters(self._current_sample_for_recommendation())
        except Exception as exc:
            messagebox.showerror("Auto Recommend Failed", str(exc))
            return
        self.method_var.set(params["method"])
        self.window_var.set(params["window"])
        self.k_var.set(params["k"])
        self._on_method_change()
        self._update_value_labels()
        self.auto_label.config(
            text=(
                f"{params['method']}, win={params['window']}, k={params['k']:.2f}; "
                f"{params.get('illumination_level', 'unknown')} light, "
                f"{params.get('complexity_level', 'unknown')} cracks"
            )
        )

    def _sample_with_current_roi(self):
        source_gray = crop_image(self.sample_gray, self.crop_rect)
        source_color = crop_image(self.sample_color, self.crop_rect)
        circle = None
        if self.crop_rect is None:
            circle = normalize_circle_roi(self.circle_roi, source_gray.shape)
            if circle is not None:
                source_color = mask_to_circle_roi(source_color, circle, outside_value=255)
        return source_gray, source_color, circle

    def _preview_current_settings(self):
        try:
            source_gray, source_color, circle = self._sample_with_current_roi()
            method = self.method_var.get()
            window = int(self.window_var.get())
            if window < 3:
                window = 3
            if window % 2 == 0:
                window += 1
            k_value = float(self.k_var.get())
            denoise_k = int(self.denoise_k_var.get())
            if denoise_k < 1:
                denoise_k = 1
            if denoise_k % 2 == 0:
                denoise_k += 1

            binary = apply_binarization(source_gray, method, window, k_value)
            if circle is not None:
                mask = create_circle_mask(binary.shape, circle)
                binary = cv2.bitwise_and(binary, binary, mask=mask)
            final = apply_denoising(binary, self.denoise_var.get(), denoise_k)
            if circle is not None:
                final = cv2.bitwise_and(final, final, mask=create_circle_mask(final.shape, circle))
            self._open_preview_window(source_color, binary, final)
        except Exception as exc:
            messagebox.showerror("Preview Failed", str(exc))

    def _preview_photo_from_array(self, image, max_size=(300, 240)):
        if image.ndim == 2:
            pil_img = Image.fromarray(image)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil_img)

    def _open_preview_window(self, source, binary, final):
        win = tk.Toplevel(self)
        win.title("Batch Preview")
        win.geometry("980x340")
        win._preview_refs = []
        row = ttk.Frame(win, padding=10)
        row.pack(fill=BOTH, expand=YES)
        for title, image in [
            ("Source / ROI", source),
            ("Binary", binary),
            ("Denoised Result", final),
        ]:
            panel = ttk.Frame(row)
            panel.pack(side=LEFT, fill=BOTH, expand=YES, padx=6)
            ttk.Label(panel, text=title, font=("Microsoft YaHei", 10, "bold")).pack(pady=(0, 6))
            photo = self._preview_photo_from_array(image)
            win._preview_refs.append(photo)
            ttk.Label(panel, image=photo).pack(fill=BOTH, expand=YES)

    def _browse_output(self):
        folder = filedialog.askdirectory(
            title="Select Batch Output Folder",
            initialdir=self.sample_dir,
            parent=self,
        )
        if folder:
            self.output_dir_var.set(folder)

    def _collect_export_options(self):
        return normalize_export_options({
            key: bool(var.get()) for key, var in self.output_vars.items()
        } | {
            "metrics": {
                key: bool(var.get()) for key, var in self.metric_vars.items()
            },
            "segment_fields": {
                key: bool(var.get()) for key, var in self.segment_field_vars.items()
            },
        })

    def _start_batch(self):
        self._update_value_labels()
        output_dir = self.output_dir_var.get().strip()
        if not output_dir:
            messagebox.showwarning("Warning", "Please select an output folder")
            return

        export_options = self._collect_export_options()
        if not any(
            export_options.get(key)
            for key in ("result_images", "binary_images", "source_crops", "summary_excel", "segment_excel")
        ):
            messagebox.showwarning("Warning", "Please select at least one export type")
            return

        window = int(self.window_var.get())
        if window < 3:
            window = 3
        if window % 2 == 0:
            window += 1
        denoise_k = int(self.denoise_k_var.get())
        if denoise_k < 1:
            denoise_k = 1
        if denoise_k % 2 == 0:
            denoise_k += 1

        params = dict(self.initial_params)
        params.update({
            "method": self.method_var.get(),
            "v1": window,
            "v2": float(self.k_var.get()),
            "denoise_mode": self.denoise_var.get(),
            "k_denoise": denoise_k,
            "crop_rect": normalize_crop_rect(self.crop_rect, self.sample_gray.shape),
            "circle_roi": (
                normalize_circle_roi(self.circle_roi, self.sample_gray.shape)
                if self.crop_rect is None else None
            ),
            "export_options": export_options,
            "out_dir": output_dir,
        })
        self.grab_release()
        self.destroy()
        self.on_start(params)
