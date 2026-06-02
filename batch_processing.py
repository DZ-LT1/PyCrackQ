"""Batch processing for crack image analysis."""

import os
import queue
import threading
import datetime
import math
import numpy as np
import cv2
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
from skimage.filters import threshold_sauvola, threshold_niblack
from skimage.morphology import skeletonize
from skimage.measure import label as sk_label, regionprops
from skimage.segmentation import watershed
from PyCrackQ.config import MIN_CRACK_AREA
from PyCrackQ.image_processing import detect_junctions
from PyCrackQ.batch_options import (
    build_segment_record,
    build_summary_record,
    create_circle_mask,
    crop_image,
    normalize_export_options,
)
from PyCrackQ.analysis import (
    calculate_accurate_metrics,
    calculate_segment_metrics,
    get_fractal_dim,
)


class BatchProcessor:
    """Batch image processor for crack analysis."""

    def __init__(self, image_reader, image_saver, log_callback=None):
        """Initialize the batch processor with I/O dependencies.

        Args:
            image_reader: Callable with signature (filepath, flags) -> image_array.
                          Example: cv2.imread or a custom wrapper.
            image_saver: Callable with signature (filepath, image_array) -> bool.
                         Must return True on success, False on failure.
            log_callback: Optional callable that receives a single string message
                          for logging progress and events.
        """
        self._image_reader = image_reader
        self._image_saver = image_saver
        self._log_callback = log_callback

        self._queue = None
        self._thread = None
        self._progress_win = None
        self._pbar = None
        self._lbl_stat = None

    def start(self, files, in_dir, out_dir, params):
        """Start batch processing of image files.

        Creates a progress window, starts a background thread for processing,
        and returns immediately. Progress is communicated via queue and
        displayed through periodic polling via _check_batch_queue.

        Args:
            files: List of image file names (str) to process.
            in_dir: Input directory path containing the image files.
            out_dir: Output directory path where results will be saved.
            params: Dictionary of processing parameters:
                - method (str): Binarization method.
                - v1 (int): Window size / threshold value.
                - v2 (float): k parameter.
                - denoise_mode (str): Denoising filter mode.
                - k_denoise (int): Denoising kernel size.
                - scale (float): Scale factor (pixels per unit).
                - unit (str): Physical unit name (e.g., "mm", "cm").
                - crop_rect (tuple|None): Optional (x1, y1, x2, y2) crop
                  coordinates selected on the sample image.
                - circle_roi (tuple|None): Optional (cx, cy, r) circular ROI
                  selected on the sample image.
                - export_options (dict): Output type and metric selection.
        """
        # Sanitize parameters.
        if params['v1'] < 3:
            params['v1'] = 3
        if params['v1'] % 2 == 0:
            params['v1'] += 1
        if params['k_denoise'] % 2 == 0:
            params['k_denoise'] += 1
        params['export_options'] = normalize_export_options(
            params.get('export_options')
        )

        # Create progress window.
        self._progress_win = tk.Toplevel()
        self._progress_win.title("Batch processing...")
        self._progress_win.geometry("400x150")
        self._progress_win.protocol("WM_DELETE_WINDOW", lambda: None)

        self._pbar = ttk.Progressbar(
            self._progress_win, maximum=len(files), mode='determinate',
            bootstyle="success"
        )
        self._pbar.pack(pady=20, padx=20, fill=tk.X)
        self._lbl_stat = ttk.Label(self._progress_win, text="Preparing to start...")
        self._lbl_stat.pack()

        if self._log_callback is not None:
            self._log_callback(f"Starting batch processing of {len(files)} images...")

        # Create queue and start background thread.
        self._queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run_batch_logic,
            args=(files, in_dir, out_dir, params),
            daemon=True,
        )
        self._thread.start()

        # Start polling the queue.
        self._progress_win.after(100, self._check_batch_queue)

    # ------------------------------------------------------------------
    # Private: background processing logic (runs on worker thread)
    # ------------------------------------------------------------------

    def _run_batch_logic(self, files, in_dir, out_dir, params):
        batch_summary = []
        batch_segments = []
        batch_errors = []
        success_count = 0
        skip_count = 0
        error_count = 0

        method = params['method']
        v1 = params['v1']
        v2 = params['v2']
        denoise_mode = params['denoise_mode']
        k_denoise = params['k_denoise']
        scale = params['scale']
        unit = params['unit']
        crop_rect = params.get('crop_rect')
        circle_roi = params.get('circle_roi')
        export_options = normalize_export_options(params.get('export_options'))

        os.makedirs(out_dir, exist_ok=True)
        for idx, fname in enumerate(files):
            fpath = os.path.normpath(os.path.join(in_dir, fname))

            # Notify progress.
            self._queue.put({
                "type": "progress", "idx": idx, "fname": fname,
                "total": len(files)
            })

            try:
                img = self._image_reader(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    skip_count += 1
                    batch_errors.append({
                        "File Name": fname, "Status": "Skipped",
                        "Reason": "Cannot read image"
                    })
                    self._queue.put({
                        "type": "log",
                        "msg": f"[Skip] Cannot read image: {fname}"
                    })
                    continue
                img = crop_image(img, crop_rect)
                circle_mask = create_circle_mask(img.shape, circle_roi) if circle_roi else None

                # Binarization.
                binary = None
                if "Global" in method:
                    _, binary = cv2.threshold(img, v1, 255, cv2.THRESH_BINARY)
                elif "Otsu" in method:
                    _, binary = cv2.threshold(img, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                elif "Triangle" in method:
                    _, binary = cv2.threshold(img, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
                elif "Sauvola" in method:
                    t = threshold_sauvola(img, window_size=v1, k=v2)
                    binary = ((img > t) * 255).astype(np.uint8)
                elif "Niblack" in method:
                    t = threshold_niblack(img, window_size=v1, k=v2)
                    binary = ((img > t) * 255).astype(np.uint8)
                elif "Adaptive Mean" in method:
                    binary = cv2.adaptiveThreshold(
                        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                        cv2.THRESH_BINARY, v1, int(v2)
                    )
                elif "Adaptive Gaussian" in method:
                    binary = cv2.adaptiveThreshold(
                        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, v1, int(v2)
                    )
                if binary is None:
                    raise ValueError(
                        f"Unknown or unsupported binarization algorithm: {method}"
                    )

                if cv2.countNonZero(binary) > binary.size / 2:
                    binary = cv2.bitwise_not(binary)
                if circle_mask is not None:
                    binary = cv2.bitwise_and(binary, binary, mask=circle_mask)

                # Denoising.
                final = binary.copy()
                if "Gaussian" in denoise_mode:
                    final = cv2.threshold(
                        cv2.GaussianBlur(binary, (k_denoise, k_denoise), 0),
                        127, 255, cv2.THRESH_BINARY
                    )[1]
                elif "Mean" in denoise_mode:
                    final = cv2.threshold(
                        cv2.blur(binary, (k_denoise, k_denoise)),
                        127, 255, cv2.THRESH_BINARY
                    )[1]
                elif "Median" in denoise_mode:
                    final = cv2.medianBlur(binary, k_denoise)
                if circle_mask is not None:
                    final = cv2.bitwise_and(final, final, mask=circle_mask)

                # Basic metrics.
                crack_px = cv2.countNonZero(final)
                area_phy = crack_px / (scale ** 2)
                roi_px = cv2.countNonZero(circle_mask) if circle_mask is not None else final.size
                ratio = (crack_px / max(roi_px, 1)) * 100
                bool_img = final > 0
                skel = skeletonize(bool_img)
                skel_bool = skel > 0

                dist_map = cv2.distanceTransform(final, cv2.DIST_L2, 5)
                metrics = calculate_accurate_metrics(
                    final, skel_bool, dist_map,
                    apply_roi=True, scale_factor=scale,
                    is_circular_mode=circle_mask is not None,
                    circular_mask=circle_mask
                )
                length_phy = metrics['length_phy']
                avg_w_phy = metrics['avg_width_phy']
                max_w_phy = metrics['max_width_phy']

                # Junction detection.
                _, _, j_count = detect_junctions(skel_bool)

                # Fractal dimension.
                sizes, counts = get_fractal_dim(bool_img)
                if sizes and counts and len(sizes) >= 2 and all(c > 0 for c in counts):
                    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
                    fractal_dim = -coeffs[0]
                else:
                    fractal_dim = 0

                # Connected component filtering.
                num_labels_img, labels_img, stats_img, _ = \
                    cv2.connectedComponentsWithStats(final, connectivity=8)
                clean_binary = np.zeros_like(final)
                for i in range(1, num_labels_img):
                    if stats_img[i, cv2.CC_STAT_AREA] >= MIN_CRACK_AREA:
                        clean_binary[labels_img == i] = 255

                smooth_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (3, 3)
                )
                smoothed_binary = cv2.morphologyEx(
                    clean_binary, cv2.MORPH_CLOSE, smooth_kernel
                )

                skel_smooth_bool = skeletonize(smoothed_binary > 0)
                junction_mask_smooth, _, _ = detect_junctions(skel_smooth_bool)

                skel_no_j = skel_smooth_bool.copy()
                skel_no_j[junction_mask_smooth] = False
                labeled_skel, _ = sk_label(
                    skel_no_j, connectivity=2, return_num=True
                )
                props_skel = regionprops(labeled_skel)

                valid_props = [
                    p for p in props_skel
                    if p.area >= max(1, MIN_CRACK_AREA)
                ]
                valid_props.sort(
                    key=lambda p: (int(p.centroid[0] / 50), p.centroid[1])
                )

                clean_labeled_skel = np.zeros_like(labeled_skel)
                for new_id, p in enumerate(valid_props, start=1):
                    for coords in p.coords:
                        clean_labeled_skel[coords[0], coords[1]] = new_id

                num_valid_labels = len(valid_props)
                dist_map_smooth = cv2.distanceTransform(
                    smoothed_binary, cv2.DIST_L2, 5
                )
                mask_smooth = smoothed_binary > 0
                expanded_labels = watershed(
                    -dist_map_smooth, clean_labeled_skel, mask=mask_smooth
                )

                # Visualization image.
                img_color = self._image_reader(fpath, cv2.IMREAD_COLOR)
                if img_color is None:
                    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img_color = crop_image(img_color, crop_rect)
                vis_img = (img_color * 0.4).astype(np.uint8)
                colors = np.random.randint(
                    50, 255, size=(num_valid_labels + 1, 3), dtype=np.uint8
                )
                colors[0] = [0, 0, 0]
                vis_img[expanded_labels > 0] = colors[
                    expanded_labels[expanded_labels > 0]
                ]
                if circle_mask is not None:
                    vis_img[circle_mask == 0] = [235, 235, 235]

                # Annotate segments.
                drawn_boxes = []
                font_scale = max(0.3, min(0.6, vis_img.shape[1] / 2000))
                thickness = 1

                for new_id, p in enumerate(valid_props, start=1):
                    coords = p.coords
                    segment_skel = (clean_labeled_skel == new_id)
                    segment_mask = (
                        (expanded_labels == new_id).astype(np.uint8) * 255
                    )
                    seg_metrics = calculate_segment_metrics(
                        segment_mask, segment_skel, dist_map_smooth,
                        apply_roi=True, scale_factor=scale,
                        is_circular_mode=circle_mask is not None,
                        circular_mask=circle_mask
                    )

                    if export_options.get("segment_excel"):
                        batch_segments.append(build_segment_record(
                            fname, new_id, unit,
                            seg_metrics['length_phy'],
                            seg_metrics['avg_width_phy'],
                            export_options,
                        ))

                    cy, cx = p.centroid
                    text = str(new_id)
                    (tw, th), base = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )

                    pad = 2
                    best_tx, best_ty = int(cx - tw / 2), int(cy)
                    placed = False
                    search_radius = 2.0
                    angle = 0.0

                    while not placed and search_radius < 150:
                        box1 = [
                            best_tx - pad,
                            best_ty - th - pad,
                            best_tx + tw + pad,
                            best_ty + base + pad,
                        ]
                        collision = False
                        for box2 in drawn_boxes:
                            if not (
                                box1[2] < box2[0] or box1[0] > box2[2]
                                or box1[3] < box2[1] or box1[1] > box2[3]
                            ):
                                collision = True
                                break
                        if not collision:
                            placed = True
                        else:
                            angle += 0.8
                            search_radius += 1.5
                            best_tx = int(
                                cx - tw / 2 + search_radius * math.cos(angle)
                            )
                            best_ty = int(
                                cy + search_radius * math.sin(angle)
                            )

                    best_tx = max(0, min(vis_img.shape[1] - tw, best_tx))
                    best_ty = max(th, min(vis_img.shape[0] - base, best_ty))

                    drawn_boxes.append([
                        best_tx - pad,
                        best_ty - th - pad,
                        best_tx + tw + pad,
                        best_ty + base + pad,
                    ])

                    dist_moved = math.hypot(
                        best_tx + tw / 2 - cx, best_ty - th / 2 - cy
                    )
                    if dist_moved > 15:
                        cv2.line(
                            vis_img,
                            (int(cx), int(cy)),
                            (int(best_tx + tw / 2), int(best_ty - th / 2)),
                            (150, 255, 150), 1, cv2.LINE_AA,
                        )
                        cv2.circle(
                            vis_img, (int(cx), int(cy)), 2,
                            (150, 255, 150), -1,
                        )

                    cv2.putText(
                        vis_img, text, (best_tx, best_ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), thickness + 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        vis_img, text, (best_tx, best_ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 255, 255), thickness, cv2.LINE_AA,
                    )

                if export_options.get("source_crops"):
                    source_path = os.path.join(out_dir, f"Cropped_{fname}")
                    source_img = img_color.copy()
                    if circle_mask is not None:
                        source_img[circle_mask == 0] = [255, 255, 255]
                    if not self._image_saver(source_path, source_img):
                        raise OSError(
                            f"Failed to save cropped source image: {source_path}"
                        )

                if export_options.get("binary_images"):
                    stem = os.path.splitext(fname)[0]
                    binary_path = os.path.join(out_dir, f"Binary_{stem}.png")
                    if not self._image_saver(binary_path, final):
                        raise OSError(
                            f"Failed to save binary image: {binary_path}"
                        )

                if export_options.get("result_images"):
                    res_path = os.path.join(out_dir, f"Result_{fname}")
                    if not self._image_saver(res_path, vis_img):
                        raise OSError(
                            f"Failed to save result image: {res_path}"
                        )

                if export_options.get("summary_excel"):
                    batch_summary.append(build_summary_record(
                        fname, unit,
                        {
                            "area": area_phy,
                            "area_ratio": ratio,
                            "length": length_phy,
                            "average_width": avg_w_phy,
                            "maximum_width": max_w_phy,
                            "junction_count": j_count,
                            "fractal_dimension": fractal_dim,
                            "segment_count": num_valid_labels,
                        },
                        export_options,
                    ))
                success_count += 1
            except Exception as e:
                error_count += 1
                err_msg = str(e)
                batch_errors.append({
                    "File Name": fname, "Status": "Failed", "Reason": err_msg
                })
                self._queue.put({
                    "type": "log",
                    "msg": f"[Error] Processing failed {fname}: {err_msg}",
                })

        # Save Excel report.
        save_path = os.path.join(
            out_dir,
            f"Batch_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        )
        try:
            needs_excel = (
                export_options.get("summary_excel")
                or export_options.get("segment_excel")
                or bool(batch_errors)
            )
            if needs_excel:
                with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                    if export_options.get("summary_excel"):
                        pd.DataFrame(batch_summary).to_excel(
                            writer, sheet_name='Summary Report', index=False
                        )
                    if export_options.get("segment_excel"):
                        pd.DataFrame(batch_segments).to_excel(
                            writer, sheet_name='All Crack Details', index=False
                        )
                    if batch_errors:
                        pd.DataFrame(batch_errors).to_excel(
                            writer, sheet_name='Failed and Skipped', index=False
                        )
            else:
                save_path = out_dir
            self._queue.put({
                "type": "done",
                "save_path": save_path,
                "success": True,
                "total": len(files),
                "processed": success_count,
                "skipped": skip_count,
                "failed": error_count,
            })
        except Exception as e:
            self._queue.put({
                "type": "done", "save_path": save_path,
                "success": False, "error": str(e),
            })

    # ------------------------------------------------------------------
    # Private: GUI polling (runs on main thread via .after)
    # ------------------------------------------------------------------

    def _check_batch_queue(self):
        try:
            while True:
                msg = self._queue.get_nowait()
                msg_type = msg.get("type")

                if msg_type == "progress":
                    self._lbl_stat.config(
                        text=f"Processing: {msg['fname']} "
                             f"({msg['idx'] + 1}/{msg['total']})"
                    )
                    self._pbar.step()

                elif msg_type == "log":
                    if self._log_callback is not None:
                        self._log_callback(msg["msg"])

                elif msg_type == "done":
                    if self._progress_win.winfo_exists():
                        self._progress_win.destroy()

                    if msg.get("success"):
                        messagebox.showinfo(
                            "Complete",
                            f"Batch processing completed!\n"
                            f"Total: {msg.get('total', 0)} images\n"
                            f"Succeeded: {msg.get('processed', 0)} images\n"
                            f"Skipped: {msg.get('skipped', 0)} images\n"
                            f"Failed: {msg.get('failed', 0)} images\n"
                            f"Report saved to: {msg['save_path']}",
                        )
                        if self._log_callback is not None:
                            self._log_callback(
                                f"Batch processing completed, succeeded "
                                f"{msg.get('processed', 0)} / "
                                f"{msg.get('total', 0)}, skipped "
                                f"{msg.get('skipped', 0)}, failed "
                                f"{msg.get('failed', 0)}, report: "
                                f"{msg['save_path']}"
                            )
                    else:
                        messagebox.showerror(
                            "Save Failed", msg.get("error", "Unknown Error")
                        )
                    return

        except queue.Empty:
            pass

        # Continue polling if thread is still alive or queue has pending messages.
        if self._thread.is_alive() or not self._queue.empty():
            self._progress_win.after(100, self._check_batch_queue)
