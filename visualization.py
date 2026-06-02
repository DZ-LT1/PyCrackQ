import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import YES, BOTH, RIGHT, LEFT, BOTTOM, X, Y, VERTICAL, HORIZONTAL


class ZoomableImageFrame(ttk.Frame):
    """A reusable frame that displays an OpenCV image with wheel zoom, right-drag
    pan, and a control bar with zoom indicator, save, and reset buttons.

    Args:
        parent: Parent tk widget.
        cv_image: OpenCV image (BGR color or grayscale, uint8 numpy array).
        max_display_size: Initial maximum display dimension in pixels (default 550).
    """

    def __init__(self, parent, cv_image, max_display_size=550):
        super().__init__(parent)
        self._cv_image = cv_image
        if len(cv_image.shape) == 2:
            h, w = cv_image.shape
        else:
            h, w = cv_image.shape[:2]
        self._img_h, self._img_w = h, w
        self._max_display = max_display_size
        self._updating = False  # re-entrancy guard

        initial_scale = min(max_display_size / w, max_display_size / h, 1.0)
        self._state = {
            "scale": initial_scale,
            "img_id": None,
            "tk_img": None,
            "new_w": 0,
            "new_h": 0,
        }

        # Canvas + scrollbars
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self._v_scroll = ttk.Scrollbar(canvas_frame, orient=VERTICAL)
        self._h_scroll = ttk.Scrollbar(canvas_frame, orient=HORIZONTAL)
        self._canvas = tk.Canvas(canvas_frame, bg="#2b2b2b",
                                 yscrollcommand=self._v_scroll.set,
                                 xscrollcommand=self._h_scroll.set)
        self._v_scroll.config(command=self._canvas.yview)
        self._h_scroll.config(command=self._canvas.xview)

        self._v_scroll.pack(side=RIGHT, fill=Y)
        self._h_scroll.pack(side=BOTTOM, fill=X)
        self._canvas.pack(side=LEFT, expand=YES, fill=BOTH)

        # Control bar
        ctrl_bar = ttk.Frame(self)
        ctrl_bar.pack(fill=X, side=BOTTOM, pady=(2, 0))

        self._info_label = ttk.Label(ctrl_bar, text="", bootstyle="inverse-secondary", padding=4)
        self._info_label.pack(side=LEFT, fill=X, expand=True)

        ttk.Button(ctrl_bar, text="Reset", bootstyle="secondary-outline",
                   command=self._reset_zoom).pack(side=RIGHT, padx=2)
        ttk.Button(ctrl_bar, text="Save Image", bootstyle="success",
                   command=self._save_image).pack(side=RIGHT, padx=2)

        # Bind events
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind("<Button-4>", self._on_mousewheel)
        self._canvas.bind("<Button-5>", self._on_mousewheel)
        self._canvas.bind("<ButtonPress-3>", self._start_pan)
        self._canvas.bind("<B3-Motion>", self._do_pan)
        self._canvas.bind("<ButtonPress-2>", self._start_pan)
        self._canvas.bind("<B2-Motion>", self._do_pan)
        self._canvas.bind("<Configure>", self._update_image_position)

        self._render_image()

    def _bgr_to_rgb(self, img):
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _render_image(self):
        scale = self._state["scale"]
        new_w = max(1, int(self._img_w * scale))
        new_h = max(1, int(self._img_h * scale))
        self._state["new_w"] = new_w
        self._state["new_h"] = new_h

        img_resized = cv2.resize(self._cv_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_rgb = self._bgr_to_rgb(img_resized)
        self._state["tk_img"] = ImageTk.PhotoImage(Image.fromarray(img_rgb))

        if self._state["img_id"] is None:
            self._state["img_id"] = self._canvas.create_image(0, 0, anchor=tk.NW, image=self._state["tk_img"])
        else:
            self._canvas.itemconfig(self._state["img_id"], image=self._state["tk_img"])
        self._update_image_position()

    def set_image(self, cv_image):
        """Replace the displayed image with a new OpenCV image (BGR or grayscale)."""
        self._cv_image = cv_image
        if len(cv_image.shape) == 2:
            self._img_h, self._img_w = cv_image.shape
        else:
            self._img_h, self._img_w = cv_image.shape[:2]
        self._reset_zoom()
        self._render_image()

    def _update_image_position(self, event=None):
        if self._updating:
            return
        if self._state["img_id"] is None:
            return
        self._updating = True
        try:
            cw = max(self._canvas.winfo_width(), 1)
            ch = max(self._canvas.winfo_height(), 1)
            nw, nh = self._state["new_w"], self._state["new_h"]
            ix = max((cw - nw) // 2, 0)
            iy = max((ch - nh) // 2, 0)
            self._canvas.coords(self._state["img_id"], ix, iy)
            self._canvas.config(scrollregion=(0, 0, max(nw + ix, cw), max(nh + iy, ch)))
            self._info_label.config(text=f"Zoom: {self._state['scale'] * 100:.0f}% | Wheel=zoom  Right-drag=pan")
        finally:
            self._updating = False

    def _on_mousewheel(self, event):
        old = self._state["scale"]
        if getattr(event, 'num', 0) == 4 or getattr(event, 'delta', 0) > 0:
            self._state["scale"] *= 1.15
        elif getattr(event, 'num', 0) == 5 or getattr(event, 'delta', 0) < 0:
            self._state["scale"] /= 1.15
        self._state["scale"] = max(0.05, min(self._state["scale"], 20.0))
        if abs(self._state["scale"] - old) > 1e-6:
            self._render_image()

    def _start_pan(self, event):
        self._canvas.scan_mark(event.x, event.y)

    def _do_pan(self, event):
        self._canvas.scan_dragto(event.x, event.y, gain=1)

    def _reset_zoom(self):
        self._state["scale"] = min(self._max_display / self._img_w,
                                   self._max_display / self._img_h, 1.0)
        self._render_image()

    def _save_image(self):
        file_path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Image",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("TIFF Image", "*.tiff"),
                ("Bitmap Image", "*.bmp"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return
        # Save at original resolution
        img_to_save = self._bgr_to_rgb(self._cv_image)
        Image.fromarray(img_to_save).save(file_path)


def show_rose_plot(parent, angles, weights, unit="px"):
    rose_window = tk.Toplevel(parent)
    rose_window.title("Crack Orientation Rose Diagram")
    rose_window.geometry("600x600")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    n_bins = 36
    bin_width = 2 * np.pi / n_bins
    angles_rad = np.radians(angles)
    angles_rad_symmetric = np.concatenate([angles_rad, angles_rad + np.pi])
    weights_symmetric = np.concatenate([weights, weights])
    counts, bin_edges = np.histogram(angles_rad_symmetric, bins=n_bins, range=(0, 2*np.pi), weights=weights_symmetric)
    ax.bar(bin_edges[:-1], counts, width=bin_width, bottom=0.0, color='crimson', alpha=0.7, edgecolor='black')
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_title(f"Crack Orientation Rose Diagram (Length Weighted: {unit})", y=1.08)
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=rose_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, rose_window)
    toolbar.update()
    toolbar.pack(fill=X)

    def on_close():
        plt.close(fig)
        rose_window.destroy()

    rose_window.protocol("WM_DELETE_WINDOW", on_close)


def show_fractal_plot(parent, sizes, counts, coeffs, D, r2):
    fractal_window = tk.Toplevel(parent)
    fractal_window.title("Fractal Dimension Analysis")
    fractal_window.geometry("600x500")

    fig, ax = plt.subplots(figsize=(8, 6))
    log_sizes = np.log(sizes)
    log_counts = np.log(counts)
    ax.scatter(log_sizes, log_counts, c='red', label='Raw Data')
    ax.plot(log_sizes, np.polyval(coeffs, log_sizes), 'b--', label=f'Fit: D={D:.3f}')
    ax.set_title(f"Fractal Dimension (Box-Counting)\nD = {D:.4f}, R² = {r2:.4f}")
    ax.set_xlabel("log(Box Size)")
    ax.set_ylabel("log(Box Count)")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=fractal_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, fractal_window)
    toolbar.update()
    toolbar.pack(fill=X)

    def on_close():
        plt.close(fig)
        fractal_window.destroy()

    fractal_window.protocol("WM_DELETE_WINDOW", on_close)


def show_histograms(parent, ids, lengths, widths, unit="px"):
    if not ids:
        return

    hist_window = tk.Toplevel(parent)
    hist_window.title("Crack Statistics Analysis")
    hist_window.geometry("900x450")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ids_str = [str(i) for i in ids]
    ax1.bar(ids_str, lengths, color='#3498db', edgecolor='black', alpha=0.8)
    ax1.set_title(f'Crack Length by Segment ID ({unit})')
    ax1.set_xlabel('Crack ID')
    ax1.set_ylabel(f'Length ({unit})')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(ids_str, widths, color='#e74c3c', edgecolor='black', alpha=0.8)
    ax2.set_title(f'Crack Width by Segment ID ({unit})')
    ax2.set_xlabel('Crack ID')
    ax2.set_ylabel(f'Avg Width ({unit})')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=hist_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, hist_window)
    toolbar.update()
    toolbar.pack(fill=X)

    def on_close():
        plt.close(fig)
        hist_window.destroy()

    hist_window.protocol("WM_DELETE_WINDOW", on_close)


def show_clod_analysis(parent, clods, summary, vis_image, unit="px"):
    """Open a window with soil clod analysis results.

    Args:
        parent: Tk parent widget.
        clods: list of per-clod dicts from analyze_soil_clods().
        summary: summary dict from analyze_soil_clods().
        vis_image: BGR color-coded clod map.
        unit: Physical unit string (e.g. "px", "mm").
    """
    clod_window = tk.Toplevel(parent)
    clod_window.title("Soil Clod Analysis")
    clod_window.state('zoomed')

    # Main layout: left = image, right = stats + histograms
    paned = tk.PanedWindow(clod_window, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
    paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Left panel — color-coded clod map
    left_frame = ttk.Frame(paned)
    paned.add(left_frame, minsize=400, width=600)

    left_label = ttk.Label(left_frame, text="Clod Map (each color = one soil clod, black = crack, gray = excluded)",
                           font=("Microsoft YaHei", 9))
    left_label.pack(anchor=tk.W, pady=(0, 2))

    vis_image_bgr = vis_image  # already BGR from analyze_soil_clods
    zoom_frame = ZoomableImageFrame(left_frame, vis_image_bgr, max_display_size=600)
    zoom_frame.pack(fill=tk.BOTH, expand=True)

    # Right panel — stats + histograms
    right_frame = ttk.Frame(paned)
    paned.add(right_frame, minsize=400, width=700)

    # Summary section
    summary_frame = ttk.Labelframe(right_frame, text=" Summary Statistics ", padding=8,
                                   bootstyle="info")
    summary_frame.pack(fill=tk.X, pady=(0, 8))

    area_unit = f"{unit}²"
    length_unit = unit

    stats_text = (
        f"Clod Count: {summary['clod_count']}\n"
        f"Clod Area Ratio: {summary['clod_area_ratio']}%\n"
        f"Clods per {length_unit}²: {summary['clods_per_unit_area']:.4f}\n"
        f"Total ROI Area: {summary['total_roi_area_phy']} {area_unit}\n"
        f"Mean Clod Area: {summary['mean_area']} {area_unit}\n"
        f"Median Clod Area: {summary['median_area']} {area_unit}\n"
        f"Min / Max Area: {summary['min_area']} / {summary['max_area']} {area_unit}\n"
        f"Std Dev Area: {summary['std_area']} {area_unit}\n"
        f"Mean Shape Factor: {summary['mean_shape_factor']} (1.0 = perfect circle)"
    )
    ttk.Label(summary_frame, text=stats_text, font=("Consolas", 10),
              justify=tk.LEFT).pack(anchor=tk.W)

    # Histograms section
    hist_frame = ttk.Labelframe(right_frame, text=" Distributions ", padding=8,
                                bootstyle="secondary")
    hist_frame.pack(fill=tk.BOTH, expand=True)

    if clods:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        areas = [c['area_phy'] for c in clods]
        shape_factors = [c['shape_factor'] for c in clods]

        # Area histogram
        n_bins = min(20, len(areas))
        ax1.hist(areas, bins=n_bins, color='#3498db', edgecolor='black', alpha=0.8)
        ax1.set_title(f'Clod Area Distribution ({area_unit})')
        ax1.set_xlabel(f'Area ({area_unit})')
        ax1.set_ylabel('Count')
        ax1.grid(axis='y', alpha=0.3)
        ax1.axvline(summary['mean_area'], color='red', linestyle='--',
                    label=f"Mean: {summary['mean_area']:.1f}")
        ax1.axvline(summary['median_area'], color='green', linestyle='-.',
                    label=f"Median: {summary['median_area']:.1f}")
        ax1.legend()

        # Shape factor histogram
        ax2.hist(shape_factors, bins=n_bins, color='#e74c3c', edgecolor='black', alpha=0.8)
        ax2.set_title('Shape Factor Distribution')
        ax2.set_xlabel('Shape Factor (1.0 = circle)')
        ax2.set_ylabel('Count')
        ax2.grid(axis='y', alpha=0.3)
        ax2.axvline(summary['mean_shape_factor'], color='blue', linestyle='--',
                    label=f"Mean: {summary['mean_shape_factor']:.3f}")
        ax2.legend()

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=hist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        hist_toolbar = NavigationToolbar2Tk(canvas, hist_frame)
        hist_toolbar.update()
        hist_toolbar.pack(fill=X)

        def on_close():
            plt.close(fig)
            clod_window.destroy()

        clod_window.protocol("WM_DELETE_WINDOW", on_close)
    else:
        ttk.Label(hist_frame, text="No valid soil clods found.\n"
                  "Check: are cracks forming closed polygons?\n"
                  "Try reducing the min clod area threshold.",
                  font=("Microsoft YaHei", 10), bootstyle="warning").pack(expand=True)


def open_enlarged_result_window(parent, cv_img, title):
    if cv_img is None:
        return
    top = tk.Toplevel(parent)
    top.title(f"{title} - Wheel Zoom / Right-drag Pan")
    sw = top.winfo_screenwidth()
    sh = top.winfo_screenheight()
    win_w, win_h = min(1200, sw - 100), min(800, sh - 100)
    top.geometry(f"{win_w}x{win_h}+{(sw - win_w) // 2}+{(sh - win_h) // 2}")

    zoom_frame = ZoomableImageFrame(top, cv_img, max_display_size=win_w)
    zoom_frame.pack(fill=tk.BOTH, expand=True)


def show_connectivity_analysis(parent, metrics, skel_bool, vis_info):
    """Open crack network connectivity analysis window."""
    conn_window = tk.Toplevel(parent)
    conn_window.title("Crack Network Connectivity Analysis")
    conn_window.geometry("1100x700")

    # Main layout: left = graph visualization, right = metrics
    paned = tk.PanedWindow(conn_window, orient=tk.HORIZONTAL,
                           sashrelief=tk.RAISED, sashwidth=4)
    paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Left — skeleton graph visualization
    left_frame = ttk.Frame(paned)
    paned.add(left_frame, minsize=400, width=550)

    ttk.Label(left_frame, text="Crack Network Graph (red=junctions, blue=endpoints, "
              "colored lines=segments by component)",
              font=("Microsoft YaHei", 9)).pack(anchor=tk.W, pady=(0, 4))

    h, w = skel_bool.shape[:2]
    vis_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    vis_rgb[:] = [40, 40, 40]

    # Draw segments colored by component
    labeled_comp = vis_info['labeled_components']
    valid_comp_ids = vis_info['valid_component_ids']
    np.random.seed(7)
    comp_colors = np.random.randint(80, 255, size=(max(valid_comp_ids) + 2, 3),
                                     dtype=np.uint8)
    for cid in valid_comp_ids:
        mask = labeled_comp == cid
        vis_rgb[mask] = comp_colors[cid]

    # Draw junctions in red
    vis_rgb[vis_info['junction_mask']] = [255, 50, 50]

    # Draw endpoints in blue
    vis_rgb[vis_info['endpoint_mask']] = [50, 120, 255]

    # Draw the skeleton pixels themselves (overlay on colors)
    # Junctions and endpoints already drawn; remaining skeleton in white
    plain_skel = skel_bool & (~vis_info['junction_mask']) & (~vis_info['endpoint_mask'])
    vis_rgb[plain_skel] = [220, 220, 220]

    # ZoomableImageFrame expects BGR
    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
    zoom_frame = ZoomableImageFrame(left_frame, vis_bgr)
    zoom_frame.pack(fill=tk.BOTH, expand=True)

    # Right — metrics panel
    right_frame = ttk.Frame(paned)
    paned.add(right_frame, minsize=350, width=500)

    # Key metrics
    metrics_frame = ttk.Labelframe(right_frame, text=" Connectivity Metrics ",
                                    padding=10, bootstyle="info")
    metrics_frame.pack(fill=tk.X, pady=(0, 8))

    J, E, S, C = (metrics['junction_count'], metrics['endpoint_count'],
                  metrics['segment_count'], metrics['component_count'])

    metrics_text = (
        f"Junctions (J):          {J}\n"
        f"Endpoints (E):          {E}\n"
        f"Crack Segments (S):     {S}\n"
        f"Connected Components (C): {C}\n"
        f"Average Node Degree:    {metrics['avg_node_degree']}\n"
        f"Network Density (D):    {metrics['network_density']}\n"
        f"Euler Number (χ):       {metrics['euler_number']}\n"
        f"Connectivity Index (CI): {metrics['connectivity_index']}"
    )
    ttk.Label(metrics_frame, text=metrics_text, font=("Consolas", 11),
              justify=tk.LEFT).pack(anchor=tk.W)

    # Formula reference
    formula_frame = ttk.Labelframe(right_frame, text=" Formulas ", padding=8,
                                    bootstyle="secondary")
    formula_frame.pack(fill=tk.X, pady=(0, 8))
    formula_text = (
        "x = C - S + J\n"
        "D = 2S / J(J-1)    (J >= 2)\n"
        "CI = (J - E + S) / C"
    )
    ttk.Label(formula_frame, text=formula_text, font=("Consolas", 9),
              justify=tk.LEFT, bootstyle="secondary").pack(anchor=tk.W)

    # Interpretation
    interp_frame = ttk.Labelframe(right_frame, text=" Interpretation ",
                                   padding=8, bootstyle="success")
    interp_frame.pack(fill=tk.X)

    ci = metrics['connectivity_index']
    if ci <= 0:
        level = "Disconnected / fragmentary — cracks do not form a coherent network.\nSuitable for: early-stage cracking, sparse crack studies."
    elif ci <= 5:
        level = "Moderately connected — partial crack network with some dead ends.\nSuitable for: most desiccation cracking studies."
    elif ci <= 20:
        level = "Well-connected — mature crack network with many intersections.\nSuitable for: permeability analysis, fluid flow modeling."
    else:
        level = "Highly connected — dense crack network approaching full connectivity.\nSuitable for: fragmentation analysis, soil structure assessment."

    if metrics['euler_number'] < 0:
        level += "\n\nNegative Euler number → network has many closed loops (typical for mature desiccation cracks)."
    elif metrics['euler_number'] > 0:
        level += "\n\nPositive Euler number → tree-like structure with few closed loops (typical for early-stage or directional cracks)."

    ttk.Label(interp_frame, text=level, font=("Microsoft YaHei", 9),
              wraplength=400, justify=tk.LEFT).pack(anchor=tk.W)

    def on_close():
        conn_window.destroy()

    conn_window.protocol("WM_DELETE_WINDOW", on_close)


def show_junction_classification(parent, result, skel_bool):
    """Open junction type classification window."""
    junc_window = tk.Toplevel(parent)
    junc_window.title("Junction Type Classification")
    junc_window.geometry("1100x700")

    paned = tk.PanedWindow(junc_window, orient=tk.HORIZONTAL,
                           sashrelief=tk.RAISED, sashwidth=4)
    paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Left — color-coded junction map
    left_frame = ttk.Frame(paned)
    paned.add(left_frame, minsize=400, width=550)

    ttk.Label(left_frame, text="Junction Types: orange=T, cyan=Y, magenta=X, yellow=Multi, blue=endpoints",
              font=("Microsoft YaHei", 9)).pack(anchor=tk.W, pady=(0, 4))

    h, w = skel_bool.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:] = [40, 40, 40]

    vis_info = result['vis_info']

    # Skeleton in light gray (excluding junctions and endpoints)
    skel_plain = skel_bool.copy()
    skel_plain[vis_info['T_mask']] = False
    skel_plain[vis_info['Y_mask']] = False
    skel_plain[vis_info['X_mask']] = False
    skel_plain[vis_info['Multi_mask']] = False
    skel_plain[vis_info['endpoint_mask']] = False
    vis[skel_plain] = [200, 200, 200]

    # T-type: orange #FF8C00
    vis[vis_info['T_mask']] = [0, 140, 255]
    # Y-type: cyan #00CED1
    vis[vis_info['Y_mask']] = [209, 206, 0]
    # X-type: magenta #FF00FF
    vis[vis_info['X_mask']] = [255, 0, 255]
    # Multi-type: yellow #FFFF00
    vis[vis_info['Multi_mask']] = [0, 255, 255]
    # Endpoints: blue
    vis[vis_info['endpoint_mask']] = [255, 150, 50]

    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    zoom_frame = ZoomableImageFrame(left_frame, vis_bgr)
    zoom_frame.pack(fill=tk.BOTH, expand=True)

    # Right panel
    right_frame = ttk.Frame(paned)
    paned.add(right_frame, minsize=350, width=500)

    # Classification table
    table_frame = ttk.Labelframe(right_frame, text=" Junction Classification ",
                                  padding=10, bootstyle="info")
    table_frame.pack(fill=tk.X, pady=(0, 8))

    T, Y, X, M = (result['T_count'], result['Y_count'],
                  result['X_count'], result['Multi_count'])
    total = result['total']

    table_text = (
        f"Total junctions detected: {total}\n\n"
        f"{'Type':<12} {'Count':<8} {'Pct':<8} {'Color':<12} {'Meaning'}\n"
        f"{'─'*65}\n"
        f"{'T-type':<12} {T:<8} {result['T_pct']}%{'':<4} {'Orange':<12} secondary meets primary\n"
        f"{'Y-type':<12} {Y:<8} {result['Y_pct']}%{'':<4} {'Cyan':<12} synchronous 3-way cracking\n"
        f"{'X-type':<12} {X:<8} {result['X_pct']}%{'':<4} {'Magenta':<12} two cracks crossing\n"
        f"{'Multi':<12} {M:<8} {result['Multi_pct']}%{'':<4} {'Yellow':<12} complex multi-branch\n"
    )
    ttk.Label(table_frame, text=table_text, font=("Consolas", 10),
              justify=tk.LEFT).pack(anchor=tk.W)

    # Interpretation
    interp_frame = ttk.Labelframe(right_frame, text=" Geological Interpretation ",
                                   padding=8, bootstyle="success")
    interp_frame.pack(fill=tk.X, pady=(0, 8))

    if total == 0:
        interp = "No junctions found — crack network has no intersections (all isolated segments or single cracks)."
    else:
        interp = ""
        # Dominant type analysis
        if T > Y and T > X and T > M:
            interp += "Dominated by T-type junctions → hierarchical cracking pattern:\n"
            interp += "primary cracks form first, secondary cracks join perpendicularly.\n"
            interp += "Typical of directional drying or stress-controlled cracking.\n\n"
        elif Y > T and Y > X and Y > M:
            interp += "Dominated by Y-type junctions → isotropic cracking pattern:\n"
            interp += "cracks nucleate simultaneously in multiple directions.\n"
            interp += "Typical of uniform desiccation with homogeneous soil.\n\n"
        elif X > T and X > Y and X > M:
            interp += "Dominated by X-type junctions → crossing crack pattern:\n"
            interp += "two preferential crack directions intersect.\n"
            interp += "May indicate sequential cracking episodes or multiple stress directions.\n\n"

        if T > 0 and Y > 0:
            ratio = T / max(Y, 1)
            if ratio > 2:
                interp += f"T/Y ratio = {ratio:.1f} → strongly hierarchical network.\n"
            elif ratio < 0.5:
                interp += f"T/Y ratio = {ratio:.1f} → predominantly synchronous cracking.\n"

    ttk.Label(interp_frame, text=interp, font=("Microsoft YaHei", 9),
              wraplength=400, justify=tk.LEFT).pack(anchor=tk.W)

    # Pie chart
    if total > 0:
        chart_frame = ttk.Frame(right_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True)

        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ['T-type', 'Y-type', 'X-type', 'Multi']
        sizes = [T, Y, X, M]
        colors_pie = ['#FF8C00', '#00CED1', '#FF00FF', '#FFD700']
        explode = [0.03] * 4
        filtered_labels = []
        filtered_sizes = []
        filtered_colors = []

        for lbl, sz, clr in zip(labels, sizes, colors_pie):
            if sz > 0:
                filtered_labels.append(f'{lbl} ({sz})')
                filtered_sizes.append(sz)
                filtered_colors.append(clr)

        if filtered_sizes:
            ax.pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors,
                   explode=explode[:len(filtered_sizes)], autopct='%1.1f%%',
                   shadow=False, startangle=90)
            ax.set_title('Junction Type Distribution')

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        pie_toolbar = NavigationToolbar2Tk(canvas, chart_frame)
        pie_toolbar.update()
        pie_toolbar.pack(fill=X)

        def on_close():
            plt.close(fig)
            junc_window.destroy()

        junc_window.protocol("WM_DELETE_WINDOW", on_close)