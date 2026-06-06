import math
import numpy as np
import cv2
from collections import deque
from skimage.measure import label as sk_label, regionprops
from skimage.segmentation import watershed
from PyCrackQ.config import MIN_CRACK_AREA, MIN_CLOD_AREA
from PyCrackQ.image_processing import detect_junctions


def trace_segment_euclidean_length(skel_bool, coords):
    if len(coords) <= 1:
        return 1.0 if len(coords) == 1 else 0.0

    coord_set = set(map(tuple, coords))

    neighbor_kernel = np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]], dtype=np.uint8)
    min_r, min_c = np.min(coords, axis=0)
    max_r, max_c = np.max(coords, axis=0)
    local_h = max_r - min_r + 3
    local_w = max_c - min_c + 3
    local_skel = np.zeros((local_h, local_w), dtype=np.uint8)
    for r, c in coords:
        local_skel[r - min_r + 1, c - min_c + 1] = 1
    neighbor_count = cv2.filter2D(local_skel, -1, neighbor_kernel)
    endpoints_local = []
    for r, c in coords:
        lr, lc = r - min_r + 1, c - min_c + 1
        if neighbor_count[lr, lc] == 1:
            endpoints_local.append((r, c))

    if endpoints_local:
        start = endpoints_local[0]
    else:
        centroid_r = np.mean(coords[:, 0])
        centroid_c = np.mean(coords[:, 1])
        dists = np.sqrt((coords[:, 0] - centroid_r)**2 + (coords[:, 1] - centroid_c)**2)
        start = tuple(coords[np.argmin(dists)])

    visited = set()
    visited.add(start)
    total_length = 0.0
    stack = [start]
    sqrt2 = math.sqrt(2)

    while stack:
        cr, cc = stack.pop()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in coord_set and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    step_len = sqrt2 if (dr != 0 and dc != 0) else 1.0
                    total_length += step_len
                    stack.append((nr, nc))

    if not endpoints_local and total_length > 0:
        if len(coords) > 2:
            avg_step = total_length / (len(coords) - 1)
            total_length += avg_step

    return max(total_length, 1.0) if total_length > 0 else 1.0


def compute_precise_skeleton_length(skel_bool):
    labeled = sk_label(skel_bool, connectivity=2, return_num=False)
    props = regionprops(labeled)

    total_length = 0.0
    for prop in props:
        coords = prop.coords
        seg_len = trace_segment_euclidean_length(skel_bool, coords)
        total_length += seg_len

    return total_length


def _get_boundary_intersection_mask(skel_bool, circular_mask, is_circular_mode):
    if not is_circular_mode or circular_mask is None:
        return np.zeros_like(skel_bool, dtype=bool)

    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = cv2.dilate(circular_mask, kernel, iterations=1) - cv2.erode(circular_mask, kernel, iterations=1)

    boundary_dist = cv2.distanceTransform(255 - boundary, cv2.DIST_L2, 5)
    near_boundary = boundary_dist <= 3

    return np.logical_and(skel_bool, near_boundary)


def calculate_accurate_metrics(binary_mask, skel_bool, dist_map=None, scale_factor=1.0, is_circular_mode=False, circular_mask=None, apply_roi=True):
    scale = scale_factor if scale_factor and scale_factor > 0 else 1.0
    if binary_mask.dtype == bool:
        binary_mask = (binary_mask.astype(np.uint8) * 255)
    if skel_bool.dtype != bool:
        skel_bool = skel_bool > 0

    if apply_roi and is_circular_mode and circular_mask is not None:
        mask_bool = circular_mask > 0
        skel_bool = np.logical_and(skel_bool, mask_bool)
        binary_mask = cv2.bitwise_and(binary_mask, binary_mask, mask=circular_mask)

    if apply_roi and is_circular_mode and circular_mask is not None:
        boundary_mask = _get_boundary_intersection_mask(skel_bool, circular_mask, is_circular_mode)
    else:
        boundary_mask = np.zeros_like(skel_bool, dtype=bool)

    length_px = compute_precise_skeleton_length(skel_bool)

    if length_px == 0 and np.sum(skel_bool) > 0:
        length_px = 1.0

    area_px = np.sum(binary_mask > 0)

    if dist_map is not None and np.sum(skel_bool) > 0:
        if apply_roi and is_circular_mode and circular_mask is not None:
            valid_skel = skel_bool & (~boundary_mask) & (circular_mask > 0)
            if np.any(valid_skel):
                local_widths = dist_map[valid_skel] * 2
                avg_width_px = np.mean(local_widths)
            else:
                all_valid = skel_bool & (circular_mask > 0)
                if np.any(all_valid):
                    local_widths = dist_map[all_valid] * 2
                    avg_width_px = np.mean(local_widths)
                else:
                    avg_width_px = area_px / length_px if length_px > 0 else 0
        else:
            local_widths = dist_map[skel_bool] * 2
            avg_width_px = np.mean(local_widths)
    else:
        avg_width_px = area_px / length_px if length_px > 0 else 0

    max_width_px = 0
    if dist_map is not None and np.sum(skel_bool) > 0:
        if apply_roi and is_circular_mode and circular_mask is not None:
            interior_crack = (binary_mask > 0) & (~boundary_mask)
            if np.any(interior_crack):
                max_width_px = np.max(dist_map[interior_crack]) * 2
            elif np.any(binary_mask > 0):
                max_width_px = np.max(dist_map[binary_mask > 0]) * 2
            else:
                max_width_px = 0
        else:
            crack_mask = binary_mask > 0
            max_width_px = np.max(dist_map[crack_mask]) * 2 if np.any(crack_mask) else 0

    return {
        'length_px': length_px,
        'length_phy': length_px / scale,
        'avg_width_px': avg_width_px,
        'avg_width_phy': avg_width_px / scale,
        'max_width_px': max_width_px,
        'max_width_phy': max_width_px / scale,
        'area_px': area_px,
        'area_phy': area_px / (scale ** 2)
    }


def calculate_segment_metrics(binary_mask, skel_bool, dist_map=None, scale_factor=1.0, is_circular_mode=False, circular_mask=None, apply_roi=True):
    scale = scale_factor if scale_factor and scale_factor > 0 else 1.0
    if binary_mask.dtype == bool:
        binary_mask = (binary_mask.astype(np.uint8) * 255)
    if skel_bool.dtype != bool:
        skel_bool = skel_bool > 0

    can_apply_roi = (
        apply_roi
        and is_circular_mode
        and circular_mask is not None
        and skel_bool.shape == circular_mask.shape
        and binary_mask.shape[:2] == circular_mask.shape
    )
    if can_apply_roi:
        mask_bool = circular_mask > 0
        skel_bool = np.logical_and(skel_bool, mask_bool)
        binary_mask = cv2.bitwise_and(binary_mask, binary_mask, mask=circular_mask)

    if can_apply_roi:
        boundary_mask = _get_boundary_intersection_mask(skel_bool, circular_mask, is_circular_mode)
    else:
        boundary_mask = np.zeros_like(skel_bool, dtype=bool)

    coords = np.argwhere(skel_bool)
    if len(coords) == 0:
        length_px = 0
    else:
        length_px = trace_segment_euclidean_length(skel_bool, coords)

    if length_px == 0 and np.sum(skel_bool) > 0:
        length_px = 1.0

    area_px = np.sum(binary_mask > 0)

    if dist_map is not None and np.sum(skel_bool) > 0:
        if can_apply_roi:
            valid_skel = skel_bool & (~boundary_mask) & (circular_mask > 0)
            if np.any(valid_skel):
                local_widths = dist_map[valid_skel] * 2
                avg_width_px = np.mean(local_widths)
            else:
                all_valid = skel_bool & (circular_mask > 0)
                if np.any(all_valid):
                    local_widths = dist_map[all_valid] * 2
                    avg_width_px = np.mean(local_widths)
                else:
                    avg_width_px = area_px / length_px if length_px > 0 else 0
        else:
            local_widths = dist_map[skel_bool] * 2
            avg_width_px = np.mean(local_widths)
    else:
        avg_width_px = area_px / length_px if length_px > 0 else 0

    max_width_px = 0
    if dist_map is not None and np.sum(skel_bool) > 0:
        if can_apply_roi:
            interior_crack = (binary_mask > 0) & (~boundary_mask)
            if np.any(interior_crack):
                max_width_px = np.max(dist_map[interior_crack]) * 2
            elif np.any(binary_mask > 0):
                max_width_px = np.max(dist_map[binary_mask > 0]) * 2
            else:
                max_width_px = 0
        else:
            crack_mask = binary_mask > 0
            max_width_px = np.max(dist_map[crack_mask]) * 2 if np.any(crack_mask) else 0

    return {
        'length_px': length_px,
        'length_phy': length_px / scale,
        'avg_width_px': avg_width_px,
        'avg_width_phy': avg_width_px / scale,
        'max_width_px': max_width_px,
        'max_width_phy': max_width_px / scale,
        'area_px': area_px,
        'area_phy': area_px / (scale ** 2)
    }


def get_fractal_dim(Z, progress_callback=None):
    h, w = Z.shape
    p = min(h, w)

    if p < 4:
        return [], []

    sizes = []
    counts = []

    box_sizes = [2, 3, 4, 6, 8, 12, 16, 32, 64]
    box_sizes = [s for s in box_sizes if s <= min(h, w)]

    total_steps = len(box_sizes)
    for step, box_size in enumerate(box_sizes, start=1):
        if box_size < 2 or box_size > p:
            continue

        count = 0
        for r_start in range(0, h, box_size):
            for c_start in range(0, w, box_size):
                r_end = min(r_start + box_size, h)
                c_end = min(c_start + box_size, w)
                block = Z[r_start:r_end, c_start:c_end]
                if np.any(block):
                    count += 1

        if count > 0:
            sizes.append(box_size)
            counts.append(count)
        if progress_callback is not None:
            progress_callback(step, total_steps, box_size, count)

    if len(sizes) < 3:
        return [], []

    return sizes, counts


def calculate_branch_angles(cached_angle_skel, r, c, radius=20):
    h, w = cached_angle_skel.shape
    q = deque([(r, c, 0.0)])
    visited = set([(r, c)])
    branch_traces = {}

    sqrt2 = 1.41421356

    initial_neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0: continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if cached_angle_skel[nr, nc]:
                    initial_neighbors.append((nr, nc))

    for start_nr, start_nc in initial_neighbors:
        trace = [(start_nr, start_nc)]
        visited_branch = set([(r, c), (start_nr, start_nc)])
        branch_q = deque([(start_nr, start_nc,
                           sqrt2 if (start_nr - r != 0 and start_nc - c != 0) else 1.0)])

        while branch_q:
            curr_r, curr_c, dist = branch_q.popleft()
            if dist >= radius:
                trace.append((curr_r, curr_c))
                continue
            for dr2 in [-1, 0, 1]:
                for dc2 in [-1, 0, 1]:
                    if dr2 == 0 and dc2 == 0: continue
                    nnr, nnc = curr_r + dr2, curr_c + dc2
                    if 0 <= nnr < h and 0 <= nnc < w:
                        if cached_angle_skel[nnr, nnc] and (nnr, nnc) not in visited_branch:
                            visited_branch.add((nnr, nnc))
                            step = sqrt2 if (dr2 != 0 and dc2 != 0) else 1.0
                            branch_q.append((nnr, nnc, dist + step))
                            trace.append((nnr, nnc))

        branch_traces[(start_nr, start_nc)] = trace
        visited.update(visited_branch)

    angles = []
    branch_lengths = []

    for start_pt, trace in branch_traces.items():
        if len(trace) < 3:
            dy = trace[0][0] - r
            dx = trace[0][1] - c
            deg = math.degrees(math.atan2(dy, dx))
            if deg < 0: deg += 360
            angles.append(deg)
            branch_lengths.append(math.hypot(dy, dx))
            continue

        pts = np.array(trace, dtype=np.float64)
        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid

        cov_matrix = np.cov(pts_centered[:, 1], pts_centered[:, 0])  # (col, row) -> (x, y)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        idx = np.argmax(eigenvalues)
        principal_dir = eigenvectors[:, idx]

        dy_from_junc = centroid[0] - r
        dx_from_junc = centroid[1] - c

        if principal_dir[0] * dx_from_junc + principal_dir[1] * dy_from_junc < 0:
            principal_dir = -principal_dir

        deg = math.degrees(math.atan2(principal_dir[1], principal_dir[0]))
        if deg < 0: deg += 360

        branch_len = 0
        prev = (r, c)
        for pt in trace:
            branch_len += math.hypot(pt[0] - prev[0], pt[1] - prev[1])
            prev = pt

        angles.append(deg)
        branch_lengths.append(branch_len)

    merged_angles = []
    angle_branch_pairs = sorted(zip(angles, branch_lengths), key=lambda x: x[1], reverse=True)

    for deg, blen in angle_branch_pairs:
        is_new = True
        for i, (existing_deg, _) in enumerate(merged_angles):
            diff = abs(existing_deg - deg)
            if diff > 180: diff = 360 - diff
            merge_thresh = max(15, min(30, 50 / max(blen, 1)))
            if diff < merge_thresh:
                is_new = False
                break
        if is_new:
            merged_angles.append((deg, blen))

    return sorted([a for a, _ in merged_angles])


def analyze_soil_clods(binary_mask, scale_factor=1.0, is_circular_mode=False,
                       circular_mask=None, min_clod_area=None):
    """Identify and measure soil clods (polygons enclosed by crack networks).

    Inverts the binary crack mask to find soil regions, excludes border-touching
    regions (background), and computes geometric metrics for each clod.

    Args:
        binary_mask: 2D uint8 or bool array where 255/True = crack, 0/False = soil.
        scale_factor: Pixels per physical unit (default 1.0 = px).
        is_circular_mode: Whether a circular ROI is active.
        circular_mask: 2D uint8 mask (same shape as binary_mask), 255 inside ROI.
        min_clod_area: Minimum clod area in px2. Defaults to MIN_CLOD_AREA from config.

    Returns:
        dict with keys:
            'clods': list of per-clod dicts (id, area_px, area_phy, perimeter_px,
                     perimeter_phy, equivalent_diameter, shape_factor)
            'summary': dict with clod_count, clod_area_ratio, clods_per_unit_area,
                       mean_area, mean_shape_factor, unit
            'vis_image': BGR color-coded clod map (numpy array)
    """
    if min_clod_area is None:
        min_clod_area = MIN_CLOD_AREA
    scale = scale_factor if scale_factor and scale_factor > 0 else 1.0
    unit = "px" if scale == 1.0 else "mm"

    if binary_mask.dtype == bool:
        binary_mask = (binary_mask.astype(np.uint8) * 255)

    h, w = binary_mask.shape[:2]

    # Close small gaps in crack lines before inversion to ensure
    # proper soil clod separation. A 1px crack line surrounded by soil
    # gets erased by median-filter denoising; this dilation bridges
    # those gaps so the crack network remains a proper barrier.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel)

    # Invert: soil clods become white (255), cracks become black (0)
    clod_mask = cv2.bitwise_not(binary_closed)

    # Apply circular ROI if active
    if is_circular_mode and circular_mask is not None:
        clod_mask = cv2.bitwise_and(clod_mask, clod_mask, mask=circular_mask)

    # Label all clods. The image border acts as an implicit crack boundary:
    # soil regions enclosed by cracks AND the image edge are valid clods.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        clod_mask, connectivity=4)

    # Compute total ROI area
    if is_circular_mode and circular_mask is not None:
        total_roi_area_px = cv2.countNonZero(circular_mask)
    else:
        total_roi_area_px = h * w

    # Process each valid clod
    clods = []
    # Map label → clod index for visualization
    label_to_clod_idx = {}
    np.random.seed(42)
    colors_list = np.random.randint(60, 255, size=(n_labels + 1, 3), dtype=np.uint8)
    colors_list[0] = [40, 40, 40]

    for i in range(1, n_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]
        if area_px < min_clod_area:
            continue

        # Find contours for perimeter
        clod_binary = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(clod_binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perimeter_px = cv2.arcLength(contours[0], True)

        clod_idx = len(clods)
        area_phy = area_px / (scale ** 2)
        perimeter_phy = perimeter_px / scale
        equivalent_diameter = math.sqrt(4.0 * area_px / math.pi)
        shape_factor = (4.0 * math.pi * area_px) / (perimeter_px ** 2) if perimeter_px > 0 else 0

        label_to_clod_idx[i] = clod_idx

        clods.append({
            'id': clod_idx + 1,
            'area_px': area_px,
            'area_phy': round(area_phy, 2),
            'perimeter_px': round(perimeter_px, 1),
            'perimeter_phy': round(perimeter_phy, 2),
            'equivalent_diameter': round(equivalent_diameter, 1),
            'shape_factor': round(shape_factor, 4),
        })

    # Build summary
    if clods:
        areas = [c['area_phy'] for c in clods]
        shape_factors = [c['shape_factor'] for c in clods]
        total_clod_area_px = sum(c['area_px'] for c in clods)
        clod_area_ratio = (total_clod_area_px / total_roi_area_px * 100) if total_roi_area_px > 0 else 0
        clods_per_unit = len(clods) / (total_roi_area_px / (scale ** 2)) if total_roi_area_px > 0 else 0
        summary = {
            'clod_count': len(clods),
            'clod_area_ratio': round(clod_area_ratio, 2),
            'clods_per_unit_area': round(clods_per_unit, 6),
            'unit': unit,
            'total_roi_area_phy': round(total_roi_area_px / (scale ** 2), 2),
            'mean_area': round(np.mean(areas), 2),
            'median_area': round(np.median(areas), 2),
            'min_area': round(np.min(areas), 2),
            'max_area': round(np.max(areas), 2),
            'std_area': round(np.std(areas), 2),
            'mean_shape_factor': round(np.mean(shape_factors), 4),
        }
    else:
        summary = {
            'clod_count': 0, 'clod_area_ratio': 0, 'clods_per_unit_area': 0,
            'unit': unit,
            'total_roi_area_phy': round(total_roi_area_px / (scale ** 2), 2),
            'mean_area': 0, 'median_area': 0, 'min_area': 0, 'max_area': 0,
            'std_area': 0, 'mean_shape_factor': 0,
        }

    # Build color-coded visualization (BGR)
    vis_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    vis_bgr[:] = [50, 50, 50]  # default for excluded regions
    vis_bgr[binary_mask > 0] = [20, 20, 20]  # cracks in near-black

    for label_id, c_idx in label_to_clod_idx.items():
        vis_bgr[labels == label_id] = colors_list[c_idx + 1]

    if is_circular_mode and circular_mask is not None:
        vis_bgr[circular_mask == 0] = [60, 60, 60]

    return {
        'clods': clods,
        'summary': summary,
        'vis_image': vis_bgr,
    }


def analyze_crack_connectivity(skel_bool):
    """Compute crack network connectivity metrics from a skeleton.

    Converts the skeleton to a graph, counting junctions, endpoints, segments,
    and connected components. Derives connectivity metrics from the graph.

    Args:
        skel_bool: Boolean 2D array, True = skeleton pixel.

    Returns:
        dict with keys:
            'junction_count' (int): J — pixels with >= 3 branches.
            'endpoint_count' (int): E — pixels with exactly 1 neighbor.
            'segment_count' (int): S — branch count after removing junctions.
            'component_count' (int): C — connected skeleton components.
            'avg_node_degree' (float): Mean branch count at junctions.
            'network_density' (float): D = 2S / (J(J-1)), 0 if J<2.
            'euler_number' (int): x = C - S + J.
            'connectivity_index' (float): CI = (J - E + S) / C, 0 if C=0.
            'vis_info' (dict): data for visualization (junction_mask,
                               endpoint_mask, labeled_segments, labeled_components).
    """
    if skel_bool.dtype != bool:
        skel_bool = skel_bool > 0

    junction_mask, endpoint_mask, junction_count = detect_junctions(skel_bool)

    # Count endpoint pixels directly (those not junction pixels)
    # Endpoint: skeleton pixel with exactly 1 neighbor in 8-neighborhood
    neighbor_kernel = np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]], dtype=np.uint8)
    skel_uint8 = skel_bool.astype(np.uint8)
    neighbor_count = cv2.filter2D(skel_uint8, -1, neighbor_kernel)
    true_endpoints = (skel_bool & (neighbor_count == 1))
    endpoint_count = np.sum(true_endpoints)

    # Remove junctions to split skeleton into isolated segments
    skel_no_junctions = skel_bool.copy()
    skel_no_junctions[junction_mask] = False

    # Segment count = connected components after junction removal
    labeled_segments = sk_label(skel_no_junctions, connectivity=2)
    segment_props = regionprops(labeled_segments)
    valid_segments = [p for p in segment_props if p.area >= 2]
    segment_count = len(valid_segments)

    # Connected components of the original skeleton (intact)
    labeled_components = sk_label(skel_bool, connectivity=2)
    component_props = regionprops(labeled_components)
    valid_components = [p for p in component_props if p.area >= 2]
    component_count = len(valid_components)

    # Average node degree: mean number of branches per junction
    if junction_count > 0:
        # Count branch directions per junction via local 3x3 analysis
        junction_ys, junction_xs = np.where(junction_mask)
        total_branches = 0
        for y, x in zip(junction_ys, junction_xs):
            # Count connected components in the 3x3 neighborhood (excluding center)
            y0, y1 = max(0, y - 1), min(skel_bool.shape[0], y + 2)
            x0, x1 = max(0, x - 1), min(skel_bool.shape[1], x + 2)
            local = skel_bool[y0:y1, x0:x1].copy()
            cy, cx = y - y0, x - x0
            local[cy, cx] = False
            # Count 4-connected components in local neighborhood
            local_labels = sk_label(local, connectivity=1)
            n_branches = local_labels.max()
            total_branches += n_branches
        avg_degree = total_branches / junction_count
    else:
        avg_degree = 0.0

    # Network density: fraction of possible edges present
    if junction_count >= 2:
        network_density = 2.0 * segment_count / (junction_count * (junction_count - 1))
    else:
        network_density = 0.0

    # Euler characteristic: x = V - E + F for planar graph
    # Here: x = C - S + J (connected components - segments + junctions)
    euler_number = component_count - segment_count + junction_count

    # CI = (J - E + S) / C, normalized per connected component
    if component_count > 0:
        connectivity_index = (junction_count - endpoint_count + segment_count) / component_count
    else:
        connectivity_index = 0.0

    return {
        'junction_count': junction_count,
        'endpoint_count': endpoint_count,
        'segment_count': segment_count,
        'component_count': component_count,
        'avg_node_degree': round(avg_degree, 2),
        'network_density': round(network_density, 4),
        'euler_number': euler_number,
        'connectivity_index': round(connectivity_index, 2),
        'vis_info': {
            'junction_mask': junction_mask,
            'endpoint_mask': true_endpoints,
            'labeled_segments': labeled_segments,
            'labeled_components': labeled_components,
            'valid_segment_ids': {p.label for p in valid_segments},
            'valid_component_ids': {p.label for p in valid_components},
        },
    }


def classify_junctions(skel_bool):
    """Classify each junction pixel as T, Y, X, or Multi type.

    T-type (3 branches): one angle ~180°, secondary crack meets primary.
    Y-type (3 branches): all angles < 165°, synchronous cracking.
    X-type (4 branches): two cracks cross.
    Multi-type (5+ branches): complex intersection.

    Args:
        skel_bool: Boolean 2D skeleton array.

    Returns:
        dict with keys:
            'total': total junction count.
            'T_count', 'Y_count', 'X_count', 'Multi_count': per-type counts.
            'T_pct', 'Y_pct', 'X_pct', 'Multi_pct': per-type percentages.
            'junctions': list of {pos, type, branch_count, angles}.
            'vis_info': dict with 'T_mask', 'Y_mask', 'X_mask', 'Multi_mask',
                        'endpoint_mask' for color-coded visualization.
    """
    if skel_bool.dtype != bool:
        skel_bool = skel_bool > 0

    junction_mask, endpoint_mask, _ = detect_junctions(skel_bool)

    j_ys, j_xs = np.where(junction_mask)
    h, w = skel_bool.shape

    # Compute endpoints for visualization
    neighbor_kernel = np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]], dtype=np.uint8)
    skel_uint8 = skel_bool.astype(np.uint8)
    neighbor_count = cv2.filter2D(skel_uint8, -1, neighbor_kernel)
    true_endpoints = (skel_bool & (neighbor_count == 1))

    # Cluster nearby junction pixels into single geometric junctions
    # Junction pixels within CLUSTER_RADIUS belong to the same cluster
    CLUSTER_RADIUS = 8
    j_pixels = [(int(y), int(x)) for y, x in zip(j_ys, j_xs)]
    cluster_labels = -np.ones(len(j_pixels), dtype=int)
    cluster_id = 0

    for i in range(len(j_pixels)):
        if cluster_labels[i] >= 0:
            continue
        # Start a new cluster
        cluster_labels[i] = cluster_id
        cluster_queue = [j_pixels[i]]
        while cluster_queue:
            cy, cx = cluster_queue.pop()
            for j in range(len(j_pixels)):
                if cluster_labels[j] >= 0:
                    continue
                dy = j_pixels[j][0] - cy
                dx = j_pixels[j][1] - cx
                if dy * dy + dx * dx <= CLUSTER_RADIUS * CLUSTER_RADIUS:
                    cluster_labels[j] = cluster_id
                    cluster_queue.append(j_pixels[j])
        cluster_id += 1

    # For each cluster, find the representative pixel with most branches
    T_count = Y_count = X_count = Multi_count = 0
    T_mask = np.zeros((h, w), dtype=bool)
    Y_mask = np.zeros((h, w), dtype=bool)
    X_mask = np.zeros((h, w), dtype=bool)
    Multi_mask = np.zeros((h, w), dtype=bool)
    junctions = []

    for cid in range(cluster_id):
        c_indices = np.where(cluster_labels == cid)[0]
        if len(c_indices) == 0:
            continue

        # Analyze each pixel in the cluster, pick the one with most branches
        best_pixel = None
        best_angles = []
        best_n = 0
        all_pixel_results = []

        for idx in c_indices:
            y, x = j_pixels[idx]
            angles = calculate_branch_angles(skel_bool, y, x, radius=20)
            n_br = len(angles)
            all_pixel_results.append((y, x, n_br, angles))
            if n_br > best_n:
                best_n = n_br
                best_pixel = (y, x)
                best_angles = angles

        # Use the best pixel in the cluster; skip if fewer than 3 branches
        if best_n < 3:
            continue

        y, x = best_pixel
        n_branches = best_n

        if n_branches == 3:
            diffs = []
            for i in range(3):
                for j in range(i + 1, 3):
                    diff = abs(best_angles[i] - best_angles[j])
                    if diff > 180:
                        diff = 360 - diff
                    diffs.append(diff)
            max_diff = max(diffs) if diffs else 0

            if max_diff >= 165:
                jtype = 'T'
                T_count += 1
                T_mask[y, x] = True
            else:
                jtype = 'Y'
                Y_count += 1
                Y_mask[y, x] = True

        elif n_branches == 4:
            jtype = 'X'
            X_count += 1
            X_mask[y, x] = True

        else:
            jtype = 'Multi'
            Multi_count += 1
            Multi_mask[y, x] = True

        junctions.append({
            'pos': (int(y), int(x)),
            'type': jtype,
            'branch_count': n_branches,
            'angles': [round(a, 1) for a in best_angles],
        })

    total = T_count + Y_count + X_count + Multi_count
    if total > 0:
        T_pct = round(T_count / total * 100, 1)
        Y_pct = round(Y_count / total * 100, 1)
        X_pct = round(X_count / total * 100, 1)
        Multi_pct = round(Multi_count / total * 100, 1)
    else:
        T_pct = Y_pct = X_pct = Multi_pct = 0.0

    return {
        'total': total,
        'T_count': T_count, 'T_pct': T_pct,
        'Y_count': Y_count, 'Y_pct': Y_pct,
        'X_count': X_count, 'X_pct': X_pct,
        'Multi_count': Multi_count, 'Multi_pct': Multi_pct,
        'junctions': junctions,
        'vis_info': {
            'T_mask': T_mask,
            'Y_mask': Y_mask,
            'X_mask': X_mask,
            'Multi_mask': Multi_mask,
            'endpoint_mask': true_endpoints,
        },
    }
