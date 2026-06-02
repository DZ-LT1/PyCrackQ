import cv2
import numpy as np
from skimage.filters import threshold_sauvola, threshold_niblack
from skimage.morphology import skeletonize
from skimage.measure import label as sk_label


def apply_binarization(gray_img, method, v1, v2):
    """Apply binarization to a grayscale image using the specified method.

    Args:
        gray_img: Grayscale uint8 image.
        method: Binarization method string. One of: "Global", "Otsu", "Triangle",
                "Sauvola", "Niblack", "Adaptive Mean", "Adaptive Gaussian".
        v1: Window size (odd, >= 3) or threshold value for Global method.
        v2: k parameter (used by Sauvola, Niblack, Adaptive methods).

    Returns:
        Binary uint8 image (255 for foreground/crack pixels).
    """
    if gray_img is None:
        raise ValueError("gray_img is None")

    v1 = int(v1)
    if v1 < 3:
        v1 = 3
    if v1 % 2 == 0:
        v1 += 1
    v2 = float(v2)

    if "Global" in method:
        _, binary = cv2.threshold(gray_img, v1, 255, cv2.THRESH_BINARY)
    elif "Otsu" in method:
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif "Triangle" in method:
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    elif "Sauvola" in method:
        thresh = threshold_sauvola(gray_img, window_size=v1, k=v2)
        binary = ((gray_img > thresh) * 255).astype(np.uint8)
    elif "Niblack" in method:
        thresh = threshold_niblack(gray_img, window_size=v1, k=v2)
        binary = ((gray_img > thresh) * 255).astype(np.uint8)
    elif "Adaptive Mean" in method:
        binary = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, v1, int(v2))
    elif "Adaptive Gaussian" in method:
        binary = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, v1, int(v2))
    else:
        raise ValueError(f"Unknown binarization method: {method}")

    # White pixel inversion check: if more than half the image is white,
    # invert so that cracks remain the minority (foreground).
    white_pixels = cv2.countNonZero(binary)
    if white_pixels > binary.size / 2:
        binary = cv2.bitwise_not(binary)

    return binary


def apply_denoising(binary_img, mode, k):
    """Apply denoising filter to a binary image.

    Args:
        binary_img: Binary uint8 image.
        mode: Denoising mode. One of: "None", "Gaussian", "Mean", "Median".
        k: Kernel size (odd, >= 1).

    Returns:
        Filtered binary uint8 image.
    """
    if binary_img is None:
        raise ValueError("binary_img is None")

    k = int(k)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1

    if "None" in mode:
        result = binary_img.copy()
    elif "Gaussian" in mode:
        blurred = cv2.GaussianBlur(binary_img, (k, k), 0)
        _, result = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    elif "Mean" in mode:
        blurred = cv2.blur(binary_img, (k, k))
        _, result = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    elif "Median" in mode:
        result = cv2.medianBlur(binary_img, k)
    else:
        raise ValueError(f"Unknown denoising mode: {mode}")

    return result


def get_skeleton(binary_img):
    """Skeletonize a binary image.

    Args:
        binary_img: Binary uint8 image.

    Returns:
        Skeleton as uint8 image (255 on skeleton, 0 elsewhere), or None if input is None.
    """
    if binary_img is None:
        return None
    bool_image = binary_img > 127
    skel = skeletonize(bool_image)
    skel_img = (skel * 255).astype(np.uint8)
    return skel_img


def get_distance_map(binary_img):
    """Compute distance transform of a binary image.

    Args:
        binary_img: Binary uint8 image.

    Returns:
        Distance map (float array), or None if input is None.
    """
    if binary_img is None:
        return None
    dist_map = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    return dist_map


def prune_spurs(skel_bool, min_branch_len=8):
    """Iteratively remove short terminal branches (spurs) from a skeleton.

    Args:
        skel_bool: Boolean skeleton array.
        min_branch_len: Minimum branch length to keep (shorter branches are pruned).

    Returns:
        Pruned boolean skeleton array.
    """
    pruned = skel_bool.copy()
    h, w = pruned.shape
    neighbor_kernel = np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]], dtype=np.uint8)

    changed = True
    iteration = 0
    max_iterations = 20

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        pruned_uint8 = pruned.astype(np.uint8)
        neighbor_count = cv2.filter2D(pruned_uint8, -1, neighbor_kernel)
        endpoints = np.argwhere((pruned_uint8 == 1) & (neighbor_count == 1))

        if len(endpoints) == 0:
            break

        for ep_r, ep_c in endpoints:
            if pruned[ep_r, ep_c] == 0:
                continue

            branch = [(ep_r, ep_c)]
            visited = set()
            visited.add((ep_r, ep_c))
            stack = [(ep_r, ep_c)]
            is_spur = True
            branch_length = 0

            while stack and branch_length < min_branch_len:
                cr, cc = stack.pop()
                branch_length += 1
                # Count non-self neighbors on the skeleton.
                n_count = 0
                next_nodes = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and pruned[nr, nc]:
                            n_count += 1
                            if (nr, nc) not in visited:
                                next_nodes.append((nr, nc))
                                visited.add((nr, nc))

                if branch_length >= min_branch_len:
                    is_spur = False
                    break

                if n_count >= 3 and branch_length > 1:
                    is_spur = False
                    break

                for node in next_nodes:
                    stack.append(node)

            if is_spur and 1 <= branch_length < min_branch_len:
                for br, bc in branch:
                    pruned[br, bc] = False
                changed = True

    return pruned


def detect_junctions(skel_bool):
    """Detect junction pixels on a skeleton.

    A junction is a skeleton pixel where removing it splits the local
    3x3 neighborhood into >= 3 connected components (4-connectivity).
    Spur pruning removes short (< 5px) terminal noise before detection.

    Args:
        skel_bool: Boolean skeleton array.

    Returns:
        Tuple of (junction_mask, endpoint_mask, junction_count):
        - junction_mask: Boolean array, True at junction pixels.
        - endpoint_mask: Boolean array, True at endpoint pixels.
        - junction_count: Integer count of junction pixels.
    """
    h, w = skel_bool.shape

    # Prune only very short spurs to reduce noise without damaging real branches.
    pruned = prune_spurs(skel_bool, min_branch_len=5)
    skel_uint8 = pruned.astype(np.uint8)

    # Neighbor count on pruned skeleton.
    neighbor_kernel = np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_uint8, -1, neighbor_kernel)

    endpoint_mask = np.logical_and(pruned, neighbor_count == 1)
    candidate_mask = np.logical_and(pruned, neighbor_count >= 3)

    junction_mask = np.zeros_like(skel_bool, dtype=bool)
    candidate_coords = np.argwhere(candidate_mask)

    for r, c in candidate_coords:
        # Extract 3x3 local window.
        r_min, r_max = max(0, r - 1), min(h, r + 2)
        c_min, c_max = max(0, c - 1), min(w, c + 2)
        local = skel_uint8[r_min:r_max, c_min:c_max].copy()
        local_r, local_c = r - r_min, c - c_min
        local[local_r, local_c] = 0  # Remove the candidate pixel.

        if np.sum(local) < 3:
            continue

        # 4-connectivity: removing the junction must split into >= 3 components.
        labeled_local, num_components = sk_label(local > 0, connectivity=1, return_num=True)
        if num_components >= 3:
            junction_mask[r, c] = True

    # Dilate to break the skeleton cleanly at each junction cluster.
    if np.any(junction_mask):
        dilate_kernel = np.ones((2, 2), dtype=np.uint8)
        junction_mask = cv2.dilate(junction_mask.astype(np.uint8), dilate_kernel).astype(bool)

    junction_count = int(np.sum(junction_mask))
    return junction_mask, endpoint_mask, junction_count


def create_circular_mask(image_shape, center, radius):
    """Create a circular binary mask.

    Args:
        image_shape: Shape tuple (h, w) or (h, w, c) of the target image.
        center: Tuple (x, y) center of the circle.
        radius: Radius of the circle in pixels.

    Returns:
        Binary uint8 mask with 255 inside the circle and 0 outside.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def apply_circular_mask(image, mask):
    """Apply a circular mask to an image.

    Args:
        image: Input image (any number of channels).
        mask: Binary uint8 mask.

    Returns:
        Masked image. If mask is None, returns the input unchanged.
    """
    if mask is None:
        return image
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked


def _odd_clamped(value, min_value=15, max_value=75):
    value = int(round(value))
    value = max(min_value, min(max_value, value))
    if value % 2 == 0:
        value += 1
    if value > max_value:
        value -= 2
    return max(min_value, value)


def _as_gray_uint8(gray_img):
    img = np.asarray(gray_img)
    if img.ndim == 3:
        img = np.mean(img[:, :, :3], axis=2)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _otsu_separability(gray_img):
    hist = np.bincount(gray_img.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0

    prob = hist / total
    bins = np.arange(256, dtype=np.float64)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bins)
    mu_total = mu[-1]

    denom = omega * (1.0 - omega)
    valid = denom > 1e-12
    if not np.any(valid):
        return 0.0

    between = np.zeros_like(denom)
    between[valid] = ((mu_total * omega[valid] - mu[valid]) ** 2) / denom[valid]
    total_var = float(np.var(gray_img.astype(np.float32)))
    if total_var <= 1e-12:
        return 0.0
    return float(np.max(between) / total_var)


def _image_traits(gray_img):
    img = _as_gray_uint8(gray_img)
    h, w = img.shape[:2]
    short_side = max(1, min(h, w))
    img_f = img.astype(np.float32)

    std_val = float(np.std(img_f))
    p1, p5, p95, p99 = np.percentile(img_f, [1, 5, 95, 99])
    dynamic_range = float(max(p95 - p5, p99 - p1, std_val * 4.0, 1.0))

    illumination_kernel = _odd_clamped(short_side / 3.0, 31, 151)
    low_freq = cv2.GaussianBlur(img_f, (illumination_kernel, illumination_kernel), 0)
    illumination_score = float(np.std(low_freq) / dynamic_range)

    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gx ** 2 + gy ** 2)
    normalized_grad = gradient_mag / dynamic_range
    edge_density = float(np.mean(normalized_grad > 0.35))
    edge_strength = float(np.mean(normalized_grad))

    local_kernel = np.ones((9, 9), dtype=np.float32) / 81.0
    local_mean = cv2.filter2D(img_f, -1, local_kernel)
    local_sq_mean = cv2.filter2D(img_f ** 2, -1, local_kernel)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    texture_score = float(np.mean(np.sqrt(local_var)) / dynamic_range)

    if dynamic_range < 35 or std_val < 18:
        contrast_level = "low"
    elif dynamic_range < 80 or std_val < 35:
        contrast_level = "medium"
    else:
        contrast_level = "high"

    if illumination_score >= 0.22:
        illumination_level = "high"
    elif illumination_score >= 0.12:
        illumination_level = "moderate"
    else:
        illumination_level = "low"

    if edge_density >= 0.10 or texture_score >= 0.35:
        complexity_level = "high"
    elif edge_density >= 0.04 or texture_score >= 0.20:
        complexity_level = "medium"
    else:
        complexity_level = "low"

    return {
        "image": img,
        "height": h,
        "width": w,
        "short_side": short_side,
        "std_dev": std_val,
        "dynamic_range": dynamic_range,
        "contrast_level": contrast_level,
        "illumination_score": illumination_score,
        "illumination_level": illumination_level,
        "edge_density": edge_density,
        "edge_strength": edge_strength,
        "texture_score": texture_score,
        "complexity_level": complexity_level,
        "bimodality_score": _otsu_separability(img),
    }


def recommend_parameters(gray_img):
    """Recommend automatic binarization method, window, and k values.

    The recommendation uses histogram separability, low-frequency illumination
    variation, local texture, and edge density. This lets auto mode switch
    between global and local thresholding, and tune local parameters according
    to uneven lighting and crack-network complexity.
    """
    if gray_img is None or gray_img.size == 0:
        return {
            'method': 'Sauvola (Recommended)', 'window': 25, 'k': 0.20,
            'contrast_level': 'medium', 'illumination_level': 'low',
            'complexity_level': 'medium', 'std_dev': 0.0,
            'dynamic_range': 0.0, 'illumination_score': 0.0,
            'edge_density': 0.0, 'texture_score': 0.0,
            'bimodality_score': 0.0, 'reason': 'empty image fallback',
        }

    traits = _image_traits(gray_img)
    short_side = traits["short_side"]
    contrast = traits["contrast_level"]
    illumination = traits["illumination_level"]
    complexity = traits["complexity_level"]
    bimodality = traits["bimodality_score"]

    base_window = _odd_clamped(short_side / 40.0, 15, 35)
    window = base_window

    if illumination == "high":
        window += 32
    elif illumination == "moderate":
        window += 26

    if contrast == "low":
        window += 10
    elif contrast == "high":
        window -= 2

    if complexity == "high":
        window -= 12
    elif complexity == "medium":
        window -= 4

    if traits["texture_score"] > 0.40:
        window += 8

    window = _odd_clamped(window, 15, 75)

    k = 0.20
    if illumination == "high":
        k += 0.08
    elif illumination == "moderate":
        k += 0.04

    if complexity == "high":
        k -= 0.06
    elif complexity == "medium":
        k -= 0.02

    if contrast == "low":
        k -= 0.04
    elif contrast == "high":
        k += 0.02

    if traits["texture_score"] > 0.40:
        k += 0.04

    k = min(0.35, max(0.08, k))

    if illumination == "low" and bimodality >= 0.72 and complexity != "high" and contrast != "low":
        method = "Otsu"
        reason = "uniform illumination and strong bimodal histogram"
        window = 25
        k = 0.0
    elif complexity == "high" and contrast == "low" and illumination != "high":
        method = "Niblack"
        reason = "fine low-contrast crack network"
        window = _odd_clamped(min(window, 35), 15, 75)
        k = min(0.18, max(0.08, k))
    else:
        method = "Sauvola (Recommended)"
        if illumination == "high":
            reason = "uneven illumination requires local statistics"
        elif complexity == "high":
            reason = "complex crack network requires local detail preservation"
        else:
            reason = "balanced local thresholding for soil crack texture"

    return {
        'method': method,
        'window': window,
        'k': round(k, 3),
        'contrast_level': contrast,
        'illumination_level': illumination,
        'complexity_level': complexity,
        'std_dev': round(traits["std_dev"], 1),
        'dynamic_range': round(traits["dynamic_range"], 1),
        'illumination_score': round(traits["illumination_score"], 3),
        'edge_density': round(traits["edge_density"], 3),
        'texture_score': round(traits["texture_score"], 3),
        'bimodality_score': round(bimodality, 3),
        'reason': reason,
    }
