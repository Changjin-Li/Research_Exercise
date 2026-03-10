import os
import numpy as np
import trimesh
import glob
import bisect
from numba import njit

def accumulate_sim3_transforms(transforms):
    """
    Accumulate adjacent SIM(3) transforms into transforms from the initial frame to each subsequent frame.
    Args:
    transforms: list, each element is a tuple (R, s, t)
        R: 3x3 rotation matrix (np.array)
        s: scale factor (scalar)
        t: 3x1 translation vector (np.array)
    Returns:
    Cumulative transforms list, each element is (R_cum, s_cum, t_cum) representing the transform from frame 0 to frame k
    """
    if not transforms:
        return []

    cumulative_transforms = [transforms[0]]

    for i in range(1, len(transforms)):
        s_cum_prev, R_cum_prev, t_cum_prev = cumulative_transforms[i-1]
        s_next, R_next, t_next = transforms[i]
        R_cum_next = R_cum_prev @ R_next
        s_cum_next = s_cum_prev * s_next
        t_cum_next = s_cum_prev * (R_cum_prev @ t_next) + t_cum_prev
        cumulative_transforms.append((s_cum_next, R_cum_next, t_cum_next))

    return cumulative_transforms

def estimate_sim3(source_points, target_points):
    mu_src = np.mean(source_points, axis=0)
    mu_tgt = np.mean(target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt((src_centered ** 2).sum(axis=1).mean())
    scale_tgt = np.sqrt((tgt_centered ** 2).sum(axis=1).mean())
    s = scale_tgt / scale_src

    src_scaled = src_centered * s

    H = src_scaled.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - s * (R @ mu_src)
    return s, R, t

def align_point_maps(point_map1, conf1, point_map2, conf2, conf_threshold):
    """point_map2 -> point_map1"""
    b1 = point_map1.shape[0]
    b2 = point_map2.shape[0]
    b = min(b1, b2)

    aligned_points1, aligned_points2 = [], []
    for i in range(b):
        mask1 = conf1[i] > conf_threshold
        mask2 = conf2[i] > conf_threshold
        valid_mask = mask1 & mask2

        idx = np.where(valid_mask)
        if len(idx[0]) == 0:
            continue

        pts1 = point_map1[i][idx]
        pts2 = point_map2[i][idx]
        aligned_points1.append(pts1)
        aligned_points2.append(pts2)

    if len(aligned_points1) == 0:
        raise ValueError("No matching point pairs were found!")

    all_pts1 = np.concatenate(aligned_points1, axis=0)
    all_pts2 = np.concatenate(aligned_points2, axis=0)
    print(f"The number of corresponding points matched: {all_pts1.shape[0]}")

    s, R, t = estimate_sim3(all_pts1, all_pts2)

    mean_error = compute_alignment_error(point_map1, conf1, point_map2, conf2, conf_threshold, s, R, t)
    print(f'Mean error: {mean_error}')

    return s, R, t

def apply_sim3(points, s, R, t):
    return (s * (R @ points.T)).T + t

def apply_sim3_direct(point_maps, s, R, t):
    # point_maps: (b, h, w, 3) -> (b, h, w, 3, 1)
    point_maps_expanded = point_maps[..., np.newaxis]

    # R: (3, 3) -> (b, h, w, 3, 1) = (3, 3) @ (3, 1)
    rotated = np.matmul(R, point_maps_expanded)
    rotated = rotated.squeeze(-1) # (b, h, w, 3)
    transformed = s * rotated + t
    return transformed

def compute_alignment_error(point_map1, conf1, point_map2, conf2, conf_threshold, s, R, t):
    """
    Compute the average point alignment error (using only original inputs).
    Args:
        point_map1: target point map (b, h, w, 3)
        conf1: target confidence map (b, h, w)
        point_map2: source point map (b, h, w, 3)
        conf2: source confidence map (b, h, w)
        conf_threshold: confidence threshold
        s, R, t: transformation parameter
    """
    b1, h1, w1, _ = point_map1.shape
    b2, h2, w2, _ = point_map2.shape
    b = min(b1, b2)
    h = min(h1, h2)
    w = min(w1, w2)

    target_points = []
    source_points = []

    for i in range(b):
        mask1 = conf1[i, :h, :w] > conf_threshold
        mask2 = conf2[i, :h, :w] > conf_threshold
        valid_mask = mask1 & mask2
        idx = np.where(valid_mask)
        if len(idx[0]) == 0:
            continue

        t_pts = point_map1[i, :h, :w][idx]
        s_pts = point_map2[i, :h, :w][idx]
        target_points.append(t_pts)
        source_points.append(s_pts)

    if len(target_points) == 0:
        print("Warning: No matching point pairs found for error calculation.")
        return np.nan

    all_target = np.concatenate(target_points, axis=0)
    all_source = np.concatenate(source_points, axis=0)
    transformed = apply_sim3(all_source, s, R, t)

    error = np.linalg.norm(transformed - all_target, axis=1)
    mean_error = np.mean(error)
    std_error = np.std(error)
    median_error = np.median(error)
    max_error = np.max(error)
    print(  f"Alignment error statistics [using {len(error)} points]: "
            f"mean={mean_error:.4f}, std={std_error:.4f}, "
            f"median={median_error:.4f}, max={max_error:.4f}")
    return mean_error

def save_confident_pointcloud(points: np.ndarray, colors: np.ndarray, confs: np.ndarray, output_path: str, conf_threshold: float, sample_ratio: float=1.0):
    """
    Filter points based on confidence threshold and save as PLY file, with optional random sampling ratio.
    Args:
    - points: shape (H, W, 3) or (N, 3)
    - colors: shape (H, W, 3) or (N, 3)
    - confs: shape (H, W) or (N)
    - output_path: output PLY file path
    - conf_threshold: confidence threshold for point filtering
    - sample_ratio: sampling ratio (0 < sample_ratio <= 1.0)
    """
    points = points.reshape(-1, 3).astype(np.float32, copy=False)
    colors = colors.reshape(-1, 3).astype(np.uint8, copy=False)
    confs = confs.reshape(-1).astype(np.float32, copy=False)

    conf_mask = (confs >= conf_threshold) & (confs > 1e-5)
    points = points[conf_mask]
    colors = colors[conf_mask]

    if 0 < sample_ratio < 1.0 and len(points) > 0:
        num_points = int(len(points) * sample_ratio)
        idx = np.random.choice(len(points), num_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f'shape of sampled point: {points.shape}')
    trimesh.PointCloud(points, colors).export(output_path)
    print(f"Saved point cloud with {len(points)} points to {output_path}")

def save_confident_pointcloud_batch(points: np.ndarray, colors: np.ndarray, confs: np.ndarray, output_path: str, conf_threshold: float, sample_ratio: float=1.0, batch_size: int=1000000):
    if points.ndim == 2:
        b = 1
        points = points[np.newaxis, ...]
        colors = colors[np.newaxis, ...]
        confs = confs[np.newaxis, ...]
    elif points.ndim == 4:
        b = points.shape[0]
    else:
        raise ValueError("Unsupported points dimension. Must be 2 (N,3) or 4 (b,H,W,3).")

    total_valid = 0
    for i in range(b):
        cfs = confs[i].reshape(-1)
        total_valid += np.count_nonzero((cfs >= conf_threshold) & (cfs > 1e-5))

    num_samples = int(total_valid * sample_ratio) if sample_ratio < 1.0 else total_valid
    if num_samples == 0:
        save_ply(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), output_path)
        return

    if sample_ratio == 1.0:
        with open(output_path, 'wb') as f:
            write_ply_header(f, num_samples)

            for i in range(b):
                pts = points[i].reshape(-1, 3).astype(np.float32)
                cols = colors[i].reshape(-1, 3).astype(np.uint8)
                cfs = confs[i].reshape(-1).astype(np.float32)
                valid_mask = (cfs >= conf_threshold) & (cfs > 1e-5)
                valid_pts = pts[valid_mask]
                valid_cols = cols[valid_mask]

                for j in range(0, len(valid_pts), batch_size):
                    batch_pts = valid_pts[j:j+batch_size]
                    batch_cols = valid_cols[j:j+batch_size]
                    write_ply_batch(f, batch_pts, batch_cols)
    else:
        reservoir_pts = np.zeros((num_samples, 3), dtype=np.float32)
        reservoir_cols = np.zeros((num_samples, 3), dtype=np.uint8)
        count = 0

        for i in range(b):
            pts = points[i].reshape(-1, 3).astype(np.float32)
            cols = colors[i].reshape(-1, 3).astype(np.uint8)
            cfs = confs[i].reshape(-1).astype(np.float32)
            valid_mask = (cfs >= conf_threshold) & (cfs > 1e-5)
            valid_pts = pts[valid_mask]
            valid_cols = cols[valid_mask]
            valid_count = len(valid_pts)

            if count < num_samples:
                fill_count = min(num_samples - count, valid_count)
                reservoir_pts[count:count+fill_count] = valid_pts[:fill_count]
                reservoir_cols[count:count+fill_count] = valid_cols[:fill_count]
                count += fill_count

                if fill_count < valid_count:
                    remaining_pts = valid_pts[fill_count:]
                    remaining_cols = valid_cols[fill_count:]
                    count, reservoir_pts, reservoir_cols = optimized_vectorized_reservoir_sampling(remaining_pts, remaining_cols, count, reservoir_pts, reservoir_cols)
                else:
                    count, reservoir_pts, reservoir_cols = optimized_vectorized_reservoir_sampling(valid_pts, valid_cols, count, reservoir_pts, reservoir_cols)

        save_ply(reservoir_pts, reservoir_cols, output_path)

def optimized_vectorized_reservoir_sampling(
    new_points: np.ndarray,
    new_colors: np.ndarray,
    current_count: int,
    reservoir_points: np.ndarray,
    reservoir_colors: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Optimized vectorized reservoir sampling with batch probability calculations.
    This maintains mathematical correctness while improving performance through vectorized operations where possible.
    Args:
        new_points: New point coordinates to consider, shape (M, 3)
        new_colors: New point colors to consider, shape (M, 3)
        current_count: Number of elements seen so far
        reservoir_points: Current reservoir of sampled points, shape (K, 3)
        reservoir_colors: Current reservoir of sampled colors, shape (K, 3)
    Returns:
        Tuple of (updated_count, updated_reservoir_points, updated_reservoir_colors)
    """
    random_gen = np.random
    reservoir_size = len(reservoir_points)
    num_new_points = len(new_points)
    if num_new_points == 0:
        return current_count, reservoir_points, reservoir_colors

    # Calculate sequential indices for each new point
    point_indices = np.arange(current_count + 1, current_count + num_new_points + 1)
    # Generate random numbers for each point
    random_values = random_gen.randint(0, point_indices, size=num_new_points)
    # Determine which points should replace reservoir elements
    replace_mask = random_values < reservoir_size
    replace_positions = random_values[replace_mask]
    # Apply replacements
    if np.any(replace_mask):
        points_to_replace = new_points[replace_mask]
        colors_to_replace = new_colors[replace_mask]
        reservoir_points[replace_positions] = points_to_replace
        reservoir_colors[replace_positions] = colors_to_replace

    return current_count + num_new_points, reservoir_points, reservoir_colors

def write_ply_header(f, num_vertices):
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]
    f.write("\n".join(header).encode() + b"\n")

def write_ply_batch(f, points, colors):
    structured = np.zeros(len(points), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('red', np.uint8),
        ('green', np.uint8),
        ('blue', np.uint8)
    ])
    structured['x'] = points[:, 0]
    structured['y'] = points[:, 1]
    structured['z'] = points[:, 2]
    structured['red'] = colors[:, 0]
    structured['green'] = colors[:, 1]
    structured['blue'] = colors[:, 2]
    f.write(structured.tobytes())

def save_ply(points: np.ndarray, colors: np.ndarray, filename: str):
    with open(filename, 'wb') as f:
        write_ply_header(f, len(points))
        write_ply_batch(f, points, colors)

