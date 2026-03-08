import numpy as np


def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """If mask has more than max_trues True values, randomly keep only max_trues of them and set the rest to False."""
    true_index = np.flatnonzero(mask)
    if len(true_index) <= max_trues:
        return mask

    sampled_index = np.random.choice(true_index, size=max_trues, replace=False)
    limit_flat_mask = np.zeros(mask.size, dtype=bool)
    limit_flat_mask[sampled_index] = True
    return limit_flat_mask.reshape(mask.shape)


def create_pixel_coordinate_grid(num_frames: int, height: int, width: int):
    """
    Create a grid of pixel coordinate and frame indices for each frame.
    :return: Tuple [x_coords, y_coords, f_coords]: shape (num_frames, height, width, 3)
        - y_coords: Array of y coordinates for all frames
        - x_coords: Array of x coordinates for all frames
        - f_coords: Array of frame indices for all frames
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))
    f_index = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_index, (num_frames, height, width))
    return np.stack((x_coords, y_coords, f_coords), axis=-1)

