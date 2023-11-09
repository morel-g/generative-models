import numpy as np

TOY_DISCRETE_PARAMS = {"min": -3.0, "max": 3.0}


def get_toy_discrete_params():
    return TOY_DISCRETE_PARAMS


def has_density(x_center, y_center, dx, dy, samples):
    """
    Returns True if there are samples within the cell centered at (x_center, y_center) with dimensions dx x dy.
    """
    for x, y in samples:
        if (x_center - dx / 2 <= x <= x_center + dx / 2) and (
            y_center - dy / 2 <= y <= y_center + dy / 2
        ):
            return True
    return False


def continuous_to_discrete_2d(samples, N):
    """
    Returns the indices of the cells that have density (samples within them).
    """
    x_min, x_max = (
        TOY_DISCRETE_PARAMS["min"],
        TOY_DISCRETE_PARAMS["max"],
    )
    dx = (x_max - x_min) / N
    dy = (x_max - x_min) / N

    # Convert continuous samples into discrete indices
    x_indices = ((samples[:, 0] - x_min) / dx).astype(int)
    y_indices = ((samples[:, 1] - x_min) / dy).astype(int)

    # Create a 2D grid
    grid = np.zeros((N, N), dtype=int)

    # For each sample, update the corresponding cell in the grid
    for x, y in zip(x_indices, y_indices):
        grid[x, y] += 1

    # Generate the true_cells list based on grid values
    true_cells = []
    for (x, y), count in np.ndenumerate(grid):
        true_cells.extend([(x, y)] * count)

    return true_cells


def get_cell_center(i, j, N):
    """
    Returns the center coordinates of the cell with indices (i, j).
    """
    x_min, x_max = (
        TOY_DISCRETE_PARAMS["min"],
        TOY_DISCRETE_PARAMS["max"],
    )
    # Calculate the size of each cell
    dx = (x_max - x_min) / N
    dy = (x_max - x_min) / N

    # Calculate the center of the cell
    x_center = x_min + i * dx + dx / 2
    y_center = x_min + j * dy + dy / 2

    return (x_center, y_center)
