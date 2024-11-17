""" isocdf extension """

import numpy as np


def isocdf_seq(weights, cumulative_weights, thresholds, threshold_positions, points):
    """
    Compute isotonic cumulative distribution function sequentially.

    Parameters:
        weights (np.ndarray): Array of weights.
        cumulative_weights (np.ndarray): Array of cumulative weights.
        thresholds (np.ndarray): Array of threshold values.
        threshold_positions (np.ndarray): Array of threshold positions.
        points (np.ndarray): Points where the CDF is evaluated.

    Returns:
        np.ndarray: Computed CDF values.
    """
    weights = np.asarray(weights)
    cumulative_weights = np.asarray(cumulative_weights)
    thresholds = np.asarray(thresholds)
    threshold_positions = np.asarray(threshold_positions, dtype=int)
    points = np.asarray(points)

    num_weights = len(weights)
    num_points = len(points)
    max_point = points[-1]

    # Determine the maximal K such that thresholds[K] < thresholds[K + 1]
    K = len(thresholds) - 1
    while thresholds[K] == thresholds[K - 1]:
        K -= 1
    K -= 1
    if thresholds[K] < max_point:
        max_point = thresholds[K]

    # Prepare output and working arrays
    points = np.append(points, np.inf)
    cdf = np.ones((num_weights, num_points))
    cdf[:, : np.searchsorted(points, thresholds[0])] = 0

    z = np.zeros(num_weights)
    partition_indices = [-1] * (num_weights + 1)
    weight_sums = [0] * (num_weights + 1)
    mean_values = [0] * (num_weights + 1)
    partition_indices[0] = -1

    # Initialization
    d = 1
    partition_indices[d] = threshold_positions[0]
    z[partition_indices[d]] = cumulative_weights[0] / weights[partition_indices[d]]
    weight_sums[d] = weights[: partition_indices[d] + 1].sum()
    mean_values[d] = cumulative_weights[0] / weight_sums[d]
    # ind_cdf = num_weights * np.searchsorted(points, thresholds[0])

    # Main loop
    for k in range(1, len(thresholds)):
        if thresholds[k] > max_point:
            break

        pos_k = threshold_positions[k]
        z[pos_k] += cumulative_weights[k] / weights[pos_k]

        # Update partitions
        partition_start = next(
            idx for idx, val in enumerate(partition_indices) if pos_k <= val
        )
        a = partition_indices[partition_start - 1] + 1
        b = partition_indices[partition_start]
        d = partition_start

        # Update partition sums and means
        partition_indices[d] = pos_k
        weight_sums[d] = weights[a : pos_k + 1].sum()
        mean_values[d] = (
            np.dot(weights[a : pos_k + 1], z[a : pos_k + 1]) / weight_sums[d]
        )

        # Pooling step
        while d > 1 and mean_values[d - 1] <= mean_values[d]:
            mean_values[d - 1] = (
                weight_sums[d - 1] * mean_values[d - 1]
                + weight_sums[d] * mean_values[d]
            ) / (weight_sums[d - 1] + weight_sums[d])
            weight_sums[d - 1] += weight_sums[d]
            partition_indices[d - 1] = partition_indices[d]
            d -= 1

        # Expand partitions if needed
        if pos_k < b:
            for i in range(pos_k + 1, b + 1):
                d += 1
                partition_indices[d] = i
                weight_sums[d] = weights[i]
                mean_values[d] = z[i]
                while d > 1 and mean_values[d - 1] <= mean_values[d]:
                    mean_values[d - 1] = (
                        weight_sums[d - 1] * mean_values[d - 1]
                        + weight_sums[d] * mean_values[d]
                    ) / (weight_sums[d - 1] + weight_sums[d])
                    weight_sums[d - 1] += weight_sums[d]
                    partition_indices[d - 1] = partition_indices[d]
                    d -= 1

        # Store results in CDF
        if thresholds[k - 1] < thresholds[k]:
            while (
                points[np.searchsorted(points, thresholds[k - 1])] < thresholds[k - 1]
            ):
                pass
            while points[np.searchsorted(points, thresholds[k])] < thresholds[k]:
                cdf[:, np.searchsorted(points, thresholds[k])] = mean_values[d]

    return cdf


def pava_correct(data):
    """
    Perform the Pooled Adjacent Violators Algorithm (PAVA) for monotonicity correction.

    Parameters:
        data (np.ndarray): 2D array of data to correct.

    Returns:
        np.ndarray: Monotonic corrected array.
    """
    data = np.asarray(data)
    num_rows, num_cols = data.shape
    result = np.zeros_like(data)

    for row_idx in range(num_rows):
        weights = np.ones(num_cols)
        merged = np.arange(num_cols)

        result[row_idx, 0] = data[row_idx, 0]
        for col_idx in range(1, num_cols):
            merged[col_idx] = col_idx
            result[row_idx, col_idx] = data[row_idx, col_idx]

            while (
                col_idx > 0 and result[row_idx, col_idx] < result[row_idx, col_idx - 1]
            ):
                merged[col_idx] = merged[col_idx - 1]
                weights[col_idx - 1] += weights[col_idx]
                result[row_idx, col_idx - 1] = (
                    weights[col_idx - 1] * result[row_idx, col_idx - 1]
                    + weights[col_idx] * result[row_idx, col_idx]
                ) / weights[col_idx - 1]
                col_idx -= 1

        for col_idx in range(num_cols):
            result[row_idx, merged[col_idx] : col_idx + 1] = result[row_idx, col_idx]

    return result
