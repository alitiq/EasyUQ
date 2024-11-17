"""
utility function for easyuq model
Author: Daniel Lassahn based upon work of E.Walz (2024)
"""

from dataclasses import dataclass
import pandas as pd
import bisect
from typing import List, Optional, Tuple, Dict, Union
from scipy import sparse
from scipy.stats import rankdata
import osqp
import numpy as np
from scipy.interpolate import interp1d
from easyuq.isocdf_seq import isocdf_seq, pava_correct
from itertools import groupby
from collections import defaultdict

class PredictionResult:
    """
    A class for handling predictions and computing related metrics such as quantiles and CRPS.

    Attributes:
        predictions (list or np.ndarray): The predicted cumulative distribution functions (CDFs).
        thresholds (np.ndarray): Threshold values corresponding to the predictions.
    """

    def __init__(
        self,
        predictions: Union[List[float], np.ndarray],
        thresholds: Union[List[float], np.ndarray],
    ) -> None:
        """
        Initialize a PredictionResult instance.

        Args:
            predictions (list or np.ndarray): The predicted CDF values.
            thresholds (list or np.ndarray): The threshold points corresponding to the predictions.
        """
        self.predictions = predictions
        self.thresholds = thresholds

    def qpred(self, quantiles: np.ndarray) -> np.ndarray:
        """
        Generate quantile predictions for specified quantile levels.

        Args:
            quantiles (array-like): Quantile levels to predict (must be in [0, 1]).

        Returns:
            np.ndarray: Quantile predictions for the requested quantiles.

        Raises:
            ValueError: If quantiles are outside the range [0, 1].
        """
        quantiles = np.asarray(quantiles)

        if np.min(quantiles) < 0 or np.max(quantiles) > 1:
            raise ValueError("Quantiles must be a numeric array with values in [0, 1].")

        def interpolate_single_prediction(data: Dict[str, np.ndarray]) -> np.ndarray:
            """
            Interpolate quantiles for a single prediction using adaptive thresholds.

            Args:
                data: A single prediction object containing 'ecdf' and 'points'.

            Returns:
                np.ndarray: Interpolated quantile predictions.
            """
            # Concatenate ECDF and the maximum value of ECDF and corresponding points
            x_vals = np.hstack([data.ecdf, np.max(data.ecdf)])
            y_vals = np.hstack([data.points, data.points[-1]])

            # Use adaptive interpolation based on thresholds (quantiles)
            return interp1d_adapt_q(
                prediction=x_vals, observation=y_vals, thresholds=quantiles
            )
        # Apply interpolation to all predictions and return the stacked result
        return np.vstack(
            [interpolate_single_prediction(pred) for pred in self.predictions]
        ).squeeze()

    def crps(self, observations: Union[List[float], np.ndarray]) -> List[float]:
        """
        Compute the Continuous Ranked Probability Score (CRPS) for the predicted CDF and the true values.

        Args:
            observations (np.ndarray): The true observations.

        Returns:
            CRPS values for each data point.
        """
        predictions = self.predictions
        if type(predictions) is not list:
            raise ValueError("predictions must be a list")

        observations = np.asarray(observations)

        if observations.ndim > 1:
            raise ValueError("obs must be a 1-D array")

        if np.isnan(np.sum(observations)) == True:
            raise ValueError("obs contains nan values")

        if observations.size != 1 and len(observations) != len(predictions):
            raise ValueError("obs must have length 1 or the same length as predictions")

        def get_points(predictions):
            return np.array(predictions.points)

        def get_cdf(predictions):
            return np.array(predictions.ecdf)

        def modify_points(points):
            return np.hstack([points[0], np.diff(points)])

        def crps0(y, p, w, x) -> np.ndarray:
            return 2 * np.sum(w * (np.array((y < x)) - p + 0.5 * w) * np.array(x - y))

        x = list(map(get_points, predictions))
        p = list(map(get_cdf, predictions))
        w = list(map(modify_points, p))

        return list(map(crps0, observations, p, w, x))


def interp1d_adapt_q(
    prediction: np.ndarray, observation: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """
    Performs robust interpolation between prediction and observation values based on thresholds.

    The function adapts the interpolation method based on the comparison of the thresholds
    with the maximum value of the prediction array. It uses either 'previous' or 'next'
    interpolation methods depending on the threshold value relative to the maximum points.

    Args:
        prediction (np.ndarray): 1D array of predicted values (x-axis).
        observation (np.ndarray): 1D array of observed values (y-axis).
        thresholds (np.ndarray): Array of quantiles to evaluate the interpolation.

    Returns:
        np.ndarray: Interpolated values based on the provided thresholds.
    """
    min_points = np.max(prediction)

    if np.any(thresholds > min_points):
        inter_vals = np.zeros(thresholds.shape)
        ix1 = np.where(thresholds > min_points)[0]  # Thresholds larger than min_points
        ix2 = np.where(thresholds <= min_points)[
            0
        ]  # Thresholds smaller or equal to min_points

        if ix1.size > 0:
            inter_vals[ix1] = interp1d(
                prediction, observation, kind="previous", fill_value="extrapolate"
            )(thresholds[ix1])
        if ix2.size > 0:
            inter_vals[ix2] = interp1d(
                prediction, observation, kind="next", fill_value="extrapolate"
            )(thresholds[ix2])

        return inter_vals
    else:
        # If all thresholds are less than or equal to min_points, use 'next' interpolation
        return interp1d(prediction, observation, kind="next", fill_value="extrapolate")(
            thresholds
        )


def prepare_data_for_idr(
    covariates: pd.DataFrame, groups: Dict[str, str], orders: Dict[str, str]
) -> pd.DataFrame:
    """
    Prepares data for IDR modeling by organizing and optionally ordering columns.

    Args:
        covariates (pd.DataFrame): DataFrame of covariates.
        groups (Dict[str, str]): A dictionary mapping column names of `covariates` to group identifiers.
        orders (Dict[str, str]): A dictionary mapping group identifiers to their ordering type.
            Possible values:
            - "comp": No ordering applied.
            - "sd": Descending sort applied.
            - Other: Cumulative sum applied after sorting.

    Returns:
        pd.DataFrame: The modified DataFrame with covariates ordered based on group and order type.
    """
    # Group column names by group and sort them
    # grouped_cols = {
    #     group: list(cols)
    #     for group, cols in groupby(
    #         sorted(groups.items(), key=lambda x: x[1]), key=lambda x: x[1]
    #     )
    # }
    grouped_cols = defaultdict(list)
    for key, val in sorted(groups.items()):
        grouped_cols[val].append(key)

    # Process each group
    for key, val in grouped_cols.items():
        if len(val)>1:
            if orders[str(int(key))] == "comp":
                continue
            tmp = -np.sort(-covariates[val], axis=1)
            if orders[str(int(key))] == "sd":
                covariates[val] = tmp
            else:
                covariates[val] = np.cumsum(tmp, axis=1)
    return covariates


def componentwise_partial_order(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the componentwise partial order on rows of a matrix.

    This function identifies pairs of rows where one row is smaller
    than another in the componentwise order.

    Args:
        matrix (np.ndarray): A 2D array with at least two rows.

    Returns:
        tuple: A tuple containing:
            - `paths` (np.ndarray): A two-column array where the first column
              contains the indices of rows that are smaller in the
              componentwise order, and the second column contains the indices
              of the corresponding greater rows.
            - `column_order` (np.ndarray): An array of the same shape as `matrix`,
              where each column contains the indices of sorted values in
              ascending order for that row.

    Raises:
        ValueError: If `matrix` has fewer than two rows or is not 2D.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("`matrix` must be a 2D array with at least two rows.")

    transposed_matrix = matrix.T  # Transpose for easier row-based operations
    num_columns = transposed_matrix.shape[1]
    num_rows = transposed_matrix.shape[0]

    column_order = np.argsort(transposed_matrix, axis=1)  # Sort indices for each row
    ranks = np.apply_along_axis(rankdata, axis=1, arr=transposed_matrix, method="max")

    smaller_indices = []
    greater_indices = []

    for current_col in range(num_columns):
        # Identify potential "smaller" rows for the current column
        smaller_mask = np.full(num_columns, False)
        smaller_mask[column_order[0, : ranks[0, current_col]]] = True

        # Refine smaller rows based on ranks for each subsequent dimension
        for dim in range(1, num_rows):
            if ranks[dim, current_col] < num_columns:
                smaller_mask[column_order[dim, ranks[dim, current_col] :]] = False

        smaller_rows = np.where(smaller_mask)[0]
        num_smaller_rows = len(smaller_rows)

        # Add relationships to the results
        smaller_indices.extend(smaller_rows)
        greater_indices.extend([current_col] * num_smaller_rows)

    # Construct the result as a 2D array
    paths = np.vstack([smaller_indices, greater_indices])

    return paths, column_order.T


def tr_reduc(paths: np.ndarray, n: int) -> np.ndarray:
    """
    Perform the transitive reduction of a path matrix for a directed acyclic graph (DAG).

    This function computes the transitive reduction of a given path matrix,
    eliminating redundant edges while maintaining the reachability properties of the graph.

    Args:
        paths (np.ndarray): A 2D array where the first row contains indices of smaller vertices
            and the second row contains indices of greater vertices, representing edges.
        n (int): The number of vertices in the graph.

    Returns:
        np.ndarray: A 2D array of nonzero entries (edges) in the reduced path matrix.
    """
    # Initialize an adjacency matrix with False (no edges)
    edges = np.zeros((n, n), dtype=bool)

    # Set edges from the input path matrix
    edges[paths[0], paths[1]] = True

    # Ensure no vertex has an edge to itself
    np.fill_diagonal(edges, False)

    # Perform transitive reduction
    for k in range(n):
        # Remove redundant edges: if vertex `i` connects to `k` and `k` connects to `j`,
        # remove the direct edge from `i` to `j`.
        edges[np.ix_(edges[:, k], edges[k])] = False

    # Extract the indices of nonzero (True) entries as the reduced edge set
    return np.array(np.nonzero(edges))


@dataclass
class PredictionsIdr:
    """Idr Predictions dataclass object"""

    ecdf: np.ndarray
    points: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def neighbor_points(x, X, order_X):
    """
    Neighbor points with respect to componentwise order

    Parameters
    ----------
    x : np.array
        Two-dimensional array
    X : Two dimensional array with at least to columns
    order_X : output of function compOrd(X)

    Returns
    -------
    list given for each x[i,] the indices
    of smaller and greater neighbor points within the rows of X

    """
    X = np.asarray(X)
    x = np.asarray(x)
    col_order = order_X[1]

    nx = x.shape[0]
    k = x.shape[1]
    n = X.shape[0]
    ranks_left = np.zeros((nx, k))
    ranks_right = np.zeros((nx, k))
    for j in range(k):
        ranks_left[:, j] = np.searchsorted(a=X[:, j], v=x[:, j], sorter=col_order[:, j])
        ranks_right[:, j] = np.searchsorted(
            a=X[:, j], v=x[:, j], side="right", sorter=col_order[:, j]
        )

    x_geq_X = np.full((n, nx), False)
    x_leq_X = np.full((n, nx), True)
    for i in range(nx):
        if ranks_right[i, 0] > 0:
            x_geq_X[col_order[0 : int(ranks_right[i, 0]), 0], i] = True
        if ranks_left[i, 0] > 0:
            x_leq_X[col_order[0 : int(ranks_left[i, 0]), 0], i] = False
        for j in range(1, k):
            if ranks_right[i, j] < n:
                x_geq_X[col_order[int(ranks_right[i, j]) : n, j], i] = False
            if ranks_left[i, j] > 0:
                x_leq_X[col_order[0 : int(ranks_left[i, j]), j], i] = False
    paths = np.full((n, n), False)
    paths[order_X[0][0], order_X[0][1]] = True
    np.fill_diagonal(paths, False)

    for i in range(n):
        x_leq_X[np.ix_(paths[i, :], x_leq_X[i, :])] = False
        x_geq_X[np.ix_(paths[:, i], x_geq_X[i, :])] = False

    smaller = []
    greater = []

    for i in range(nx):
        smaller.append(x_geq_X[:, i].nonzero()[0])
        greater.append(x_leq_X[:, i].nonzero()[0])

    return smaller, greater


class Idr:
    """IDR prediction model"""

    def __init__(
        self,
        ecdf: np.ndarray,
        thresholds: np.ndarray,
        indices: List[List[int]],
        X: pd.DataFrame,
        y: pd.Series,
        groups: Dict,
        orders: Dict,
        constraints: Optional[np.ndarray] = None,
    ):
        """
        Initialize the IDR object.

        Args:
            ecdf (np.ndarray): Empirical CDF values.
            thresholds (np.ndarray): Thresholds where CDF is evaluated.
            indices (List[List[int]]): Indices of covariates in the original dataset.
            X (pd.DataFrame): Data frame of covariate values used in fitting the model.
            y (pd.Series): Response values.
            groups (Dict): Groups used for ordering in the model.
            orders (Dict): Order constraints used in fitting.
            constraints (Optional[np.ndarray]): Order constraints, default is None.
        """
        self.ecdf = ecdf
        self.thresholds = thresholds
        self.indices = indices
        self.X = X
        self.y = y
        self.groups = groups
        self.orders = orders
        self.constraints = constraints

    def predict(self, data: Optional[pd.DataFrame] = None, digits: int = 3):
        """
        Make predictions based on the fitted IDR model.

        Args:
            data (pd.DataFrame, optional): Data to predict on. If None, predictions are made on the training data.
            digits (int, optional): Number of decimal places to round predictive CDF values. Default is 3.

        Returns:
            idrpredict: An object of class `idrpredict` containing the predictions.
        """
        cdf = np.round(self.ecdf, digits)
        thresholds = self.thresholds.copy()

        if data is None:
            preds = self._predict_on_training_data(cdf, thresholds, digits)
        else:
            preds = self._predict_on_new_data(data, cdf, thresholds, digits)

        return preds

    def _predict_on_training_data(
        self, cdf: np.ndarray, thresholds: np.ndarray, digits: int
    ):
        """
        Helper function to predict on the training data.

        Args:
            cdf (np.ndarray): CDF values for the training data.
            thresholds (np.ndarray): Threshold values.
            digits (int): Rounding precision.

        Returns:
            idrpredict: An object containing predictions on the training data.
        """
        preds = []
        order_indices = []

        for i, idx_group in enumerate(self.indices):
            edf = cdf[i, :]
            sel = np.hstack([edf[0] > 0, np.diff(edf) > 0])
            tmp = PredictionsIdr(
                ecdf=edf[sel],
                points=thresholds[sel],
                lower=np.array([]),
                upper=np.array([]),
            )
            for j in idx_group:
                order_indices.append(j)
                preds.append(tmp)

        preds_rearranged = [preds[k] for k in np.argsort(order_indices)]
        return PredictionResult(predictions=preds_rearranged, thresholds=None)

    def _predict_on_new_data(
        self, data: pd.DataFrame, cdf: np.ndarray, thresholds: np.ndarray, digits: int
    ):
        """
        Helper function to predict on new data.

        Args:
            data (pd.DataFrame): Data frame to make predictions on.
            cdf (np.ndarray): CDF values from the model fit.
            thresholds (np.ndarray): Threshold values.
            digits (int): Rounding precision.

        Returns:
            idrpredict: An object containing predictions on the new data.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        X = self.X.copy()
        data = prepare_data_for_idr(
            data[X.columns], groups=self.groups, orders=self.orders
        )
        nVar = data.shape[1]
        if nVar == 1:
            return self._predict_single_variable(data, cdf, thresholds, digits)
        else:
            return self._predict_multivariate(data, cdf, thresholds, digits)

    def _predict_single_variable(
        self, data: pd.DataFrame, cdf: np.ndarray, thresholds: np.ndarray, digits: int
    ) -> PredictionResult:
        """
        Predict on new data with a single variable.

        Args:
            data (pd.DataFrame): Single variable data for prediction.
            cdf (np.ndarray): CDF values.
            thresholds (np.ndarray): Threshold values.
            digits (int): Rounding precision.

        Returns:
            PredictionResult: Predictions for single variable data.
        """
        X_train = np.array(self.X[self.X.columns[0]])
        x_test = np.array(data[data.columns[0]])

        smaller = np.array([bisect.bisect_left(X_train, a) for a in x_test])
        smaller = np.where(smaller == 0, 1, smaller) - 1
        wg = (
            np.interp(
                x_test,
                X_train,
                np.arange(1, X_train.shape[0] + 1),
                left=1,
                right=X_train.shape[0],
            )
            - np.arange(1, X_train.shape[0] + 1)[smaller]
        )
        greater = smaller + (wg > 0).astype(int)
        ws = 1 - wg

        preds = list(
            map(
                self._interpolate_predictions,
                cdf[greater.astype(int)],
                cdf[smaller.astype(int)],
                ws,
                wg,
            )
        )
        return PredictionResult(predictions=preds, thresholds=None)

    def _interpolate_predictions(
        self,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        weight_small_values: float,
        weight_great_values: float,
    ) -> "PredictionsIdr":
        """
        Helper function to interpolate between predictions for new data.

        Args:
            lower_bound (np.ndarray): Lower bound CDF.
            upper_bound (np.ndarray): Upper bound CDF.
            weight_small_values (float): Weight for the smaller value.
            weight_great_values (float): Weight for the greater value.

        Returns:
            predictions_idr: Interpolated predictions for the new data.
        """
        ls = np.insert(lower_bound[:-1], 0, 0)
        us = np.insert(upper_bound[:-1], 0, 0)
        ind = (ls < lower_bound) | (us < upper_bound)
        lower_bound = lower_bound[ind]
        upper_bound = upper_bound[ind]
        cdf = np.round(
            lower_bound * weight_great_values + upper_bound * weight_small_values, 3
        )
        return PredictionsIdr(
            ecdf=cdf, points=self.thresholds[ind], lower=lower_bound, upper=upper_bound
        )

    def _predict_multivariate(
        self, data: pd.DataFrame, cdf: np.ndarray, thresholds: np.ndarray, digits: int
    ):
        """
        Predict on new data with multiple variables.

        Args:
            data (pd.DataFrame): Multivariate data for prediction.
            cdf (np.ndarray): CDF values.
            thresholds (np.ndarray): Threshold values.
            digits (int): Rounding precision.

        Returns:
            idrpredict: Predictions for multivariate data.
        """
        smaller, greater = neighbor_points(data, self.X, order_X=self.constraints)

        incomparables = np.array(list(map(len, smaller))) + np.array(list(map(len, greater))) == 0

        preds = []
        order_indices = []

        if any(incomparables):
            y = self.y
            edf = np.round(ecdf_formal(thresholds, y.explode()), digits)
            sel = edf > 0
            edf = edf[sel]
            points = thresholds[sel]
            upr = np.where(edf == 1)[0]
            if upr < len(edf) - 1:
                points = np.delete(points, np.arange(upr, len(edf)))
                edf = np.delete(edf, np.arange(upr, len(edf)))
                # dat = {'points':points, 'lower':edf, 'cdf':edf, 'upper':edf}
            # tmp = pd.DataFrame(dat, columns = ['points', 'lower', 'cdf', 'upper'])
            tmp = PredictionsIdr(ecdf=edf, points=points, lower=edf, upper=edf)
            for i in np.where(incomparables == True)[0]:
                preds.append(tmp)
                order_indices.append(i)
        for i in np.where(incomparables == False)[0]:
            if smaller[i].size > 0 and greater[i].size == 0:
                upper = np.round(np.amin(cdf[smaller[i].astype(int), :], axis=0), digits)
                sel = np.hstack([upper[0] != 0, np.diff(upper) != 0])
                upper = upper[sel]
                lower = np.zeros(len(upper))
                estimCDF = upper
            elif smaller[i].size == 0 and greater[i].size > 0:
                lower = np.round(np.amax(cdf[greater[i].astype(int), :], axis=0), digits)
                sel = np.hstack([lower[0] != 0, np.diff(lower) != 0])
                lower = lower[sel]
                upper = np.ones(len(lower))
                estimCDF = lower
            else:
                lower = np.round(np.amax(cdf[greater[i].astype(int), :], axis=0), digits)
                upper = np.round(np.amin(cdf[smaller[i].astype(int), :], axis=0), digits)
                sel = np.hstack([lower[0] != 0, np.diff(lower) != 0]) + np.hstack([upper[0] != 0, np.diff(upper) != 0])
                lower = lower[sel]
                upper = upper[sel]
                estimCDF = np.round(0.5 * (lower + upper), digits)

            # dat = {'points': thresholds[sel], 'lower': lower, 'cdf': estimCDF, 'upper': upper}
            # tmp = pd.DataFrame(dat, columns = ['points', 'lower', 'cdf', 'upper'])
            tmp = PredictionsIdr(ecdf=estimCDF, points=thresholds[sel], lower=lower, upper=upper)
            order_indices.append(i)
            preds.append(tmp)
        preds_rearranged = [preds[k] for k in np.argsort(order_indices)]
        return PredictionResult(
            predictions=preds_rearranged, thresholds=np.where(incomparables)
        )

    def _get_climatological_forecast(self) -> PredictionsIdr:
        """
        Compute the climatological forecast when data is incomparable.

        Returns:
            predictions_idr: The climatological forecast based on empirical CDF.
        """
        edf = np.round(ecdf_formal(self.thresholds, self.y.explode()), 3)
        sel = edf > 0
        edf = edf[sel]
        points = self.thresholds[sel]
        upper = edf[-1]
        return PredictionsIdr(ecdf=edf, points=points, lower=edf, upper=upper)

    @staticmethod
    def _calculate_cdf_bounds(
        i: int,
        cdf: np.ndarray,
        smaller: List[np.ndarray],
        greater: List[np.ndarray],
        digits: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the CDF bounds for the prediction.

        Args:
            i (int): Index of the data point.
            cdf (np.ndarray): CDF values.
            smaller (List[np.ndarray]): List of smaller CDF indices.
            greater (List[np.ndarray]): List of greater CDF indices.
            digits (int): Rounding precision.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Lower bound, upper bound, and estimated CDF.
        """
        # Check if there are any smaller elements at index i
        if len(smaller[i]) > 0:
            # Compute the lower bound as the max of the 'smaller' indices
            lower_bound = np.max(cdf[smaller[i]])
        else:
            # No smaller elements, lower bound is 0
            lower_bound = 0

        # Check if there are any greater elements at index i
        if len(greater[i]) > 0:
            # Compute the upper bound as the min of the 'greater' indices
            upper_bound = np.min(cdf[greater[i]])
        else:
            # No greater elements, upper bound is 1
            upper_bound = 1

        # Round both bounds to the specified precision
        lower_bound = np.round(lower_bound, digits)
        upper_bound = np.round(upper_bound, digits)

        # Calculate the estimated CDF as the average of the lower and upper bounds
        estimated_cdf = np.round((lower_bound + upper_bound) / 2, digits)

        return lower_bound, upper_bound, estimated_cdf


def isotonic_distributional_regression(
    observations: np.ndarray,
    covariates: pd.DataFrame,
    grouping: Optional[Dict[str, int]] = None,
    orderings: Dict[str, str] = None,
    verbose: bool = False,
    max_iterations: int = 10000,
    tolerance_relative: float = 1e-5,
    tolerance_absolute: float = 1e-5,
) -> Idr:
    """
    Perform isotonic distributional regression (IDR).

    This function estimates a conditional cumulative distribution function (CDF)
    for a given set of observations and covariates.

    Args:
        observations (np.ndarray): 1-D array of observed values.
        covariates (pd.DataFrame): DataFrame of covariates.
        grouping (Optional[Dict[str, int]]): Dictionary assigning column names of
            `covariates` to groups.
        orderings (Dict[str, str]): Dictionary defining the ordering type for each group.
            Acceptable values are 'comp', 'icx', and 'sd'.
        verbose (bool): Whether to print solver messages (default: False).
        max_iterations (int): Maximum iterations for the optimization solver.
        tolerance_relative (float): Relative tolerance for the optimization solver.
        tolerance_absolute (float): Absolute tolerance for the optimization solver.

    Returns:
        Idr: An instance containing results of the IDR estimation.

    Raises:
        ValueError: If input validation fails for `covariates`, `observations`,
            or configuration parameters.
    """
    if orderings is None:
        orderings = {"1": "comp"}

    if not isinstance(covariates, pd.DataFrame):
        raise ValueError("`covariates` must be a pandas DataFrame")

    if grouping is None:
        grouping = {column: 1 for column in covariates.columns}

    observations = np.asarray(observations)
    if observations.ndim != 1:
        raise ValueError("`observations` must be a 1-D array")

    if covariates.shape[0] <= 1:
        raise ValueError("`covariates` must have more than one row")

    if np.isnan(observations).any():
        raise ValueError("`observations` contains NaN values")

    if covariates.isnull().values.any():
        raise ValueError("`covariates` contains NaN values")

    if len(observations) != covariates.shape[0]:
        raise ValueError(
            "The length of `observations` must match the number of rows in `covariates`"
        )

    if not all(order in {"comp", "icx", "sd"} for order in orderings.values()):
        raise ValueError("Orderings must be one of {'comp', 'icx', 'sd'}")

    if not set(covariates.columns).issubset(grouping.keys()):
        raise ValueError("All variables in `covariates` must be present in `grouping`")

    # Process thresholds
    unique_thresholds = np.sort(np.unique(observations))
    num_thresholds = len(unique_thresholds)

    if num_thresholds <= 1:
        raise ValueError("`observations` must contain more than one distinct value")

    # Prepare data for IDR
    processed_data = prepare_data_for_idr(covariates, grouping, orderings)
    num_variables = processed_data.shape[1]

    column_names = list(processed_data.columns)

    processed_data["observations"] = observations
    processed_data["indices"] = np.arange(len(observations))
    # Group observations

    grouped_data = (
        processed_data.groupby(column_names).agg({"observations": list, "indices": list})
        .sort_values(by=column_names[::-1]).reset_index()
    )
    grouped_observations = grouped_data["observations"]
    grouped_indices = grouped_data["indices"]
    group_weights = np.array([len(idx) for idx in grouped_indices])
    num_groups = grouped_data.shape[0]

    # Isotonic regression
    if num_variables == 1:
        position_vector = np.repeat(
            np.arange(len(grouped_indices)),
            [len(sublist) for sublist in grouped_indices],
        )
        cdf_vector = isocdf_seq(
            group_weights,
            np.ones(len(observations)),
            np.sort(observations),
            position_vector,
            unique_thresholds,
        )
        cdf_result = np.reshape(cdf_vector, (num_groups, num_thresholds), order="F")
        constraints = None
    else:
        constraints = componentwise_partial_order(grouped_data)
        cumulative_distribution = np.zeros((num_groups, num_thresholds - 1))
        reduced_transitivity = tr_reduc(constraints[0], num_groups)
        constraint_weights = sparse.csc_matrix(
            (group_weights, (np.arange(num_groups), np.arange(num_groups)))
        )
        constraint_lower_bound = np.zeros(reduced_transitivity.shape[1])

        reduced_transitivity = sparse.csc_matrix((np.repeat([1, -1], reduced_transitivity.shape[1]), (np.tile(np.arange(reduced_transitivity.shape[1]), 2), reduced_transitivity.flatten())),
                          shape=(reduced_transitivity.shape[1], num_groups))

        for threshold_index in range(num_thresholds - 1):
            optimization_target = -group_weights * grouped_observations.apply(
                lambda group: np.mean(
                    np.array(group) <= unique_thresholds[threshold_index]
                )
            )
            solver = osqp.OSQP()
            solver.setup(
                P=constraint_weights,
                q=optimization_target.to_numpy(),
                A=reduced_transitivity,
                l=constraint_lower_bound,
                verbose=verbose,
                max_iter=max_iterations,
                eps_rel=tolerance_relative,
                eps_abs=tolerance_absolute,
            )
            solution = solver.solve()
            cumulative_distribution[:, threshold_index] = np.clip(solution.x, 0, 1)

        cumulative_distribution = pava_correct(cumulative_distribution)
        cdf_result = np.ones((num_groups, num_thresholds))
        cdf_result[:, :-1] = cumulative_distribution

    # Construct and return the Idr object
    return Idr(
        cdf_result,
        unique_thresholds,
        grouped_indices,
        grouped_data[column_names],
        grouped_observations,
        grouping,
        orderings,
        constraints,
    )


def ecdf_formal(x, data):
    """
    Compute the values of the formal ECDF generated from `data` at x.
    I.e., if F is the ECDF, return F(x).

    Parameters:
        x : int, float, or array_like
            Positions at which the formal ECDF is to be evaluated.
        data : array_like
            One-dimensional array of data to use to generate the ECDF.

    Returns:
    output : float or ndarray
        Value of the ECDF at `x`.
    """
    # Remember if the input was scalar
    if np.isscalar(x):
        return_scalar = True
    else:
        return_scalar = False

    # If x has any nans, raise a RuntimeError
    if np.isnan(x).any():
        raise RuntimeError("Input cannot have NaNs.")

    # Convert x to array
    x = _convert_data(x, inf_ok=True)

    # Convert data to sorted NumPy array with no nan's
    data = _convert_data(data, inf_ok=True)

    # Compute formal ECDF value
    out = _ecdf_formal(x, np.sort(data))

    if return_scalar:
        return out[0]
    return out


def _ecdf_formal(x, data):
    """
    Compute the values of the formal ECDF generated from `data` at x.
    I.e., if F is the ECDF, return F(x).

    Parameters:
        x : array_like
            Positions at which the formal ECDF is to be evaluated.
        data : array_like
            *Sorted* data set to use to generate the ECDF.

    Returns:
        output : float or ndarray
            Value of the ECDF at `x`.
    """
    output = np.empty_like(x)

    for i, x_val in enumerate(x):
        j = 0
        while j < len(data) and x_val >= data[j]:
            j += 1

        output[i] = j

    return output / len(data)


def _convert_data(data, inf_ok=False, min_len=1):
    """
    Convert inputted 1D data set into NumPy array of floats.
    All nan's are dropped.

    Parameters:
        data : int, float, or array_like
            Input data, to be converted.
        inf_ok : bool, default False
            If True, np.inf values are allowed in the arrays.
        min_len : int, default 1
            Minimum length of array.

    Returns:
    output : ndarray
        `data` as a one-dimensional NumPy array, dtype float.
    """
    # If it's scalar, convert to array
    if np.isscalar(data):
        data = np.array([data], dtype=np.float)

    # Convert data to NumPy array
    data = np.array(data, dtype=np.float)

    # Make sure it is 1D
    if len(data.shape) != 1:
        raise RuntimeError("Input must be a 1D array or Pandas series.")

    # Remove NaNs
    data = data[~np.isnan(data)]

    # Check for infinite entries
    if not inf_ok and np.isinf(data).any():
        raise RuntimeError("All entries must be finite.")

    # Check to minimal length
    if len(data) < min_len:
        raise RuntimeError(
            "Array must have at least {0:d} non-NaN entries.".format(min_len)
        )

    return data
