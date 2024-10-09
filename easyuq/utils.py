"""
utility function for easyuq model
Author: Daniel Lassahn based upon work of E.Walz (2024)
"""
import pandas as pd
import bisect
from typing import List, Optional, Tuple, Dict, Type
from scipy import sparse
import osqp
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict
import dc_stat_think as dcst
from _isodisreg import isocdf_seq, pavaCorrect_c


class PredictionResult:
    def __init__(self, predictions, thresholds):
        self.predictions = predictions
        self.thresholds = thresholds

    def qpred(self, quantiles):
        """
        Generate quantile predictions for specified quantile levels.

        Args:
            quantiles (array-like): Quantile levels to predict (between 0 and 1).

        Returns:
            Quantile predictions for the requested quantiles.
        """
        quantiles = np.asarray(quantiles)

        if np.min(quantiles) < 0 or np.max(quantiles) > 1:
            raise ValueError("Quantiles must be a numeric vector with entries in [0,1].")

        def interpolate_single_prediction(data):
            """
            Interpolates for a single prediction using the adaptive method.
            """
            # Concatenate ECDF and the maximum value of ECDF and corresponding points
            x_vals = np.hstack([data.ecdf, np.max(data.ecdf)])
            y_vals = np.hstack([data.points, data.points[-1]])

            # Use adaptive interpolation based on thresholds (quantiles)
            return interp1d_adapt_q(prediction=x_vals, observation=y_vals, thresholds=quantiles)

        # Apply interpolation to all predictions and return the stacked result
        return np.vstack([interpolate_single_prediction(pred) for pred in self.predictions]).squeeze()

    def crps(self, observations: np.ndarray):
        """
        Compute the Continuous Ranked Probability Score (CRPS) for the predicted CDF and the true values.

        Args:
            observations (np.ndarray): The true observations.

        Returns:
            CRPS values for each data point.
        """
        # Implement CRPS calculation
        crps_scores = np.zeros(len(observations))
        for i in range(len(observations)):
            # Example CRPS calculation based on empirical CDF and true values
            crps_scores[i] = np.mean(np.abs(self.predictions[i] - observations[i]))

        return crps_scores

def interp1d_adapt_q(prediction: np.ndarray, observation: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
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
        ix2 = np.where(thresholds <= min_points)[0]  # Thresholds smaller or equal to min_points

        if ix1.size > 0:
            inter_vals[ix1] = interp1d(prediction, observation, kind='previous', fill_value="extrapolate")(thresholds[ix1])
        if ix2.size > 0:
            inter_vals[ix2] = interp1d(prediction, observation, kind='next', fill_value="extrapolate")(thresholds[ix2])

        return inter_vals
    else:
        # If all thresholds are less than or equal to min_points, use 'next' interpolation
        return interp1d(prediction, observation, kind='next', fill_value="extrapolate")(thresholds)


def prepare_data_for_idr(X: pd.DataFrame, groups: Dict[str, str], orders: Dict[str, str]) -> pd.DataFrame:
    """
    Prepare data for IDR modeling with given orders

    Parameters
    ----------
    X : pd.DataFrame of covariates
    groups : dictionary
        assigning column names of X to groups
    orders : dictionary
        assigning groups to type of ordering


    Returns
    -------
    X : pd.DataFrame

    """
    res = defaultdict(list)
    for key, val in sorted(groups.items()):
        res[val].append(key)
    for key, val in res.items():
        if len(val) > 1:
            if orders[str(int(key))] == "comp":
                continue
            tmp = -np.sort(-X[val], axis=1)
            if orders[str(int(key))] == "sd":
                X[val] = tmp
            else:
                X[val] = np.cumsum(tmp, axis=1)
    return X


def comp_ord(X):
    """Componentwise partial order on rows of array x.

    Compares the columns x[j, j] of a matrix x in the componentwise
    order.


    Parameters
    ----------
    x : np.array
        Two-dimensional array with at least two columns.
    Returns
    -------
    paths : np.array
        Two-column array, containing in the first coolumns the indices
        of the rows of x that are smaller in the componentwise order,
        and in the second column the indices of the corresponding
        greater rows.
    col_order : np.array
        Array of the same dimension of x, containing in each column the
        order of the column in x.
    """
    X = np.asarray(X)
    if X.ndim != 2 | X.shape[0] < 2:
        raise ValueError("X should have at least two rows")
    Xt = X.transpose()
    m = Xt.shape[1]
    d = Xt.shape[0]
    colOrder = np.argsort(Xt, axis=1)
    ranks = np.apply_along_axis(scipy.stats.rankdata, 1, Xt, method='max')
    smaller = []
    greater = []
    for k in range(m):
        nonzeros = np.full((m), False)
        nonzeros[colOrder[0, 0:ranks[0, k]]] = True

        for l in range(1, d):
            if ranks[l, k] < m:
                nonzeros[colOrder[l, ranks[l, k]:m]] = False
        nonzeros = np.where(nonzeros)[0]
        n_nonzeros = nonzeros.shape[0]
        smaller.extend(nonzeros)
        greater.extend([k] * n_nonzeros)
    paths = np.vstack([smaller, greater])
    return paths, colOrder.transpose()


def tr_reduc(paths, n):
    """Transitive reduction of path matrix.

    Transforms transitive reduction of a directed acyclic graph.

    Parameters
    ----------
    x : np.array
        Two-dimensional array containing the indices of the smaller
        vertices in the first row and the indices of the
        greater vertices in the second row.
    Returns
    -------

    """
    edges = np.full((n, n), False)
    edges[paths[0], paths[1]] = True
    np.fill_diagonal(edges, False)
    for k in range(n):
        edges[np.ix_(edges[:, k], edges[k])] = False
    edges = np.array(np.nonzero(edges))
    return edges


class PredictionsIdr:
    def __init__(self, ecdf: np.ndarray, points: np.ndarray, lower: np.ndarray, upper: np.ndarray):
        """
        Args:
            ecdf (np.ndarray): Empirical cumulative distribution function values.
            points (np.ndarray): Array of points where the ECDF is evaluated.
            lower (np.ndarray): Lower bound of prediction intervals.
            upper (np.ndarray): Upper bound of prediction intervals.
        """
        self.ecdf = np.asarray(ecdf)
        self.points = np.asarray(points)
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)


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
        ranks_right[:, j] = np.searchsorted(a=X[:, j], v=x[:, j], side="right", sorter=col_order[:, j])

    x_geq_X = np.full((n, nx), False)
    x_leq_X = np.full((n, nx), True)
    for i in range(nx):
        if ranks_right[i, 0] > 0:
            x_geq_X[col_order[0:int(ranks_right[i, 0]), 0], i] = True
        if ranks_left[i, 0] > 0:
            x_leq_X[col_order[0:int(ranks_left[i, 0]), 0], i] = False
        for j in range(1, k):
            if ranks_right[i, j] < n:
                x_geq_X[col_order[int(ranks_right[i, j]):n, j], i] = False
            if ranks_left[i, j] > 0:
                x_leq_X[col_order[0:int(ranks_left[i, j]), j], i] = False
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
    def __init__(self, ecdf: np.ndarray, thresholds: np.ndarray, indices: List[List[int]],
                 X: pd.DataFrame, y: pd.Series, groups: Dict, orders: Dict, constraints: Optional[np.ndarray] = None):
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

    def _predict_on_training_data(self, cdf: np.ndarray, thresholds: np.ndarray, digits: int):
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
            tmp = PredictionsIdr(ecdf=edf[sel], points=thresholds[sel], lower=np.array([]), upper=np.array([]))
            for j in idx_group:
                order_indices.append(j)
                preds.append(tmp)

        preds_rearranged = [preds[k] for k in np.argsort(order_indices)]
        return PredictionResult(predictions=preds_rearranged, thresholds=None)

    def _predict_on_new_data(self, data: pd.DataFrame, cdf: np.ndarray, thresholds: np.ndarray, digits: int):
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

        X = self.X
        data = prepare_data_for_idr(data[X.columns], groups=self.groups, orders=self.orders)
        nVar = data.shape[1]

        if nVar == 1:
            return self._predict_single_variable(data, cdf, thresholds, digits)
        else:
            return self._predict_multivariate(data, cdf, thresholds, digits)

    def _predict_single_variable(self, data: pd.DataFrame, cdf: np.ndarray, thresholds: np.ndarray, digits: int) -> PredictionResult:
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
        wg = np.interp(x_test, X_train, np.arange(1, X_train.shape[0] + 1), left=1, right=X_train.shape[0]) \
             - np.arange(1, X_train.shape[0] + 1)[smaller]
        greater = smaller + (wg > 0).astype(int)
        ws = 1 - wg

        preds = list(map(self._interpolate_predictions, cdf[greater.astype(int)], cdf[smaller.astype(int)], ws, wg))
        return PredictionResult(predictions=preds, thresholds=None)

    def _interpolate_predictions(self, l: np.ndarray, u: np.ndarray, ws: float, wg: float) -> 'PredictionsIdr':
        """
        Helper function to interpolate between predictions for new data.

        Args:
            l (np.ndarray): Lower bound CDF.
            u (np.ndarray): Upper bound CDF.
            ws (float): Weight for the smaller value.
            wg (float): Weight for the greater value.

        Returns:
            predictions_idr: Interpolated predictions for the new data.
        """
        ls = np.insert(l[:-1], 0, 0)
        us = np.insert(u[:-1], 0, 0)
        ind = (ls < l) | (us < u)
        l = l[ind]
        u = u[ind]
        cdf = np.round(l * wg + u * ws, 3)
        return PredictionsIdr(ecdf=cdf, points=self.thresholds[ind], lower=l, upper=u)

    def _predict_multivariate(self, data: pd.DataFrame, cdf: np.ndarray, thresholds: np.ndarray, digits: int):
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
        nPoints = neighbor_points(data, self.X, order_X=self.constraints)
        smaller, greater = nPoints

        incomparables = np.array([len(sm) == 0 and len(gr) == 0 for sm, gr in zip(smaller, greater)])
        preds = []
        order_indices = []

        for i, is_incomparable in enumerate(incomparables):
            if is_incomparable:
                preds.append(self._get_climatological_forecast())
                order_indices.append(i)
            else:
                lower, upper, estimCDF = self._calculate_cdf_bounds(i, cdf, smaller, greater, digits)
                preds.append(PredictionsIdr(ecdf=estimCDF, points=thresholds, lower=lower, upper=upper))
                order_indices.append(i)

        preds_rearranged = [preds[k] for k in np.argsort(order_indices)]
        return PredictionResult(predictions=preds_rearranged, thresholds=np.where(incomparables))

    def _get_climatological_forecast(self) -> PredictionsIdr:
        """
        Compute the climatological forecast when data is incomparable.

        Returns:
            predictions_idr: The climatological forecast based on empirical CDF.
        """
        edf = np.round(dcst.ecdf_formal(self.thresholds, self.y.explode()), 3)
        sel = edf > 0
        edf = edf[sel]
        points = self.thresholds[sel]
        upper = edf[-1]
        return PredictionsIdr(ecdf=edf, points=points, lower=edf, upper=upper)

    @staticmethod
    def _calculate_cdf_bounds(i: int, cdf: np.ndarray, smaller: List[np.ndarray], greater: List[np.ndarray], digits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def idr(y, X, groups=None, orders=dict({"1": "comp"}), verbose=False, max_iter=10000, eps_rel=0.00001, eps_abs=0.00001,
        progress: bool = True) -> Idr:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")

    if groups is None:
        groups = dict(zip(X.columns, np.ones(X.shape[1])))

    y = np.asarray(y)

    if y.ndim > 1:
        raise ValueError('idr only handles 1-D arrays of observations')

    if X.shape[0] <= 1:
        raise ValueError('X must have more than 1 row')

    if np.isnan(np.sum(y)):
        raise ValueError("y contains nan values")

    if X.isnull().values.any():
        raise ValueError("X contains nan values")

    if y.size != X.shape[0]:
        raise ValueError("length of y must match number of rows in X")

    if not all(item in ['comp', 'icx', 'sd'] for item in orders.values()):
        raise ValueError("orders must be in comp, icx, sd")

    M = all(elem in groups.keys() for elem in X.columns)
    if not M:
        raise ValueError("some variables must be used in groups and in X")

    thresholds = np.sort(np.unique(y))
    nThr = len(thresholds)

    if nThr == 1:
        raise ValueError("y must contain more than 1 distinct value")

    Xp = prepare_data_for_idr(X, groups, orders)
    nVar = Xp.shape[1]
    Xp['y'] = y
    Xp['ind'] = np.arange(len(y))
    oldNames = Xp.columns

    X_grouped = Xp.groupby(list(oldNames)).agg({'y': list, 'ind': list}).reset_index()
    cpY = X_grouped["y"]
    indices = X_grouped["ind"]

    weights = np.array([len(idx) for idx in indices])
    N = X_grouped.shape[0]

    if nVar == 1:
        constr = None
        posY = np.repeat(np.arange(len(indices)), [len(sublist) for sublist in indices])
        cdf_vec = isocdf_seq(weights, np.ones(y.shape[0]), np.sort(y), posY, thresholds)
        cdf1 = np.reshape(cdf_vec, (N, nThr), order="F")
    else:
        constr = comp_ord(Xp)
        cdf = np.zeros((N, nThr - 1))
        A = tr_reduc(constr[0], N)
        nConstr = A.shape[1]
        l = np.zeros(nConstr)

        P = sparse.csc_matrix((weights, (np.arange(N), np.arange(N))))
        for i in range(nThr - 1):
            q = -weights * np.array(cpY.apply(lambda x: np.mean(np.array(x) <= thresholds[i])))
            m = osqp.OSQP()
            m.setup(P=P, q=q, A=A, l=l, verbose=verbose, max_iter=max_iter, eps_rel=eps_rel, eps_abs=eps_abs)
            sol = m.solve()
            cdf[:, i] = np.maximum(0, np.minimum(1, sol.x))

        cdf = pavaCorrect_c(cdf)
        cdf1 = np.ones((N, nThr))
        cdf1[:, :-1] = cdf

    return Idr(cdf1, thresholds, indices, X_grouped[oldNames], cpY, groups, orders, constr)
