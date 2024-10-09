"""
easuq model
Author: Daniel Lassahn based upon work of E.Walz (2024)
"""
from typing import Union, Type
import numpy as np
import pandas as pd
from utils import idr


def easyuq_conformal_prediction(
    historic_forecast: pd.DataFrame,
    historic_observation: pd.DataFrame,
    current_forecast: pd.DataFrame,
    lead_time_col: str,
    n_ensemble_members: int = 100
) -> pd.DataFrame:
    """
    Performs conformal prediction using the EasyUQ model on the provided DataFrames.

    Args:
        historic_forecast (pd.DataFrame): DataFrame containing the historic forecast with a DateTime index.
        historic_observation (pd.DataFrame): DataFrame containing the historic observations (truth) with a DateTime index.
        current_forecast (pd.DataFrame): DataFrame containing the current forecast with a DateTime index.
        lead_time_col (str): Column name for the lead time in each DataFrame.
        n_ensemble_members (int, optional): Number of ensemble members to generate. Defaults to 100.

    Returns:
        pd.DataFrame: DataFrame containing the expected value, standard deviation, CRPS, and ensemble predictions for each lead time.
    """
    def get_expected_value(prediction: Type[idr], q_steps: float = 0.0001) -> float:
        quantiles = np.arange(q_steps, 1, q_steps)
        quantile_pred = prediction.qpred(quantiles=quantiles)
        expected_value = (quantile_pred * q_steps).sum(axis=1)
        return expected_value

    def get_standard_deviation(prediction: Type[idr], exp: np.ndarray, q_steps: float = 0.0001) -> float:
        quantiles = np.arange(q_steps, 1, q_steps)
        quantile_pred = prediction.qpred(quantiles=quantiles)
        return np.sqrt(np.sum(np.power(quantile_pred - np.expand_dims(exp, axis=1), 2) * q_steps, axis=1))

    def generate_ensemble_members(exp: Union[float, np.ndarray, pd.Series], sd: float, n_members: int) -> np.ndarray:
        """ generate ensemble members """
        ensemble_members = np.random.normal(loc=exp, scale=sd, size=(n_members, len(exp)))
        return ensemble_members

    results = []

    lead_times = historic_forecast[lead_time_col].unique()

    for lead_time in lead_times:
        # Filter data for the current lead time
        hist_forecast_lt = historic_forecast[historic_forecast[lead_time_col] == lead_time]
        hist_observation_lt = historic_observation[historic_observation[lead_time_col] == lead_time]
        current_forecast_lt = current_forecast[current_forecast[lead_time_col] == lead_time]

        # Ensure the datetime index aligns
        y_train = hist_observation_lt.drop(columns=[lead_time_col]).values
        x_train = hist_forecast_lt.drop(columns=[lead_time_col]).values
        x_test = current_forecast_lt.drop(columns=[lead_time_col]).values

        # Train the EasyUQ model
        fitted_idr = idr(y_train, pd.DataFrame(x_train))
        preds_test = fitted_idr.predict(pd.DataFrame(x_test))

        # Get expected value and standard deviation
        exp = get_expected_value(preds_test)
        sd = get_standard_deviation(preds_test, exp)

        # Compute CRPS
        crps = preds_test.crps(y_train)

        # Generate ensemble members
        ensemble_members = generate_ensemble_members(exp, sd, n_ensemble_members)

        # Store results
        result_dict = {
            'lead_time': lead_time,
            'expected_value': exp,
            'standard_deviation': sd,
            'crps': crps,
            'ensemble_members': ensemble_members
        }
        results.append(result_dict)

    # Convert results into a DataFrame
    return pd.DataFrame(results)
