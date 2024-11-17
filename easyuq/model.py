"""
easuq model
Author: Daniel Lassahn based upon work of E.Walz (2024)
"""
from datetime import datetime
from typing import Union, Type
import numpy as np
import pandas as pd
from easyuq.utils import isotonic_distributional_regression
from easyuq.time_utils import align_observations_with_multi_forecast_index

def easyuq_conformal_prediction(
    forecast_data: pd.DataFrame,
    observation: pd.DataFrame,
    current_forecast_datetime: Union[datetime, pd.Timestamp],
    n_ensemble_members: int = 100,
) -> pd.DataFrame:
    """
    Performs conformal prediction using the EasyUQ model on the provided DataFrames.

    Note: The DataFrame requires a MultiIndex with dt_calc and dt_fore

    Args:
        forecast_data (pd.DataFrame): DataFrame containing the historic forecast with a DateTime index.
        observation (pd.DataFrame): DataFrame containing the historic observations (truth) with a DateTime
            index.
        current_forecast_datetime (datetime, pd.Timestamp): Timestamp contains the current forecast with a DateTime index.
        n_ensemble_members (int, optional): Number of ensemble members to generate. Defaults to 100.

    Returns:
        pd.DataFrame: DataFrame containing the expected value, standard deviation, CRPS, and ensemble predictions
            for each lead time.
    """
    def get_expected_value(
        prediction: Type[isotonic_distributional_regression], q_steps: float = 0.001
    ) -> float:
        quantiles = np.arange(q_steps, 1, q_steps)
        quantile_pred = prediction.qpred(quantiles=quantiles)
        if len(quantile_pred.shape) <= 1:
            expected_value = (np.expand_dims(quantile_pred, axis=1) * q_steps).sum(axis=1)
        else:
            expected_value = (quantile_pred * q_steps).sum(axis=1)

        return expected_value

    def get_standard_deviation(
        prediction: Type[isotonic_distributional_regression],
        exp: np.ndarray,
        q_steps: float = 0.0001,
    ) -> float:
        quantiles = np.arange(q_steps, 1, q_steps)
        quantile_pred = prediction.qpred(quantiles=quantiles)
        return np.sqrt(
            np.sum(
                np.power(quantile_pred - np.expand_dims(exp, axis=1), 2) * q_steps,
                axis=1,
            )
        )

    def generate_ensemble_members(
        exp: Union[float, np.ndarray, pd.Series], sd: float, n_members: int
    ) -> np.ndarray:
        """generate ensemble members"""
        ensemble_members = np.random.normal(
            loc=exp, scale=sd, size=(n_members, len(exp))
        )
        return ensemble_members

    forecast_data["lead_time"] = forecast_data.index.get_level_values("dt_fore") - forecast_data.index.get_level_values("dt_calc")
    forecast_of_interest = forecast_data[forecast_data.index.get_level_values("dt_calc") == current_forecast_datetime]

    observation = observation.loc[:current_forecast_datetime, :]
    forecast_data = forecast_data[forecast_data.index.get_level_values("dt_calc") < current_forecast_datetime]
    forecast_data, observation = align_observations_with_multi_forecast_index(forecast_data, observation)

    results = []

    for lead_time in forecast_of_interest.lead_time:
        # Filter data for the current lead time
        hist_forecast_lt = forecast_data[
            forecast_data["lead_time"] == lead_time
        ]
        hist_observation_lt = observation[
            (forecast_data["lead_time"] == lead_time).values
        ]
        current_forecast_lt = forecast_of_interest[
            forecast_of_interest["lead_time"] == lead_time
        ]

        y_train = hist_observation_lt.values.ravel()
        x_train = hist_forecast_lt.values
        x_test = current_forecast_lt.values

        if len(np.unique(y_train)) <= 12:  # random selected number of at least 10 different values
            continue

        # Train the EasyUQ model
        fitted_idr = isotonic_distributional_regression(y_train, pd.DataFrame(x_train))
        preds_test = fitted_idr.predict(pd.DataFrame(x_test))
        preds_train = fitted_idr.predict(pd.DataFrame(x_train))

        # Get expected value and standard deviation
        exp = get_expected_value(preds_test)
        sd = get_standard_deviation(preds_test, exp)


        # Compute CRPS
        crps = preds_train.crps(y_train)
        # Generate ensemble members
        ensemble_members = generate_ensemble_members(exp, sd, n_ensemble_members)

        # Store results
        result_dict = {
            "lead_time": lead_time,
            "expected_value": exp,
            "standard_deviation": sd,
            "crps": crps,
            "ensemble_members": ensemble_members,
        }
        results.append(result_dict)

    # Convert results into a DataFrame
    return pd.DataFrame(results)
