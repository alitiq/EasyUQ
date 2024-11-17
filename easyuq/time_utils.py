""" time utils functions """
from typing import Tuple
import numpy as np
import pandas as pd

def align_observations_with_multi_forecast_index(
    forecast_data: pd.DataFrame,
    observation_data: pd.DataFrame,
    observation_index_name: str = "dt",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    this function only works for data with one target and

    takes both predictors and target values and derives intersection
    of both to create two matching dataframes by using dt_fore

    forecast_data contains MultiIndex with dt_calc, dt_fore
        dt_calc: INIT/calculation run timestamp
        dt_fore: leading forecast timestamp

    """
    dt_fores = forecast_data.index.get_level_values("dt_fore")
    target_data = observation_data.reindex(np.sort(np.unique(dt_fores)))
    target_data = target_data.loc[dt_fores, :]
    target_data.index.names = [observation_index_name]
    target_data_mask = target_data.isna().values
    forecast_data, target_data = (
        forecast_data[~target_data_mask],
        target_data[~target_data_mask],
    )
    forecast_data_mask = forecast_data.isna().any(axis=1).values
    return forecast_data[~forecast_data_mask], target_data[~forecast_data_mask]

