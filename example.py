""" example on """
from datetime import datetime

import pandas as pd
import numpy as np
from easyuq.model import easyuq_conformal_prediction


# We use GHI observation from WMO weather station 10471 in Leipzig, that can be download here: https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
#         f"climate/10_minutes/solar/recent/

forecast_data = pd.read_pickle('exp_data/ghi_test_forecast.pkl')
observation = pd.read_pickle('exp_data/ghi_test_obs.pkl')

# forecast_data.head()
#                                 global_horizontal_irradiance
# dt_calc    dt_fore
# 2023-07-01 2023-07-01 00:00:00                           0.0
#            2023-07-01 00:10:00                           0.0
#            2023-07-01 00:20:00                           0.0
#            2023-07-01 00:30:00                           0.0
#            2023-07-01 00:40:00                           0.0



result = easyuq_conformal_prediction(forecast_data, observation, datetime(2024, 10, 31))

from IPython import embed;embed()

q_50 = result.ensemble_members.apply(lambda x: np.percentile(np.array(x).ravel(), 0.5))

q_50 = []
for ensembles in result.ensemble_members:
    q_50.append(np.percentile(np.array(ensembles).ravel(), 0.5))

from matplotlib import pyplot as plt