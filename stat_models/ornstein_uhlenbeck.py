from ou_noise import ou
import numpy as np


def ornstein_uhlenbeck_prediction(time_series, meta_dictionary, visualise=False):
    """dX = -A(X-mean)dt+v*dZ"""
    forecast_horizon = meta_dictionary["h"]
    x = np.array(time_series, dtype="float")
    t_historic = np.arange(0, len(x), 1)
    t_horizon = np.arange(0, forecast_horizon, 1)

    mean_rev_speed, mean_rev_level, vola = ou.mle(t_historic, x)

    adjuster = max(x.max() - x.min(), 1)

    return ou.path(x[-1], t_horizon, mean_rev_speed, mean_rev_level, vola / adjuster)
