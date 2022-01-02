import statsmodels.api as sm

import numpy as np
import pandas as pd
import orbit
from orbit.models.lgt import LGTMAP, LGTAggregated, LGTFull
from orbit.diagnostics.plot import plot_predicted_data
from orbit.diagnostics.plot import plot_predicted_components
from orbit.utils.dataset import load_iclaims
from orbit.utils.plot import get_orbit_style


def lgt_prediction(time_series, meta_dictionary, visualise=False):

    seasonality = meta_dictionary["s"]
    freq = meta_dictionary["f"]
    forecast_horizon = meta_dictionary["h"]

    time_series_c = pd.DataFrame(time_series).reset_index()
    time_series_c.rename(columns={time_series_c.columns[1]: "values"}, inplace=True)
    time_series_c.drop(columns="index", inplace=True)

    times = pd.date_range("1-10-01", periods=len(time_series_c), freq="H")
    times_forecast = pd.date_range(times[-1], periods=forecast_horizon, freq="H")

    time_series_c["times"] = times
    time_series_c["values"] = np.array(time_series_c["values"], dtype=float)

    prediction_df = pd.DataFrame([])
    prediction_df["times"] = times_forecast
    prediction_df["values"] = [0] * len(times_forecast)

    date_col = "times"
    response_col = "values"

    lgt = LGTMAP(response_col="values", date_col="times", seasonality=seasonality,)

    lgt.fit(df=time_series_c)
    predicted_df = lgt.predict(df=prediction_df)

    if visualise:
        _ = plot_predicted_data(
            training_actual_df=train_df,
            predicted_df=predicted_df,
            date_col=date_col,
            actual_col=response_col,
            test_actual_df=test_df,
            title="Prediction with LGTMAP Model",
        )
    return predicted_df["prediction"]
