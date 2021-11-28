import statsmodels.api as sm

def ols_prediction(time_series, meta_dictionary):
    
    forecast_horizon = meta_dictionary["h"]
    end_index = len(time_series)
    
    regressor = sm.add_constant([i for i in range(end_index)])

    model = sm.regression.linear_model.OLS(time_series, regressor).fit()
    
    Xnew = sm.add_constant(list(range(end_index, end_index + forecast_horizon)))

    return model.predict(Xnew)
