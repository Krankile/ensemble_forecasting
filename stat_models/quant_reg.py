from statsmodels.regression.quantile_regression  import QuantReg 
import numpy as np
import statsmodels.api as sm

def quant_reg_prediction(time_series, meta_variables, q=.95, visualise=False):
    """Take in meta_variables "h" and q_high_low 1=>q=.99, 0 => q=0.01"""
    
    regressor = sm.add_constant(np.array(range(0,len(time_series))))
    time_series_constant = np.array(time_series, dtype='float')

    reg = QuantReg(time_series_constant, regressor).fit(q=q)
    
    future_t = sm.add_constant(list(range(len(time_series), len(time_series) + meta_variables["h"])))
    
    return reg.predict(future_t)