import torch

from ..utils.scalers import scalers
 

def feature_extractor(df, manual_auto_tp_toggle, normalization, n_models):

    batch_size = df.shape[0]

    # Get forecasts
    forecasts = df.loc[:, "auto_arima_forec_0":"quant_99_reg_47"]

    # Get feature inputs
    if manual_auto_tp_toggle == "":
        raise Exception(
            "Manual_or_auto_toggle needs to cointain either m or a for input to be non-empty")

    inputs_start = "x_acf1" if "m" in manual_auto_tp_toggle.lower() else "lstm_0"
    inputs_end = "lstm_31" if "a" in manual_auto_tp_toggle.lower() else "series_length"

    inputs = df.loc[:, inputs_start:inputs_end]

    inputs_cat = df.loc[:, ['type', 'period']].astype("category")
    emb_dims = [(x, min(x // 2, 50))
                for x in map(lambda y: len(inputs_cat[y].cat.categories), inputs_cat)]

    for col in inputs_cat:
        inputs_cat[col] = inputs_cat[col].cat.codes

    inputs_cat = torch.as_tensor(inputs_cat.to_numpy(), dtype=torch.long)

    scaler = scalers[normalization]
    inputs_normalized = scaler.fit_transform(inputs.to_numpy())

    # Get actuals
    actuals = df.loc[:, "actual_0":"actual_47"].to_numpy()
    forecasts = forecasts.to_numpy().reshape(
        (batch_size, n_models, 48)).swapaxes(1, 2)

    return (inputs_cat, emb_dims), inputs_normalized, forecasts, actuals
