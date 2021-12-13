from os import cpu_count

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler


def standardize(data, scaler=None):
    if scaler is None:
        scaler = StandardScaler().fit(data)

    data = scaler.transform(data)
    return data, scaler


def feature_extractor(df, feature_set, n_models, standardize=True, scaler=None):

    batch_size = df.shape[0]

    # Get forecasts
    forecasts = df.loc[:, "auto_arima_0":"quant_99_reg_47"]

    # Get feature inputs
    if feature_set == "":
        raise Exception(
            "Manual_or_auto_toggle needs to cointain either m or a for input to be non-empty"
        )

    inputs_start = "x_acf1" if "m" in feature_set.lower() else "lstm_0"
    inputs_end = "lstm_31" if "a" in feature_set.lower() else "series_length"

    inputs = df.loc[:, inputs_start:inputs_end].to_numpy()

    if standardize or scaler is not None:
        inputs, scaler = standardize(inputs, scaler)

    inputs_cat = df.loc[:, ["type", "period"]].astype("category")
    emb_dims = [
        (x, min(x // 2, 50))
        for x in map(lambda y: len(inputs_cat[y].cat.categories), inputs_cat)
    ]

    for col in inputs_cat:
        inputs_cat[col] = inputs_cat[col].cat.codes

    inputs_cat = torch.as_tensor(inputs_cat.to_numpy(), dtype=torch.long)

    # Get actuals
    actuals = df.loc[:, "actuals_0":"actuals_47"].to_numpy()
    forecasts = forecasts.to_numpy().reshape((batch_size, n_models, 48)).swapaxes(1, 2)

    return (inputs_cat, emb_dims), inputs, forecasts, actuals, scaler


class M4EnsembleData(Dataset):

    def __init__(self, meta_path, feature_set, n_models, subset=None, verbose=True):
        if isinstance(meta_path, pd.DataFrame):
            meta_df = meta_path.copy()
        elif isinstance(meta_path, str):
            meta_df = (
                pd.read_feather(meta_path)
                .set_index("m4id")
                .replace(np.nan, 0)
                .loc[subset]
            )
        else:
            raise Exception(
                "Only pandas DataFrame or path to a feather file legal arguments"
            )

        if verbose: print(f"Loaded df of shape {meta_df.shape}")

        self.h = meta_df["h"].astype(np.int16)
        self.divs = meta_df["mase_divisor"]
        self.n_smape = meta_df["naive2_smape"]
        self.n_mase = meta_df["naive2_mase"]

        self.index = meta_df.index.values
        self.length = meta_df.shape[0]

        (self.cats, self.emb_dims), self.input, self.forecast, self.actuals, scaler = feature_extractor(
            meta_df, feature_set, n_models)

        self.num_cont = self.input.shape[1]

        return self, scaler

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            self.cats[idx],
            self.input[idx],
            self.forecast[idx],
            self.actuals[idx],
            self.divs[idx],
            self.n_smape[idx],
            self.n_mase[idx],
            self.h[idx],
        )

def ensemble_loaders(
    datapath,
    splitpath=None,
    batch_size=512,
    feature_set="ma",
    n_models=9,
    cpus=None,
    training=True,
    verbose=True,
    standardize=False,
):
    cpus = cpus or cpu_count()
    if verbose: print(f"CPU count: {cpus}")

    train_idxs, val_idxs = slice(None, None), None

    if splitpath:
        split = pd.read_feather(splitpath).set_index("m4id")
        train_idxs = split[split.val == False].index
        val_idxs = split[split.val == True].index

    data1, scaler = M4EnsembleData(datapath, feature_set, n_models,
                           subset=train_idxs, verbose=verbose, standarize=standardize)
    loader1 = DataLoader(data1, batch_size=batch_size,
                         shuffle=training, num_workers=cpus, drop_last=training)

    if val_idxs is not None:
        data2, _ = M4EnsembleData(
            datapath, feature_set, n_models, subset=val_idxs, verbose=verbose, scaler=scaler)
        loader2 = DataLoader(data2, batch_size=batch_size,
                             shuffle=False, num_workers=cpus)

        return loader1, loader2, data1.emb_dims, data1.num_cont, data1.length

    return loader1, data1.emb_dims, data1.num_cont, data1.length


def ensemble_loaders_kfold(
    data,
    val,
    batch_size=512,
    feature_set="ma",
    n_models=9,
    cpus=None,
    training=True,
):
    cpus = cpus or cpu_count()
    print(f"CPU count: {cpus}")

    data1 = M4EnsembleData(data, feature_set, n_models)
    loader1 = DataLoader(data1, batch_size=batch_size,
                         shuffle=training, num_workers=cpus, drop_last=training)

    data2 = M4EnsembleData(
        val, feature_set, n_models)
    loader2 = DataLoader(data2, batch_size=batch_size,
                         shuffle=False, num_workers=cpus)

    return loader1, loader2, data1.emb_dims, data1.num_cont, data1.length
