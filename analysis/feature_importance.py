from collections import defaultdict
from functools import partial


import torch
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..models import weightnets
from ..data import ensemble_loaders
from ..utils.data import art2df, split_traval, do_standardize
from ..utils import loss_functions


def get_data(run, test, final=None):

    if test:
        assert final is not None, "Must set argument final when test True"

        if final:
            test = run.use_artifact(
                "krankile/data-processing/ensemble_standarized_test:len-500")
            test.download()
            test = pd.read_feather(test.file()).set_index("m4id")
        else:
            data, split, test = art2df(
                run, ["ensemble_traval:len-500", "traval_split_80_20:v0", "ensemble_test:v5"])
            tra, val = split_traval(data, split)

            feat_cols = data.loc[:, "x_acf1":"lstm_31"].columns

            _, scaler = do_standardize(tra[feat_cols], scaler=None)
            test[feat_cols], _ = do_standardize(test[feat_cols], scaler=scaler)

        return test

    else:
        arts = ["ensemble_traval:len-500", "traval_split_80_20:v0"]
        data, split = art2df(run, arts)
        tra, val = split_traval(data, split)

        feat_cols = data.loc[:, "x_acf1":"lstm_31"].columns
        _, scaler = do_standardize(tra[feat_cols], scaler=None)
        val[feat_cols], _ = do_standardize(val[feat_cols], scaler=scaler)

        return val


def get_loss(model, loader, device, loss="owa"):
    loss_func = loss_functions[loss]

    with torch.no_grad():
        owas = []
        for tensors in loader:
            cats, inputs, forecasts, actuals, * \
                loss_args = map(lambda x: x.to(device), tensors)
            inputs, forecasts = inputs.to(
                torch.float32), forecasts.to(torch.float32)

            y_pred = model(cats, inputs).unsqueeze(2)
            prediction = torch.matmul(forecasts, y_pred).squeeze(2)
            owa = loss_func(prediction, actuals, *loss_args)

            owas.append(owa.item())

    return np.mean(owas)


def calculate_feature_importance(*, run, device, test, final, modelpath, progress=True):
    conf = run.config

    data = get_data(run, test=test, final=final)

    get_loader = partial(ensemble_loaders,
                         batch_size=data.shape[0],
                         feature_set=conf.feature_set,
                         n_models=conf.num_models,
                         training=False, verbose=False,
                         standardize=False,
                         )

    (loader,
     emb_dims,
     num_cont,
     _,) = get_loader(datapath=data)

    model = weightnets[conf.architecture](
        num_cont=num_cont,
        out_size=conf.num_models,
        n_hidden=conf.n_hidden,
        hidden_dim=conf.hidden_dim,
        dropout=conf.dropout,
        bn=conf.bn,
        activation=conf.act,
        emb_dims=emb_dims,
    )

    model.load_state_dict(torch.load(modelpath, map_location=device))
    model = model.to(device).eval()

    # Run calculations
    baseline = get_loss(model, loader)

    cols = data.drop(columns=['n', 'h', 'mase_divisor', 'naive2_smape',
                     'naive2_mase']).loc[:, "type":"lstm_31"].columns
    it = tqdm(cols) if progess else iter(cols)

    results = defaultdict(list)
    for col in it:
        for _ in range(1):
            cp = data.copy()
            cp[col] = shuffle(cp[col]).to_numpy()

            loader, *_ = get_loader(datapath=cp)
            results[col].append(get_loss(model, loader) - baseline)

    return results