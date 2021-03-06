from sklearn.preprocessing import StandardScaler
import pandas as pd


def art2df(run, arts, *, root="krankile/data-processing/"):
    dfs = []

    if isinstance(arts, str):
        arts = [arts]

    for a in arts:

        data = run.use_artifact(root + a)
        data.download()
        dfs.append(pd.read_feather(data.file()).set_index("m4id"))

    return dfs[0] if len(arts) == 1 else dfs


def do_standardize(data, scaler=None):
    if scaler is None:
        scaler = StandardScaler().fit(data)

    data = scaler.transform(data)
    return data, scaler


def split_traval(data, split):
    tra = data.loc[split[split.val == False].index]
    val = data.loc[split[split.val == True].index]

    return tra, val
