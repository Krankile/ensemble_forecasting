import pandas as pd


def art2df(run, arts, *, root="krankile/data-processing/"):
    dfs = []

    for a in arts:

        data = run.use_artifact(root + a); data.download()
        dfs.append(
            pd.read_feather(data.file()).set_index("m4id")
        )

    return dfs