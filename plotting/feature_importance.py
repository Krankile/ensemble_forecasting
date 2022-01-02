import numpy as np
import matplotlib.pyplot as plt

from .colors import main, main2
from . import plot_standards as ps


def plot_feature_importance(
    importances,
    *,
    savepath=None,
    useerror=None,
    show=True,
    normalize=False,
    figheight=12,
    yticksize=10,
    cols=None,
):
    mean_results = {k: np.mean(v) for k, v in importances.items()}
    sorted_results = sorted(mean_results.items(), key=lambda x: x[1], reverse=True)
    lab, vals = zip(*sorted_results)
    vals = np.array(vals)

    if normalize:
        vals /= vals.sum()

    lstm_err = None
    stat_err = None
    if useerror is not None:
        if useerror == "sd":
            err = {k: np.std(v) for k, v in importances.items()}
        elif useerror == "se":
            err = {k: np.std(v) / np.sqrt(len(v)) for k, v in importances.items()}

        lstm_err = [v for k, v in err.items() if "lstm" in k]
        stat_err = [v for k, v in err.items() if "lstm" not in k]

    plt.figure(figsize=(ps.figwidth, figheight))

    zero = np.zeros(len(lab))
    plt.barh(lab, zero, color=main)

    if cols:
        plt.barh(
            y=lab, width=vals, alpha=1, color=main, xerr=err.values(), label="Clusters"
        )
    else:

        plt.barh(
            *zip(*filter(lambda x: "lstm" in x[0], zip(lab, vals))),
            alpha=1,
            color=main,
            xerr=lstm_err,
            label="LSTM AE features",
        )
        plt.barh(
            *zip(*filter(lambda x: "lstm" not in x[0], zip(lab, vals))),
            alpha=1,
            color=main2,
            xerr=stat_err,
            label="Statistical features",
        )

    plt.xlabel("OWA loss relative to baseline", fontsize=ps.textsize)
    plt.yticks(fontsize=yticksize)
    plt.xticks(fontsize=ps.ticksize)

    plt.title(
        "Feature importance calculations across all 76 features for validation data",
        fontsize=ps.textsize,
    )
    plt.legend(fontsize=ps.textsize)

    if savepath is not None:
        plt.savefig(savepath)

    if show:
        plt.show()
