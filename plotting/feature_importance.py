import numpy as np
import matplotlib.pyplot as plt

from .colors import main, main2
from . import plot_standards as ps


def plot_feature_importance(importances, *, savepath=None, show=True, normalize=False):
    mean_results = {k: np.mean(v) for k, v in importances.items()}
    sorted_results = sorted(mean_results.items(), key=lambda x: x[1], reverse=True)
    lab, vals = zip(*sorted_results)
    vals = np.array(vals)

    if normalize:
        vals /= vals.sum()

    plt.figure(figsize=(ps.figwidth, 12))

    zero = np.zeros(len(lab))
    plt.barh(lab, zero, color=main)

    plt.barh(*zip(*filter(lambda x: "lstm" in x[0], zip(lab, vals))), alpha=1, color=main, label="LSTM AE features")
    plt.barh(*zip(*filter(lambda x: "lstm" not in x[0], zip(lab, vals))), alpha=1, color=main2, label="Statistical features")
    
    plt.xlabel("OWA loss relative to baseline", fontsize=ps.textsize)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=ps.ticksize)
    
    plt.title("Feature importance calculations across all 76 features for validation data", fontsize=ps.textsize)
    plt.legend(fontsize=ps.textsize)

    if savepath is not None:
        plt.savefig(savepath)

    if show:
        plt.show()