import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence


def plot_examples(figurepath,
                  data1,
                  example_packed,
                  model,
                  lens,
                  epoch,
                  conf,
                  rows=3,
                  cols=3,
                  size=(16, 8),
                  show=False,):

    def plot(ax, d1, d2):
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.plot(d1, label=f'Original')
        ax.plot(d2, label=f'Reconstructed')
        ax.legend()

    data2 = pad_packed_sequence(model(example_packed, lens), batch_first=True, total_length=conf.maxlen)[0].detach().cpu().numpy()

    plt.style.use('seaborn-pastel')
    figs = np.array(range(rows*cols)).reshape((rows, cols))
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=size)
    meta = {
        "epoch": epoch,
        "ed": conf.embedding_dim,
        "hd": conf.hidden_dim,
        "drop": conf.dropout,
        "opt": conf.optimizer,
        "batch": conf.batch_size,
        "maxlen": conf.maxlen,
        "normal": conf.normalize_data,
    }

    fig.suptitle(f"Results on val data ({meta})", size=14)

    idx = 0

    for rfig, rdat in zip(axs, figs):
        for ax, r in zip(rfig, rdat):
            plot(ax, data1[r][:lens[r]], data2[r][:lens[r]])
            idx += 1

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    plt.savefig(figurepath)
    if not show:
        plt.close(fig)