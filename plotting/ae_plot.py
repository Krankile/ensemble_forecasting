import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence
from .colors import main, main2


def plot_examples(figurepath,
                  data1,
                  example_packed,
                  model,
                  lens,
                  conf,
                  rows=3,
                  cols=3,
                  size=(16, 8),
                  textsize=16,
                  ticksize=14,
                  show=False,):

    def plot(ax, d1, d2):
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.plot(d1, label=f'Original', color=main)
        ax.plot(d2, label=f'Reconstructed', color=main2)
        ax.tick_params(labelsize=ticksize)
        ax.legend(fontsize=textsize)

    data2 = pad_packed_sequence(model(example_packed, lens), batch_first=True, total_length=conf.maxlen)[0].detach().cpu().numpy()

    figs = np.array(range(rows*cols)).reshape((rows, cols))
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=size)

    fig.suptitle(f"Autoencoder reconstructions on val data", size=textsize)

    idx = 0

    for rfig, rdat in zip(axs, figs):
        for ax, r in zip(rfig, rdat):
            plot(ax, data1[r][:lens[r]], data2[r][:lens[r]])
            idx += 1

    axs[rows // 2][0].set_ylabel("Standardized value", fontsize=textsize)
    axs[-1][rows // 2].set_xlabel("Timestep", fontsize=textsize)

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    plt.savefig(figurepath)
    if not show:
        plt.close(fig)
