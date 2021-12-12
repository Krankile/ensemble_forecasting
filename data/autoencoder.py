from psutil import cpu_count

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ..utils import normalizers


class AutoEncoderData(Dataset):
    def __init__(self, paths, maxlen=2_000, nexamples=None, normalize="normal"):
        if isinstance(paths["series"], pd.DataFrame):
            series = paths["series"].iloc[:nexamples, :]
            self.info = paths["info"].iloc[:nexamples, :]
        elif isinstance(paths["series"], str):
            series = pd.read_feather(paths["series"]).set_index(
                "m4id").iloc[:nexamples, :]
            self.info = pd.read_feather(paths["info"]).set_index(
                "m4id").iloc[:nexamples, :]
        else:
            raise Exception("Invalid type of data")

        self.maxlen = maxlen
        self.normalize = normalize
        self.index = series.index.values

        self.lens = self.info.n.to_list()
        self.clamped_lens = torch.IntTensor(self.lens).clamp(max=maxlen)

        series = series.to_numpy()
        self.series = self.pad_series(series)

    def pad_series(self, series):
        mlen = self.maxlen
        norm = normalizers[self.normalize]

        data = np.zeros((series.shape[0], mlen, 1), dtype=np.float32)

        for i, (l, s) in enumerate(zip(self.lens, series)):
            l = min(mlen, l)
            s = s[max(0, l-mlen):l].reshape((-1, 1))
            data[i, :l] = norm()(s)

        return data

    def __len__(self):
        return self.series.shape[0]

    def __getitem__(self, idx):
        x = self.series[idx]
        lens = self.clamped_lens[idx]
        index = self.index[idx]
        return x, lens, index


def autoencoder_loaders(run, paths1, paths2=None, cpus=None):
    conf = run.config
    cpus = cpus or cpu_count()

    print(f"Using {cpus} CPUs in dataloaders")
    seq_len, num_features = conf.maxlen, 1

    data1 = AutoEncoderData(paths1, maxlen=conf.maxlen,
                            nexamples=conf.get("n_train"), normalize=conf.normalize_data)
    loader1 = DataLoader(data1, batch_size=conf.batch_size,
                         shuffle=True, num_workers=cpus, pin_memory=True)

    if paths2:
        data2 = AutoEncoderData(
            paths2, maxlen=conf.maxlen, nexamples=conf.conf.get("n_val"), normalize=conf.normalize_data)
        loader2 = DataLoader(data2, batch_size=conf.batch_size,
                             shuffle=False, num_workers=cpus, pin_memory=True)
        return loader1, loader2, seq_len, num_features

    return loader1, seq_len, num_features
