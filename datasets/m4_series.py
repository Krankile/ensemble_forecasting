from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from ..utils.feature_extractor import feature_extractor


class M4Data(Dataset):

    def __init__(self, path, manual_or_auto_toggle, type_of_normalization="standard"):
        df = pd.read_feather(path).set_index("index").replace(np.nan, 0)
        self.index = df.index.values
        self.length = df.shape[0]

        (self.cats, emb_dims), self.input, self.forecast, self.actuals, self.mask = feature_extractor(
            df, manual_or_auto_toggle, type_of_normalization)

        self.num_cont = self.input.shape[1]
        self.emb_dims = emb_dims

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.cats[idx], self.input[idx], self.forecast[idx], self.actuals[idx], self.mask[idx]
