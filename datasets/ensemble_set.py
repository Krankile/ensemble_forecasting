import pandas as pd
from torch.utils.data import Dataset

from ..utils.feature_extractor import feature_extractor


class M4Data(Dataset):

    def __init__(self, meta_path, loss_path, manual_or_auto_toggle, n_models, type_of_normalization="standard"):
        meta_df = pd.read_feather(meta_path).set_index(
            "index").replace(np.nan, 0)
        loss_df = pd.read_feather(loss_path).set_index("st").loc[meta_df.index]

        self.h = meta_df["h"].astype(np.int16)
        self.divs = loss_df["mase_divisor"]
        self.n_smape = loss_df["naive2_smape"]
        self.n_mase = loss_df["naive2_mase"]

        self.index = meta_df.index.values
        self.length = meta_df.shape[0]

        (self.cats, emb_dims), self.input, self.forecast, self.actuals = feature_extractor(
            meta_df, manual_or_auto_toggle, type_of_normalization, n_models)

        self.num_cont = self.input.shape[1]
        self.emb_dims = emb_dims

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.cats[idx], self.input[idx], self.forecast[idx], self.actuals[idx], self.divs[idx], self.n_smape[idx], self.n_mase[idx], self.h[idx]
