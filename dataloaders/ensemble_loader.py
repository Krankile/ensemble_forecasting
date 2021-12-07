from os import cpu_count

from torch.utils.data import DataLoader

from ..datasets.ensemble_set import M4EnsembleData


def ensemble_dataloaders(
    path1,
    path2=None,
    batch_size=512,
    manual_or_auto_toggle="ma",
    n_models=9,
    normalize="standard",
):

    cpus = cpu_count()
    print(f"CPU count: {cpus}")
    data1 = M4EnsembleData(path1, manual_or_auto_toggle, n_models, normalize)
    loader1 = DataLoader(data1, batch_size=batch_size, shuffle=True, num_workers=cpus, drop_last=True)
    
    if path2:
        data2 = M4EnsembleData(path2, manual_or_auto_toggle, n_models, normalize)
        loader2 = DataLoader(data2, batch_size=batch_size, shuffle=False, num_workers=cpus)

        return loader1, loader2, data1.emb_dims, data1.num_cont, data1.length

    return loader1, data1.emb_dims, data1.num_cont, data1.length