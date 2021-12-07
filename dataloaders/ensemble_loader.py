from os import cpu_count

from torch.utils.data import DataLoader

from ..datasets.ensemble_set import M4EnsembleData


def ensemble_dataloaders(
    train_path,
    val_path,
    loss_train_path,
    loss_val_path,
    batch_size,
    manual_or_auto_toggle,
    n_models, normalize="standard",
):

    cpus = cpu_count()
    print(f"CPU count: {cpus}")
    train_data = M4EnsembleData(train_path, loss_train_path, manual_or_auto_toggle, n_models, normalize)
    val_data = M4EnsembleData(val_path, loss_val_path, manual_or_auto_toggle, n_models, normalize)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=cpus, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=cpus)

    return train_loader, val_loader, train_data.emb_dims, train_data.num_cont, train_data.length