from multiprocessing import cpu_count

from torch.utils.data import DataLoader

from ..datasets.m4_series import M4Data


def get_dataloaders(train_path, val_path, batch_size, manual_or_auto_toggle, normalize="standard"):

    cpus = cpu_count()
    print(f"CPU count: {cpus}")
    train_data = M4Data(train_path, manual_or_auto_toggle, normalize)
    val_data = M4Data(val_path, manual_or_auto_toggle, normalize)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=cpus, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            shuffle=False, num_workers=cpus)

    return train_loader, val_loader, train_data.emb_dims, train_data.num_cont, train_data.length
