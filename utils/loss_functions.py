import torch
from torch import nn


def mase(pred, actual, *args):
    divs, *_ = args
    return torch.div(nn.functional.l1_loss(pred, actual, reduction="none").sum(1), divs).mean()

def smape(pred, actual, *args):
    *_, h = args
    return 200 * torch.div(((pred - actual).abs() / (pred.abs() + actual.abs() + 1e-40)).sum(1), h).mean()

def owa(pred, actual, *args):
    _, n_smape, n_mase, __ = args
    return 0.5*(torch.div(smape(pred, actual, *args), n_smape.mean()) + torch.div(mase(pred, actual, *args), n_mase.mean()))

loss_functions = {
    "smape": smape,
    "mase": mase,
    "owa": owa,
}