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

def triad_loss(pred, actual, *args):
    return owa(pred, actual, *args), smape(pred, actual, *args), mase(pred, actual, *args)

def _mase_analytics(pred, actual, *args):
    divs, *_ = args
    return torch.div(nn.functional.l1_loss(pred, actual, reduction="none").sum(1), divs)

def _smape_analytics(pred, actual, *args):
    *_, h = args
    return 200 * torch.div(((pred - actual).abs() / (pred.abs() + actual.abs() + 1e-40)).sum(1), h)

def _owa_anayltics(pred, actual, *args):
    _, n_smape, n_mase, __ = args
    return 0.5*(torch.div(smape(pred, actual, *args), n_smape) + torch.div(mase(pred, actual, *args), n_mase))

def triad_loss_analytics(pred, actual, *args):
    return _owa_anayltics(pred, actual, *args), _smape_analytics(pred, actual, *args), _mase_analytics(pred, actual, *args)

loss_functions = {
    "smape": smape,
    "mase": mase,
    "owa": owa,
    "triad": triad_loss,
}