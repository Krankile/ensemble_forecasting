from torch.optim import lr_scheduler


def schedule_noop(*args, **kwargs):
    pass

schedule_noop.step = lambda *args, **kwargs: None

schedulers = {
    None: lambda opt: schedule_noop,
    "ExponentialLR": lambda opt, **kwargs: lr_scheduler.ExponentialLR(opt, **kwargs),
    "MultiStepLR": lambda opt, **kwargs: lr_scheduler.MultiStepLR(opt, **kwargs),
}