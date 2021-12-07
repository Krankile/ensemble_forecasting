from torch.optim import lr_scheduler


def schedule_noop(*_, **__):
    pass

schedule_noop.step = lambda *_, **__: None

schedulers = {
    None: lambda _, **__: schedule_noop,
    "ExponentialLR": lambda opt, **kwargs: lr_scheduler.ExponentialLR(opt, **kwargs),
    "MultiStepLR": lambda opt, **kwargs: lr_scheduler.MultiStepLR(opt, **kwargs),
}