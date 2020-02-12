import torch.optim.lr_scheduler as lr_scheduler


def step(optimizer, last_epoch, step_size=80, gamma=0.1, **_):
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)


def multi_step(optimizer, last_epoch, milestones=[500, 5000], gamma=0.1, **_):
    if isinstance(milestones, str):
        milestones = eval(milestones)
    return lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                     gamma=gamma, last_epoch=last_epoch)


def exponential(optimizer, last_epoch, gamma=0.99, **_):
    return lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)


def none(optimizer, last_epoch, **_):
    return lr_scheduler.StepLR(optimizer, step_size=10000000, last_epoch=last_epoch)


def reduce_lr_on_plateau(optimizer, last_epoch, mode='max', factor=0.1, patience=10,
                         threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, **_):
    return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                           threshold=threshold, threshold_mode=threshold_mode,
                                           cooldown=cooldown, min_lr=min_lr)


def cosine(optimizer, last_epoch, T_max=50, eta_min=0.00001, **_):
    print('cosine annealing, T_max: {}, eta_min: {}, last_epoch: {}'.format(T_max, eta_min, last_epoch))
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min,
                                           last_epoch=last_epoch)


def one_cycle(optimizer, last_epoch, **_):
    return lr_scheduler.OneCycleLR(optimizer, max_lr=0.0003, total_steps=None, epochs=60, steps_per_epoch=76,
                                    pct_start=0.3, anneal_strategy='cos',
                                    cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
                                    div_factor=1.0, final_div_factor=1000.0, last_epoch=last_epoch)


def one_cycle_double(optimizer, last_epoch, **_):
    return lr_scheduler.OneCycleLR(optimizer, max_lr=[0.00003,0.0003], total_steps=None, epochs=80, steps_per_epoch=76,
                                    pct_start=0.2, anneal_strategy='cos',
                                    cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
                                    div_factor=1.0, final_div_factor=1000.0, last_epoch=last_epoch)


def get_scheduler(config, optimizer, last_epoch):
    print('scheduler name:', config.SCHEDULER.NAME)
    f = globals().get(config.SCHEDULER.NAME)

    if config.SCHEDULER.PARAMS is None:
        return f(optimizer, last_epoch)
    else:
        return f(optimizer, last_epoch, **config.SCHEDULER.PARAMS)
