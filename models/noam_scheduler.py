import torch
from torch.optim.lr_scheduler import _LRScheduler


class NoamScheduler(_LRScheduler):
    """
    Noam sheduler given in "Attention is all you need" paper.
    """

    def __init__(self, optimizer, warmup):
        assert warmup > 0
        self.optimizer = optimizer
        self.initial_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
        self.warmup = warmup
        self.timestep = 0
        self.noam_lr = 0.0
        super().__init__(optimizer)

    def get_lr(self):
        self.noam_lr = self.get_noam_lr()
        return [init_lr * self.noam_lr for init_lr in self.initial_lrs]

    def get_noam_lr(self):
        return min(self.timestep ** -0.5, self.timestep * self.warmup ** -1.5)

    def step(self, epoch=None):
        self.timestep += 1
        super().step(epoch)