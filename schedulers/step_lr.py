import torch.optim as optim


class Step_LR:
    def __init__(self, optimizer, args):
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma,
                                                   last_epoch=args.last_epoch, verbose=args.verbose)

    def step(self, in_=None):
        self.scheduler.step()
