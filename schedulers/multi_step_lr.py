import torch.optim as optim


class MultiStepLR:
    def __init__(self, optimizer, args):
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    def step(self, in_=None):
        self.scheduler.step()
