import torch.optim as optim


class Reduce_LR_On_Plateau:
    def __init__(self, optimizer, args):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=args.mode, factor=args.factor,
                                                              patience=args.patience, threshold=args.threshold,
                                                              verbose=args.verbose)

    def step(self, in_):
        self.scheduler.step(in_)
