import torch.optim as optim


def sgd(params, args):
    return optim.SGD(params, lr=args.lr, momentum=args.momentum, dampening=args.dampening, nesterov=args.nesterov, weight_decay=args.weight_decay)
