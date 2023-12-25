import torch.optim as optim


def adam(params, args):
    return optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
