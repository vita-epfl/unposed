import torch.optim as optim


def adamw(params, args):
    return optim.AdamW(params, lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay)
