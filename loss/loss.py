from torch import nn

def get_loss(args):
    if args.loss == 'MSELoss':
        loss_func = nn.MSELoss()
    return loss_func