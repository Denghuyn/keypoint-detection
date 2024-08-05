from torch import nn

def evaluation (pred, target):
    metric_val = nn.L1Loss()(pred, target)
    return metric_val