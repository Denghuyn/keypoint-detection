from torchvision.models import efficientnet_b0
from torch import nn

def get_model(args):
    if args.model == "efficientnet_b0":
        model = efficientnet_b0 (weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(1280, out_features=args.num_landmarks)
    else:
        raise ValueError(f"model {args.model} is not supported")
    return model