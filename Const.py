import torch


def get_device():
    if torch.cuda.is_available():
        return 'cuda:0'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


device = get_device()
