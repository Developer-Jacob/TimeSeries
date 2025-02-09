import torch


def get_device():
    exe_device: str = 'cpu'
    if torch.cuda.is_available():
        exe_device = 'cuda:0'
    elif torch.backends.mps.is_available():
        exe_device = 'mps'
    else:
        exe_device = 'cpu'
    print(exe_device)
    return exe_device


device = get_device()
