import torch
import os
from tf_model import *


def load_model(path,
               input_window,
               output_window,
               d_model=256,
               n_head=4,
               n_layers=2,
               dropout=0.1
               ):
    epoch_start = 0
    lr = 1e-4
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = TFModel(input_window, output_window, d_model, n_head, n_layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch'] + 1
        print("successfully loaded!")
        print("epoch saved until here: ", epoch_start - 1)
        print("train starts from this epoch: Epoch ", epoch_start)

    return epoch_start, model.to(device), optimizer


def save(path, epoch, model, optimizer):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    },
        path
    )
