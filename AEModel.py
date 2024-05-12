import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        if len(x.size()) == 3:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        return y


class Autoencoder(nn.Module):
    def __init__(self, input_window):
        super(Autoencoder, self).__init__()
        self.input_window = input_window
        self.encoder = nn.LSTM(
            input_size=17,
            hidden_size=32,
            dropout=0.25,
            num_layers=2,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder = nn.LSTM(
            input_size=64,
            hidden_size=32,
            dropout=0.25,
            num_layers=2,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = TimeDistributed(nn.Linear(64, 17))
    def forward(self, x):
        # x, (batch_size, sequence_length(window_size), features)
        # input x: 64, 10, 17
        # output h0: batch, size, bidden_size * bid(2) 64, 10 32
        h0, (h_n, c_n) = self.encoder(x)
        h0, (h_n, c_n) = self.decoder(h0[:, -1:, :].repeat(1, self.input_window, 1))    # 64 10 32
        out = self.fc(h0)
        return out