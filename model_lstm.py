import torch
import torch.nn as nn
from torch.autograd import Variable
from Const import device

class LTSFConv(torch.nn.Module):
    def __init__(self, feature_size):
        super(LTSFConv, self).__init__()
        # input (batch_size, feature_dim, data_length)
        self.conv1 = torch.nn.Conv1d(
            in_channels=feature_size,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=8,
            out_channels=16,
            kernel_size=3
        )

        self.conv3 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3
        )

        self.de_conv3 = nn.ConvTranspose1d(
            in_channels=32,
            out_channels=16,
            kernel_size=3
        )

        self.de_conv2 = nn.ConvTranspose1d(
            in_channels=16,
            out_channels=8,
            kernel_size=3
        )

        self.de_conv1 = nn.ConvTranspose1d(
            in_channels=8,
            out_channels=feature_size,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute((0, 2, 1))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.de_conv3(x))
        x = self.relu(self.de_conv2(x))
        x = self.de_conv1(x)
        x = x.permute((0, 2, 1))
        return x



class LTSF_LSTM(torch.nn.Module):
    def __init__(self, output_window, feature_size, hidden_size):
        super(LTSF_LSTM, self).__init__()
        self.output_window = output_window
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.bidirectional = False

        self.lstm = torch.nn.LSTM(
            input_size=feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.relu = nn.ReLU()
        self.f = nn.Linear(self.hidden_size, 1)
        self.bidirectional_f = nn.Linear(self.hidden_size * 2, 1)
    def forward(self, x):
        # x, (batch_size, sequence_length(window_size), features)

        if self.bidirectional == True:
            h_0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)).to(device)  # hidden state
            c_0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)).to(device)  # internal state
            output, (hidden, _) = self.lstm(x, (h_0, c_0))
            out = output[:, -self.output_window:, :]
            out = self.bidirectional_f(out)
        else:
            h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # hidden state
            c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # internal state
            output, (hidden, _) = self.lstm(x, (h_0, c_0))
            out = output[:, -self.output_window:, :]
            out = self.f(out)

        return out
