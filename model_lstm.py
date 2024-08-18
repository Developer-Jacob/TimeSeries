import torch
import torch.nn as nn
from torch.autograd import Variable
from Const import device
from AEModel import CnnAutoEncoder

class LTSF_LSTM(torch.nn.Module):
    def __init__(self, input_window, output_window, feature_size, hidden_size):
        super(LTSF_LSTM, self).__init__()
        self.input_window = input_window
        self.output_window = output_window
        self.hidden_size = hidden_size
        self.feature_size = feature_size
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
