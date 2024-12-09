import torch
import torch.nn as nn
from torch.autograd import Variable
from Const import device
import numpy as np


class LTSF_LSTM(torch.nn.Module):
    def __init__(self, output_window, feature_size, hidden_size, dropout=0.2):
        super(LTSF_LSTM, self).__init__()
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
            bidirectional=self.bidirectional,
            dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_window)
        )

        self.bidirectional_f = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_window)
        )

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
            out = self.fc(out)

        return out


def calculate_probabilities(predictions, std_dev, bins):
    probabilities = []
    for pred in predictions:
        bin_probs = []
        for lower, upper in bins:
            # Calculate cumulative probabilities for the bin
            p_lower = 0.5 * (1 + torch.erf((lower - pred) / (std_dev * np.sqrt(2))))
            p_upper = 0.5 * (1 + torch.erf((upper - pred) / (std_dev * np.sqrt(2))))
            bin_probs.append(p_upper - p_lower)
        probabilities.append(bin_probs)
    return torch.tensor(probabilities)