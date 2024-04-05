import torch
import torch.nn as nn
from torch.autograd import Variable
from Const import device

class LTSF_LSTM(torch.nn.Module):
    def __init__(self, output_window, feature_size, hidden_size):
        super(LTSF_LSTM, self).__init__()
        self.output_window = output_window
        self.hidden_size = hidden_size
        self.num_layers = 1

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

        self.lstm = torch.nn.LSTM(
            input_size=feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.fc_1 = nn.Linear(self.hidden_size, 256)
        # self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, 128)
        self.fc_4 = nn.Linear(128, 64)
        self.fc_5 = nn.Linear(64, 32)
        self.fc = nn.Linear(32, 1)

        self.relu = nn.ReLU()

        self.f = nn.Linear(self.hidden_size, 1)
    def forward(self, x):
        # x, (batch_size, sequence_length(window_size), features)
        x = self.conv1(x.permute(0, 2, 1))
        x = self.relu(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.de_conv2(x)
        # x = self.relu(x)
        x = self.de_conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # internal state


        output, (hidden, _) = self.lstm(x, (h_0, c_0))
        # output: (batch_size, sequence_length(window_size), hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        # hidden = hidden.view(-1, self.hidden_size)  #batch, hidden_size
        out = output[:, -self.output_window:, :]

        out = self.f(out)
        # out = self.relu(out)
        # out = self.fc_1(out)
        # out = self.relu(out)  # relu
        # # out = self.fc_2(out) #first Dense
        # # out = self.relu(out) #relu
        # out = self.fc_3(out)
        # out = self.relu(out)
        # out = self.fc_4(out)
        # out = self.relu(out)
        # out = self.fc_5(out)
        # out = self.fc(out)
        return out