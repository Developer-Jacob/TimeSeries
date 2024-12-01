import torch
import torch.nn as nn
from mamba_ssm import Mamba


class VisionMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x):
        return self.mamba(x)


class VMRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, d_model):
        super().__init__()
        self.vision_mamba = VisionMambaBlock(d_model)
        self.lstm = nn.LSTMCell(input_size, hidden_size)

    def forward(self, input, hidden):
        h, c = hidden
        mamba_out = self.vision_mamba(input)
        h, c = self.lstm(mamba_out, (h, c))
        return h, (h, c)


class VMRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, d_model):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.vmrnn_cells = nn.ModuleList([
            VMRNNCell(input_size if i == 0 else hidden_size, hidden_size, d_model)
            for i in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        hidden = [(torch.zeros(batch_size, self.hidden_size).to(x.device),
                   torch.zeros(batch_size, self.hidden_size).to(x.device))
                  for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            input = x[:, t, :]
            for i, cell in enumerate(self.vmrnn_cells):
                out, hidden[i] = cell(input, hidden[i])
                input = out

            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)
        predictions = self.fc(outputs)
        return predictions


# 모델 초기화 예시
input_size = 10
hidden_size = 64
output_size = 1
num_layers = 3
d_model = 32

model = VMRNN(input_size, hidden_size, output_size, num_layers, d_model)