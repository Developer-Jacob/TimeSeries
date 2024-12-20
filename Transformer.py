import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    # d_model: 모델차원
    # n_heads: 헤드수
    # num_layers: 계층수
    # seq_len: input window
    # output_dim: output window
    def __init__(self, input_dim, d_model, n_heads, num_layers, seq_len, output_dim, dropout_rate):
        super(TimeSeriesTransformer, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._get_positional_encoding(seq_len, d_model)
        self.batch_norm = nn.BatchNorm1d(seq_len)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected output layer
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # Embed input
        x = self.input_embedding(x) + self.positional_encoding.to(x.device)
        x = self.batch_norm(x)
        x = self.transformer_encoder(x)
        # Use the last time step's output for prediction
        output = self.fc_out(x[:, -1, :])
        return output

    def _get_positional_encoding(self, seq_len, d_model):
        # Generate Positional Encoding
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


# Model Configuration
# input_dim = 6  # 입력 특성 수 (open, high, low, close, volume 등)
# d_model = 64  # 모델 차원
# n_heads = 4  # Attention 헤드 수
# num_layers = 3  # Transformer 계층 수
# seq_len = 50  # 시계열 길이 (50일)
# output_dim = 1  # 출력 차원 (다음날 close 변화율)
# dropout_rate = 0.1  # 드롭아웃 비율
#
