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
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected output layer
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # Embed input
        x = self.input_embedding(x) + self.positional_encoding.to(x.device)
        x = self.transformer_encoder(x)
        return x
        # Use the last time step's output for prediction
        # output = self.fc_out(x[:, -1, :])
        # return output

    def _get_positional_encoding(self, seq_len, d_model):
        # Generate Positional Encoding
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


def default_model(input_window, output_window, feature_size, hidden_size, dropout_rate, num_layers, num_heads):
    transformer = TimeSeriesTransformer(
        input_dim=feature_size,
        d_model=hidden_size,
        n_heads=num_heads,
        num_layers=num_layers,
        seq_len=input_window,
        output_dim=output_window,
        dropout_rate=dropout_rate
    )
    return QuantileTransformer(
        transformer=transformer,
        hidden_dim=hidden_size,
    )


class QuantileTransformer(nn.Module):
    def __init__(self, transformer, hidden_dim, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.transformer = transformer  # 기존 Transformer 모델
        self.output_layers = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in quantiles])
        self.quantiles = quantiles

    def forward(self, x):
        hidden = self.transformer(x)  # Transformer 출력값
        # 🔹 마지막 시점(`seq_len - 1`)의 값만 선택
        hidden = hidden[:, -1, :]  # (batch_size, hidden_dim)

        # 🔹 분위수별 예측값 출력 -> (batch_size, 1) 형태 유지
        return [layer(hidden) for layer in self.output_layers]  # 분위수별 예측값 반환

# Model Configuration
# input_dim = 6  # 입력 특성 수 (open, high, low, close, volume 등)
# d_model = 64  # 모델 차원
# n_heads = 4  # Attention 헤드 수
# num_layers = 3  # Transformer 계층 수
# seq_len = 50  # 시계열 길이 (50일)
# output_dim = 1  # 출력 차원 (다음날 close 변화율)
# dropout_rate = 0.1  # 드롭아웃 비율
#
