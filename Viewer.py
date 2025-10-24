import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantileLoss(nn.Module):
    def __init__(self, quantiles=None):
        """
        분위수(Quantiles) 리스트를 입력으로 받음
        예: quantiles=[0.1, 0.5, 0.9]
        """
        super().__init__()
        self.quantiles = quantiles
        if quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]

    def forward(self, y_pred, y_true):
        """
        y_true: 실제값 (batch_size, seq_len, 1)
        y_pred: 분위수별 예측값 리스트 [q1_pred, q2_pred, q3_pred] (각각 (batch_size, seq_len, 1))
        """
        loss = 0
        for i, q in enumerate(self.quantiles):
            errors = y_true - y_pred[i]
            loss += torch.mean(torch.max(q * errors, (q - 1) * errors))
        return loss

class QuantileHuberLoss(nn.Module):
    def __init__(self, quantiles=None, delta=1.0, consistency_weight=0.1):
        """
        Args:
            quantiles (list or tensor): 예측할 분위수 리스트 (ex. [0.1, 0.5, 0.9])
            delta (float): Huber Loss에서 MSE와 MAE 전환 경계값
        """
        super().__init__()
        self.quantiles = quantiles
        if quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]
        self.delta = delta
        self.consistency_weight = consistency_weight

    def forward(self, preds, target):
        """
        Args:
            preds (list of Tensors): 각 분위수에 대한 예측값 리스트, 각각 (batch_size, 1)
            target (Tensor): 실제 정답값, shape (batch_size,)
        Returns:
            scalar loss (Tensor)
        """
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            pred = preds[i].squeeze(-1)  # (batch_size,)
            error = target - pred        # 오차: y - f(x)

            # Huber 기반 pinball loss
            huber = torch.where(
                error.abs() <= self.delta,
                0.5 * error.pow(2),
                self.delta * (error.abs() - 0.5 * self.delta)
            )
            quantile_loss = torch.max((q - 1) * error, q * error)
            loss += torch.mean(huber * quantile_loss)

        loss = loss / len(self.quantiles)  # 분위수 평균

        if len(preds) >= 2 and self.consistency_weight > 0:
            consistency = 0.0
            for i in range(len(preds) - 1):
                lower = preds[i].squeeze(-1)
                upper = preds[i + 1].squeeze(-1)
                consistency += F.relu(lower - upper).mean()
            consistency /= (len(preds) - 1)
            loss += self.consistency_weight * consistency

        return loss

# Main Execution
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define loss function
    # loss_fn = ImprovedCustomLoss(penalty_weight=0.1, sensitivity=5.0)
    # loss_fn = ImprovedCustomLoss(penalty_weight=0.1, sensitivity=5.0, large_error_weight=0.5)
    # Generate data
    y_pred = torch.linspace(-3, 3, 300)  # Predicted values
    y_true_positive = torch.full_like(y_pred, 1.0)  # Positive target
    y_true_negative = torch.full_like(y_pred, -1.0)  # Negative target

    # Compute loss for different cases
    loss_positive = loss_fn(y_pred, y_true_positive).detach().numpy()
    loss_negative = loss_fn(y_pred, y_true_negative).detach().numpy()

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y_pred.numpy(), loss_positive, label="Target: +1.0 (Same Sign)", color="blue")
    plt.plot(y_pred.numpy(), loss_negative, label="Target: -1.0 (Different Sign)", color="red")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Improved Custom Loss Function Visualization")
    plt.xlabel("Predicted Value (y_pred)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()