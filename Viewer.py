import torch.nn as nn
import torch
import torch
import torch.nn as nn

import torch
import torch.nn as nn


import torch
import torch.nn as nn
class AdaptiveCustomLoss(nn.Module):
    def __init__(self, penalty_weight=0.1, sensitivity=3.0, extra_penalty=5.0, max_penalty=20.0):
        """
        Args:
        - penalty_weight: Initial penalty scale for opposite signs.
        - sensitivity: Controls the sensitivity for small errors.
        - extra_penalty: Fixed penalty for opposite signs.
        - max_penalty: Maximum allowed penalty for opposite signs.
        """
        super(AdaptiveCustomLoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.sensitivity = sensitivity
        self.extra_penalty = extra_penalty
        self.max_penalty = max_penalty

    def forward(self, y_pred, y_true, epoch=1, total_epochs=100):
        # Loss Term: Sensitive to small errors
        loss_term = torch.arctan(self.sensitivity * torch.abs(y_pred - y_true))

        # Penalty Term: Applied only when signs are opposite
        sign_mismatch = 1 - torch.sign(y_pred * y_true)  # 1 if signs differ, 0 if same
        dynamic_penalty = self.penalty_weight * (epoch / total_epochs)  # Gradually increase penalty
        sign_penalty = sign_mismatch * (
            dynamic_penalty * torch.abs(y_pred - y_true) + self.extra_penalty
        )
        sign_penalty = torch.clamp(sign_penalty, max=self.max_penalty)  # Cap penalty

        # Combine terms
        loss = loss_term + sign_penalty
        return loss
class ImprovedCustomLoss2(nn.Module):
    def __init__(self, penalty_weight=0.1, sensitivity=3.0, extra_penalty=5.0):
        """
        Args:
        - penalty_weight: Penalty scale for opposite signs.
        - sensitivity: Controls the sensitivity for small errors.
        - extra_penalty: Fixed penalty for opposite signs.
        """
        super(ImprovedCustomLoss2, self).__init__()
        self.penalty_weight = penalty_weight
        self.sensitivity = sensitivity
        self.extra_penalty = extra_penalty

    def forward(self, y_pred, y_true):
        # Loss Term: Sensitive to small errors
        loss_term = torch.arctan(self.sensitivity * torch.abs(y_pred - y_true))

        # Penalty Term: Applied only when signs are opposite
        sign_mismatch = 1 - torch.sign(y_pred * y_true)  # 1 if signs differ, 0 if same
        sign_penalty = sign_mismatch * (
            self.penalty_weight * torch.abs(y_pred - y_true) + self.extra_penalty
        )

        # Combine terms
        loss = loss_term + sign_penalty
        return loss
class ImprovedCustomLoss(nn.Module):
    def __init__(self, penalty_weight=0.1, sensitivity=3.0, large_error_weight=0.5):
        """
        Args:
        - penalty_weight: Initial penalty scale for opposite signs.
        - sensitivity: Controls the sensitivity for small errors.
        - large_error_weight: Additional weight for large errors to prevent saturation.
        """
        super(ImprovedCustomLoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.sensitivity = sensitivity
        self.large_error_weight = large_error_weight
        self.total_epochs = 200

    def forward(self, y_pred, y_true, epoch=1):
        # Loss Term: Sensitive to small errors
        loss_term = torch.arctan(self.sensitivity * torch.abs(y_pred - y_true))

        # Additional term for large errors to prevent saturation
        large_error_term = self.large_error_weight * (y_pred - y_true) ** 2

        # Penalty Term: Smooth penalty for opposite signs
        sign_penalty = self.penalty_weight * (1 - torch.sign(y_pred * y_true)) * torch.abs(y_pred - y_true)

        # Adjust penalty dynamically (scaling penalty_weight over time)
        dynamic_penalty_weight = self.penalty_weight * (epoch / self.total_epochs)
        sign_penalty *= dynamic_penalty_weight

        # Combine terms
        loss = loss_term + large_error_term + sign_penalty
        return torch.mean(loss)


# Main Execution
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define loss function
    # loss_fn = ImprovedCustomLoss(penalty_weight=0.1, sensitivity=5.0)
    # loss_fn = ImprovedCustomLoss(penalty_weight=0.1, sensitivity=5.0, large_error_weight=0.5)
    loss_fn = AdaptiveCustomLoss(penalty_weight=0.1, sensitivity=5.0, extra_penalty=5.0, max_penalty=20.0)
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