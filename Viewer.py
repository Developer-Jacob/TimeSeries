import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class CustomLinearLoss(nn.Module):
    def __init__(self, penalty_factor=5, sensitivity_factor=0.5):
        super(CustomLinearLoss, self).__init__()
        self.penalty_factor = penalty_factor
        self.sensitivity_factor = sensitivity_factor

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        # Base loss with tanh
        base_loss = torch.tanh(error ** 2)

        # Cap penalty at penalty_factor when signs are opposite
        penalty_tensor = torch.full_like(error, self.penalty_factor)  # Ensure same dtype and device
        sign_penalty = torch.where(error < 0, torch.minimum(penalty_tensor, penalty_tensor * torch.abs(error)),
                                   torch.zeros_like(error))

        # Adjust sensitivity: Gradual increase for small errors, steeper for larger errors
        sensitivity = 1 - torch.exp(-self.sensitivity_factor * torch.abs(error))

        # Combine all components
        loss = base_loss * sensitivity + sign_penalty
        return torch.mean(loss)# Return element-wise loss for visualization


# Visualization Function
def visualize_custom_loss(penalty_factor=5, sensitivity_factor=0.5):
    loss_fn = CustomLinearLoss(penalty_factor, sensitivity_factor)
    y_pred = torch.linspace(-20, 30, 500)  # Adjust range to -20 to 30 for visualization
    y_true = torch.zeros_like(y_pred)  # Simulate true values as 0 for visualization

    # Calculate loss for each value individually
    loss_values = []
    for pred in y_pred:
        loss = loss_fn(pred.unsqueeze(0), torch.tensor([0.0]))  # Compute loss for each prediction
        loss_values.append(loss.item())

    plt.figure(figsize=(8, 6))
    plt.plot(y_pred.numpy(), loss_values, label="Custom Loss with Gradual and Steep Sensitivity")
    plt.title("Custom Loss Function Visualization with Sensitivity and Penalty")
    plt.xlabel("Prediction (y_pred)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main Execution
if __name__ == "__main__":
    print("Visualizing the custom loss function...")
    visualize_custom_loss(penalty_factor=5, sensitivity_factor=0.5)
