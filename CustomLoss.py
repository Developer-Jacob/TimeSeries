import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, penalty_weight=0.5):
        super(CustomLoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.max_value = 10
        self.alpha = 0.1

    def forward(self, y_pred, y_true):
        # Loss Term: arctan or tanh for small-value sensitivity
        y_pred_clamped = torch.clamp(y_pred, -self.max_value, self.max_value)

        # Loss Term (Log-scaled)
        loss_term = torch.log(1 + torch.abs(y_pred_clamped - y_true))

        # Penalty Term (Dynamic Penalty based on Error)
        dynamic_penalty = self.penalty_weight / (1 + torch.abs(y_pred_clamped - y_true))
        sign_penalty = dynamic_penalty * (1 - torch.sign(y_pred * y_true))

        # Combine Loss with Weighted Average
        loss = loss_term + self.alpha * sign_penalty
        return torch.mean(loss)
    def forward_origin(self, y_pred, y_true):

        # Loss Term: arctan or tanh for small-value sensitivity
        loss_term = torch.tanh(2 * torch.abs(y_pred - y_true))

        # Penalty Term: Large penalty for opposite signs
        sign_penalty = self.penalty_weight * (1 - torch.sign(y_pred * y_true))

        # Combine terms
        loss = loss_term + sign_penalty
        return torch.mean(loss)

if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt

    # Define the custom loss function
    def custom_loss(y_pred, y_true, penalty_weight=10.0):
        # Loss Term: arctan for small-value sensitivity
        loss_term = torch.tanh(2 * torch.abs(y_pred - y_true))
        # Penalty Term: Large penalty for opposite signs
        sign_penalty = penalty_weight * (1 - torch.sign(y_pred * y_true))
        # sign_penalty = penalty_weight * torch.abs(y_true - y_pred.sign()) ** 2

        # Combine terms
        return loss_term + sign_penalty


    # Generate data for graph
    y_pred = torch.linspace(-3, 3, 300)  # Predicted values
    y_true_positive = torch.tensor(1.0)  # Positive true value
    y_true_negative = torch.tensor(-1.0)  # Negative true value
    print(y_pred)
    print(y_true_positive)
    print(y_true_negative)
    # Compute loss for positive and negative true values
    loss_positive = custom_loss(y_pred, y_true_positive, penalty_weight=0.3).detach().numpy()
    # loss_negative = custom_loss(y_pred, y_true_negative, penalty_weight=10.0).detach().numpy()

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_pred.numpy(), loss_positive, label="True Value: +1.0", color="blue")
    # plt.plot(y_pred.numpy(), loss_negative, label="True Value: -1.0", color="red")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title("Custom Loss Function Visualization")
    plt.xlabel("Predicted Value (y_pred)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()
