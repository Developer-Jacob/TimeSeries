import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np


class StackedAutoEncoder:
    def __init__(self, n_epoch=100):
        self.auto1 = None
        self.auto2 = None
        self.auto3 = None
        self.auto4 = None

        self.num_hidden_1 = 10
        self.num_hidden_2 = 10
        self.num_hidden_3 = 10
        self.num_hidden_4 = 10
        self.n_epoch = n_epoch

    def forward(self, input_value):
        if self.auto1 is None:
            self.auto1 = CnnAutoEncoder(input_value.shape[1], self.num_hidden_1)
        self.auto1.train()
        self.auto1.fit(input_value, n_epoch=self.n_epoch)

        inputs = torch.autograd.Variable(torch.from_numpy(input_value.astype(np.float32)))

        if self.auto2 is None:
            self.auto2 = CnnAutoEncoder(self.num_hidden_1, self.num_hidden_2)
        self.auto2.train()
        auto1_out = self.auto1.encoder(inputs).data.numpy()
        self.auto2.fit(auto1_out, n_epoch=self.n_epoch)

        if self.auto3 is None:
            self.auto3 = CnnAutoEncoder(self.num_hidden_2, self.num_hidden_3)
        self.auto3.train()
        auto1_out = torch.autograd.Variable(torch.from_numpy(auto1_out.astype(np.float32)))

        auto2_out = self.auto2.encoder(auto1_out).data.numpy()
        self.auto3.fit(auto2_out, n_epoch=self.n_epoch)

        if self.auto4 is None:
            self.auto4 = CnnAutoEncoder(self.num_hidden_3, self.num_hidden_4)
        self.auto4.train()
        auto2_out = torch.autograd.Variable(torch.from_numpy(auto2_out.astype(np.float32)))
        auto3_out = self.auto3.encoder(auto2_out).data.numpy()
        self.auto4.fit(auto3_out, n_epoch=self.n_epoch)

    def encoded_data(self, input_value):
        self.auto1.eval()
        self.auto2.eval()
        self.auto3.eval()
        self.auto4.eval()

        input_value = torch.autograd.Variable(torch.from_numpy(input_value.astype(np.float32)))
        result = self.auto4.encoder(self.auto3.encoder(self.auto2.encoder(self.auto1.encoder(input_value))))
        return result.detach().numpy()

class CnnAutoEncoder(nn.Module):
    def __init__(
            self,
            feature_dim,
            layer_size=10,
            sparsity_target=0.05,
            sparsity_weight=0.2,
            lr=0.001,
            weight_decay=0.0
    ):
        super(CnnAutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Linear(feature_dim, layer_size),
            nn.ReLU()
        )
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(layer_size, feature_dim)
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.l1_loss = torch.nn.L1Loss(size_average=False)
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)

    def forward(self, x):
        # x = x.transpose(1, 2)
        x = self.encoder(x)
        # x[x == float('nan')] = 0
        mean = torch.mean(x, dim=0)
        sparsity_loss = torch.sum(self.kl_divergence(self.sparsity_target, mean))
        x = self.decoder(x)
        # x = x.transpose(1, 2)
        return x, sparsity_loss

    def kl_divergence(self, p, q):
        value = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
        value[value == float('Inf')] = 0
        return value # Kullback Leibler divergence

    def fit(self, x, n_epoch=10, batch_size=64, en_shuffle=False):
        for epoch in range(n_epoch):
            if en_shuffle:
                print("Data Shuffled")
                # X = sklearn.utils.shuffle(X)
            for local_step, X_batch in enumerate(self.gen_batch(x, batch_size)):
                inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.float32)))
                outputs, sparsity_loss = self.forward(inputs)

                l1_loss = self.l1_loss(outputs, inputs)
                loss = l1_loss + self.sparsity_weight * sparsity_loss
                self.optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                self.optimizer.step()  # apply gradients
                if local_step % 50 == 0:
                    print("Epoch %d/%d | Step %d/%d | train loss: %.4f | l1 loss: %.4f | sparsity loss: %.4f"
                          % (epoch + 1, n_epoch, local_step, len(x) // batch_size,
                             loss.item(), l1_loss.item(), sparsity_loss.item()))

    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i: i + batch_size]