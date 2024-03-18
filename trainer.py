import torch
from save_load import load_model, save
from tqdm import tqdm
from torch import nn

class Trainer:
    def __init__(self, input_window, output_window, path):
        self.input_window = input_window
        self.output_window = output_window
        self.path = path

    def train(self, epoch, train_loader):
        device = self.gpu_device()
        criterion = nn.MSELoss()
        epoch_start, model, optimizer = load_model(self.path, self.input_window, self.output_window)
        model.train()

        progress = tqdm(range(epoch_start, epoch))

        for i in progress:
            total_loss = 0.0
            for (inputs, outputs) in train_loader:
                optimizer.zero_grad()
                src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
                result = model(inputs.float().to(device), src_mask)
                loss = criterion(result, outputs.float().to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss
            save(self.path, i, model, optimizer)
            progress.set_description("loss: {:0.6f}".format(total_loss.cpu().item() / len(train_loader)))

    def evaluate(self, input_data):
        device = self.gpu_device()
        _, model, optimizer = load_model(self.path, self.input_window, self.output_window)
        model.eval()
        input_data = input_data.float().to(device)
        src_mask = model.generate_square_subsequent_mask(input_data.shape[1]).to(device)
        predictions = model(input_data, src_mask)
        return predictions.detach().cpu().numpy()

    @staticmethod
    def gpu_device():
        return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
