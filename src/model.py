import torch.nn as nn


class SeqForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

        # Use He(uniform) initialization for linear layer
        for name, param in self.fc.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.kaiming_uniform_(param)

    def forward(self, input):
        # Propagate the input trough lstm
        _, (hidden, _) = self.lstm(input)
        # Get the prediction for the next time step
        out = self.fc(hidden[-1, :, :])

        return out.view(-1, 1)
