import torch
import torch.nn as nn

class WeatherLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_output = 1):
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_output)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])  
        return out
