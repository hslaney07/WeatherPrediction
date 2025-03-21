import numpy as np
import torch
import torch.nn as nn

class SinePredictor(nn.Module):
    def __init__(self):
        super(SinePredictor, self).__init__()
        self.w = 2 * torch.pi / 365  # Frequency
        self.d = nn.Parameter(torch.tensor(0.0))  # Phase shift
        self.b = nn.Parameter(torch.tensor(0.0))  # Bias

    def forward(self, t):
        return torch.sin(self.w * t + self.d) + self.b