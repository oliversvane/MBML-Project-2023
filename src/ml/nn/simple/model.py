import torch
import torch.nn as nn

class ConvNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(hidden_size*30, 50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x.transpose(1,2)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x