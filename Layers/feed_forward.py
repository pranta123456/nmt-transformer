import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_hidden, d_model, p=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.layer1 = nn.Linear(d_hidden, d_model)
        self.layer2 = nn.Linear(d_model, d_hidden)
        self.dropout = nn.Dropout(p=p)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)

        return x