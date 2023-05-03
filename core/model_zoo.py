import torch
from torch import nn as nn
from torch.nn.modules.rnn import LSTM


class BaseModel(nn.Module):
    def __init__(self, num_feat, num_uid, num_output):
        super().__init__()
        self.time_length = 7
        self.num_feat = num_feat
        self.num_uid = num_uid
        self.num_output = num_output


class LinearRegression(BaseModel):
    def __init__(self, num_feat, num_uid, num_output):
        super().__init__(num_feat, num_uid, num_output)
        self.fc1 = nn.Linear(self.num_feat, 1)
        self.fc2 = nn.Linear(self.num_uid, self.num_output)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(len(x), self.time_length, -1)
        x = self.fc2(x)
        return x


class FCDropout(BaseModel):
    def __init__(self, num_feat, num_uid, num_output, num_blocks, width, bn=True, dropout=0.0):
        super().__init__(num_feat, num_uid, num_output)
        layers = [nn.Linear(self.num_feat, 1), nn.Flatten(start_dim=2), nn.Linear(self.num_uid, width)]

        for i in range(blocks):
            layers.append(
                LinearBlock(width, width, bn=bn, dropout=dropout)
            )

        layers.append(nn.Linear(width, self.num_output))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LSTMBase(BaseModel):
    def __init__(self, num_feat, num_uid, num_output, num_blocks, num_layers, width, batch_size, dropout=0.0):
        super().__init__(num_feat, num_uid, num_output)
        self.num_blocks = num_blocks
        kwargs = {'hidden_size': width, 'num_layers': num_layers, 'batch_first': True, 'dropout': dropout}
        self.layer0 = nn.Sequential(*[nn.Linear(self.num_uid, 1), nn.Flatten(start_dim=2)])
        self.layer1 = LSTM(input_size=num_feat, **kwargs)
        self.layer2 = nn.Sequential(*[LSTM(input_size=width, **kwargs) for _ in range(num_blocks)])
        self.fc = nn.Linear(width, self.num_output)

        self.hidden_shape = (num_layers, batch_size, width)

    def _init_hidden(self, device):
        return torch.randn(self.hidden_shape, device=device)

    def forward(self, x):
        x = x.transpose(2, 3)
        x = self.layer0(x)

        hidden1 = (self._init_hidden(x.device), self._init_hidden(x.device))
        hidden2 = [(self._init_hidden(x.device), self._init_hidden(x.device)) for _ in range(self.num_blocks)]
        x, hidden1 = self.layer1(x, hidden1)
        for i in range(self.num_blocks):
            x, hidden2[i] = self.layer2[i](x, hidden2[i])
        x = self.fc(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, dropout=0.0):
        super().__init__()

        self.LT = nn.Linear(in_channels, out_channels)
        self.BN = nn.LayerNorm([out_channels]) if bn else nn.Identity()
        self.DP = nn.Dropout(dropout)
        self.Act = nn.LeakyReLU()

    def forward(self, x):
        x = self.LT(x)
        x = self.BN(x)
        x = self.Act(x)
        return x


def set_model(model, **kwargs):
    if model == 'linear':
        return LinearRegression(**kwargs)
    elif model == 'fc_dropout':
        return FCDropout(**kwargs)
    elif model == 'LSTM':
        return LSTMBase(**kwargs)
