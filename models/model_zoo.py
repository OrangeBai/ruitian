import os
import torch
from torch import nn as nn
from collections import OrderedDict


class BaseModel(nn.Module):
    def __init__(self, num_feat, num_uid, num_output):
        super().__init__()
        self.time_length = 7
        self.num_feat = num_feat
        self.num_uid = num_uid


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
    def __init__(self, num_feat, num_uid, num_output, blocks, width, bn=True, dropout=True):
        super().__init__(num_feat, num_uid, num_output)
        layers = [nn.Linear(self.num_feat, 1), nn.Flatten(start_dim=2), nn.Linear(self.num_uid, width)]

        for i in range(blocks):
            layers.append(
                LinearBlock(width, width, bn=bn, dropout=dropout)
            )

        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, dropout=True):
        super().__init__()

        self.LT = nn.Linear(in_channels, out_channels)
        self.BN = nn.LayerNorm([out_channels]) if bn else nn.Identity()
        self.DP = nn.Dropout(0.5) if dropout else nn.Identity()
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
