""" Model training module """
import datetime
import json
import os
import random

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.sampler import Sampler


class PeptideNN(nn.Module):
    """
    Creates a neural network model

    Attributes:
        fc1 (nn.Linear): neural net layer 1
        dropout (nn.Dropout): dropout layer which randomly zeroes some inputs
        fc2 (nn.Linear): neural net layer 2
    """
    def __init__(self, feat_dims, dropout_rate):
        """
        Initializes neural network
        """
        super(PeptideNN, self).__init__()

        self.fc1 = nn.Linear(feat_dims, feat_dims)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(feat_dims, 1)

    def forward(self, x):
        """ Fully connected neural network """
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension # TODO: check if this is necessary for training
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x)) # making autocast happy, combine the sigmoid with the loss
        return x