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
    
class PeptideRandomSampler(Sampler):
    """
    Creates a custom random sampler

    Attributes:
        data (np.ndarray): dataset to be sampled
        seed (int): random number generator seed
    """
    def __init__(self, dataset, seed):
        """
        Initializes PeptideRandomSampler
        """
        self.data = dataset
        self.seed = seed
        
    def __iter__(self):
        random.Random(self.seed).shuffle(self.data)
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)


## Network training function
def train(model, trainloader, learning_rate, epochs, device):
    """ Trains model using training data

    Args:
        model (PeptideNN): binding prediction model
        trainloader (DataLoader): training peptide data
        learning_rate (float): step size of each iteration
        device (torch.device): device on which torch.Tensor will be allocated

    Returns:
        Model optimizer

    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train() # Set training mode to update gradients
    for _ in range(epochs):  # loop over the dataset multiple times
        for _, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].to(torch.float).to(device)
            labels = data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(-1).float())
            loss.backward()
            optimizer.step()

    return optimizer


def save_model(model, fold, models_dir, configs_dir, optimizer, config):
    """ Saves model and configuration info to model directories

    Args:
        model (PeptideNN): trained binding prediction model
        fold (int): fold count #
        models_dir (os.PathLike): model output dir location
        configs_dir (os.PathLike): model config subdir path
        optimizer (optim.Adam): model optimizer
        config (dict): dictionary with fold's training details

    """

    time = str(datetime.datetime.now()).replace(" ","_")
    path_name = f'NN-time{time}-fold{fold}'
    model_path = os.path.join(models_dir, path_name+'.pt')
    config_path = os.path.join(configs_dir, path_name+'.json')

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)

    with open(config_path, 'w') as fp:
        json.dump(config, fp)

def evaluate(model, dataloader, replicates, device):
    """ Uses model to generate prediction with replicates for variance

    Args:
        model (PeptideNN): trained binding prediction model
        dataloader (DataLoader): peptide data
        replicates (int): number of replicates for prediction
        device (torch.device): device on which torch.Tensor will be allocated

    Returns:
        List of input peptides, list of target values, and list of predicted values

    """
    model.train() # Set Training mode to enable dropouts
    with torch.no_grad():
        input_lst, target_lst, prediction_lst = [], [], []

        # Iterate over the test data and generate predictions
        for _, data in enumerate(dataloader, 0):
            inputs = data[0].to(torch.float).to(device)
            labels = data[1].to(device)

            input_lst.append(inputs)
            target_lst.append(labels)

            # Iterate over replicates
            predictions = torch.zeros(inputs.shape[0], replicates)
            for j in range(0,replicates):
                outputs = model(inputs)
                logits = outputs.data
                predictions[:,j] = logits.squeeze()

            prediction_lst.append(predictions)

    # Combine data from epochs and return
    return input_lst, target_lst, prediction_lst