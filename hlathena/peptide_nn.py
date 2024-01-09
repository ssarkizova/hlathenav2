""" Model training module """
import datetime
import json
import os
import random

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.sampler import Sampler
import logging

class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# ### Define Model
# nEpochs = 15 # 10
# batch_size = 5000   # number of training examples in batch training
# n_hidden_1 = 250 #400 #150 len-specific; tried 300; for pan-len 300 or 500; 400?
# dropout_rate = 0.0 #0.1
# patience_lr = 2    # reduce learning rate if optimization hasn't improved in 'patience_lr' epochs
# patience_es = 4    # early stop if not improvement after 'patience_es' epochs

class PeptideNN2(nn.Module):
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
        super(PeptideNN2, self).__init__()

        self.fc1 = nn.Linear(feat_dims, 250)
        self.fc2 = nn.Linear(250, 1)
        self.dropout = torch.nn.Dropout(p=dropout_rate)


    def forward(self, inputs):
        """ Fully connected neural network """
        x = torch.flatten(inputs, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x)) # making autocast happy, combine the sigmoid with the loss
        return x


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
        super().__init__()
        self.data = dataset
        self.seed = seed

    def __iter__(self):
        random.Random(self.seed).shuffle(self.data)
        return iter(self.data)

    def __len__(self):
        return len(self.data)


# Network training function
def train(model, trainloader, learning_rate, epochs, device, valloader=None, patience=5, min_delta=0):
    """ Trains model using training data

    Args:
        model (PeptideNN): binding prediction model
        trainloader (DataLoader): training peptide data
        learning_rate (float): step size of each iteration
        device (torch.device): device on which torch.Tensor will be allocated
        epochs (int): number of training epochs
        valloader (DataLoader): validation set data
        patience (int): early stopping patience
        min_delta (float): early stopping minimum loss difference

    Returns:
        Model optimizer

    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    model.train() # Set training mode to update gradients
    for e in range(epochs):  # loop over the dataset multiple times
        # Initialize train and validation loss
        train_epoch_loss = 0

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

            train_epoch_loss += loss.item()*inputs.size(0)

        train_epoch_loss = train_epoch_loss/len(trainloader.sampler)
        logging.info(f"Avg. train epoch {e} loss: {train_epoch_loss}")
        if valloader is not None:
            val_epoch_loss = eval_loss(model, valloader, device)/len(valloader.sampler)
            logging.info(f"Avg. validation epoch {e} loss: {val_epoch_loss}")
            if early_stopper.early_stop(val_epoch_loss):
                logging.info(f"Stopping early; min_val_loss={early_stopper.min_validation_loss}, val_loss={val_epoch_loss}, patience={patience}, min_delta={min_delta}")
                break

    return optimizer


def eval_loss(model, dataloader, device): # TODO: optional replicates, no dropout/model.train() if no rep
    """ Uses model to generate prediction with replicates for variance

    Args:
        model (PeptideNN): trained binding prediction model
        dataloader (DataLoader): peptide data
        replicates (int): number of replicates for prediction
        device (torch.device): device on which torch.Tensor will be allocated

    Returns:
        Loss value

    """
    criterion = nn.BCELoss()

    model.train() # Set Training mode to enable dropouts
    with torch.no_grad(): # Not doing backward pass, just return predictions w/ dropout
        eval_loss = 0

        # Iterate over the test data and generate predictions
        for _, data in enumerate(dataloader, 0): # add dataset label to get item tuple?
            inputs = data[0].to(torch.float).to(device)
            labels = data[1].to(device)

            # forward
            outputs = model(inputs)

            loss = criterion(outputs, labels.unsqueeze(-1).float())
            eval_loss += loss.item()*inputs.size(0)

    return eval_loss


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


# TESTING W/ LABEL
        # CHANGE: predict with reps
def evaluate(model, dataloader, replicates, device): # TODO: optional replicates, no dropout/model.train() if no rep
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
    with torch.no_grad(): # Not doing backward pass, just return predictions w/ dropout
        input_lst, target_lst, index_lst, prediction_lst = [], [], [], []

        # Iterate over the test data and generate predictions
        for _, data in enumerate(dataloader, 0): # add dataset label to get item tuple?
            inputs = data[0].to(torch.float).to(device)
            labels = data[1].to(device)
            indices = data[2].to(device)

            input_lst.append(inputs)
            target_lst.append(labels)
            index_lst.append(indices)

            # Iterate over replicates
            predictions = torch.zeros(inputs.shape[0], replicates)
            for j in range(0,replicates):
                outputs = model(inputs)
                logits = outputs.data
                predictions[:,j] = logits.squeeze()

            prediction_lst.append(predictions)

    # Combine data from epochs and return
    return input_lst, target_lst, index_lst, prediction_lst


# predict
"""
Assumes input is peptide_dataset_2 (i.e. returns encoded pep but no target). Already w/ features set.
    (?? Alternatively, could just ignore the return binding and have the class return tensor, -1)
Do we want to predict across reps? And find avg?
"""
def predict(model, peptide_data, replicates, device): # TODO: are these replicates changing at all?
    input_lst, predictions_lst = [], []
    with torch.no_grad():
        for _, data in enumerate(peptide_data, 0):
            inputs = data[0].to(torch.float).to(device)
            input_lst.append(inputs)

            predictions = torch.zeros(inputs.shape[0], replicates)
            for j in range(0,replicates):
                outputs = model(inputs)
                logits = outputs.data
                predictions[:,j] = logits.squeeze()

            predictions_lst.append(predictions)

    input_lst = [peptide_data.decode_peptide(p) for p in input_lst] # also return hla as column
    return zip(input_lst, predictions_lst)