""" Model training module """
import datetime
import json
import os
import random

import torch
import torch.nn.functional as F
from torch import nn, optim
from os.path import exists
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import Tensor
import math
import copy
from torch.utils.data import DataLoader
import logging
# import esm

from hlathena.peptide_transformer import (
    NoamOpt,
    get_std_opt,
    EarlyStopper,
    initialize_param,
    save_model
)


class PositionalEncoding_PyTorch(nn.Module): #from https://pytorch.org/tutorials/beginner/transformer_tutorial.html#:~:text=A%20sequence%20of%20tokens%20are,TransformerEncoderLayer.

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) #according to chatgpt, did not need to add "device = device" here since i have register_buffer 
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoderClassifier_Model1(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding_PyTorch(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.linear = nn.Linear(d_model, 1) #bring it back to dim 1 for sigmoid classifier (if using softmax, change this to 2)

    def forward(self, src, pooling_strat = "CLS", model_type = "esm", src_mask=None, src_key_padding_mask=None) -> Tensor:
        #change src (input) shape from [batch_size, seq_len, input_dim] to [seq_len, batch_size, input_dim] (latter is more common for pytorch built-in implementations)
        #src = src.permute(1, 0, 2) -- added batch_first=True so do not need this anymore
        
        if model_type == "esm": 
            #src = self.embedding(src) #add this back in when you generalize the model for other types
            src = self.pos_encoder(src)
        '''else:
            src = self.embedding(src) #add this back in when you generalize the model for other types
            src = self.pos_encoder(src)'''
        
        #run model
        #print("model")
        #print(src.shape)
        encoded_interim = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        #print(encoded_interim.shape)
        
        ## Three ways to do this: take the representation of the [CLS] token, take the mean pooling strategy, 
        # or take the raw values themselves (unsure if the last one is actually used in practice!!)
        if pooling_strat == "mean":
            #shape of encoded_interim is [batch_size, seq_len, features]
            encoded = encoded_interim.mean(dim=1) #changed dim = 0 to dim = 1 since that is now the seq_len dim
        elif pooling_strat == "CLS":
            #assuming CLS is the first token at position 0 (assuming it is the same as SOS token for our purposes - check with Wengong [TODO])
            # With batch_first=True, the CLS token for each sequence is at position 0 along the seq_len dimension, but since we're iterating over batch, index over each item in the batch.
            #shape of encoded_interim is [batch_size, seq_len, features]
            encoded = encoded_interim[:, 0, :]
        else:
            raise ValueError(f"Invalid pooling strategy: {pooling_strat}")
        #run classifier (bring it down to one neuron then use sigmoid classifier)
        logits = self.linear(encoded)
        return torch.sigmoid(logits)


def train(model, trainloader, learning_rate, epochs, device, valloader=None, lr_warmup=4000, patience=5, min_delta=0):
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
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = get_std_opt(model, warmup=lr_warmup)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    model.train() # Set training mode to update gradients
    for ep in range(epochs):  # loop over the dataset multiple times
        # Initialize train and validation loss
        train_epoch_loss = train_one_epoch(model, ep, trainloader, optimizer, criterion, device)

        if valloader is not None:
            val_epoch_loss = eval_one_epoch(model, ep, valloader, criterion, device)
            if early_stopper.early_stop(val_epoch_loss):
                logging.info(f"Stopping early; min_val_loss={early_stopper.min_validation_loss}, "
                             f"val_loss={val_epoch_loss}, "
                             f"patience={patience}, min_delta={min_delta}")
                break

    return optimizer

def train_one_epoch(model, ep, trainloader, optimizer, criterion, device):
    loss = 0
    for _, data in enumerate(trainloader, 0):
        # (concat, padding, label)
        pHLA = data[0].to(device)
        padding_mask = data[1].to(device)
        labels = data[2].to(device)
        
        # pep = data[0][0].to(device)
        # hla = data[0][1].to(device)
        # pep_mask = data[0][2].to(device)
        # hla_mask = data[0][3].to(device)
        # # pep_enumerated = data[0][0].to(device)
        # # pephla_enumerated = data[0][1].to(device)
        # # bos_tensor = data[0][2].to(device)
        # labels = data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # outputs = model(pHLA, padding_mask)
        # print(f"Padding mask shape pre-squeezing: {padding_mask.shape}")
        # print(f"Padding mask shape post-squeezing w/ extra squeeze: {padding_mask.squeeze(1).squeeze(1).squeeze(1).shape}")
        outputs = model(pHLA.squeeze().float(), src_key_padding_mask = padding_mask.squeeze(1).squeeze(1)).squeeze()

        batch_loss = criterion(outputs, labels.float())
        batch_loss.backward(retain_graph=True)

        before_lr = optimizer.optimizer.param_groups[0]['lr']
        optimizer.step()
        after_lr = optimizer.optimizer.param_groups[0]['lr']
        print("Step %d: Adam LR %.6f -> %.6f" % (optimizer._step, before_lr, after_lr))
        # logging.info("Epoch %d: Adam LR %.6f -> %.6f" % (ep, before_lr, after_lr))

        optimizer.step()

        loss += batch_loss.item() * labels.size(0)
    loss = loss / len(trainloader.sampler)
    print(f"Avg. train epoch {ep} loss: {loss}")
    logging.info(f"Avg. train epoch {ep} loss: {loss}")
    return loss


def eval_one_epoch(model, ep, dataloader, criterion, device): # TODO: optional replicates, no dropout/model.train() if no rep
    # criterion = nn.BCELoss()
    model.train() # Set Training mode to enable dropouts
    with torch.no_grad(): # Not doing backward pass, just return predictions w/ dropout
        loss = 0

        # Iterate over the test data and generate predictions
        for _, data in enumerate(dataloader, 0):
            pHLA = data[0].to(device)
            padding_mask = data[1].to(device)
            labels = data[2].to(device)
            
            # pep = data[0][0].to(device)
            # hla = data[0][1].to(device)
            # pep_mask = data[0][2].to(device)
            # hla_mask = data[0][3].to(device)
            # # pep_enumerated = data[0][0].to(device)
            # # pephla_enumerated = data[0][1].to(device)
            # # bos_tensor = data[0][2].to(device)
            # labels = data[1].to(device)

            # forward
            # outputs = model(pep, hla, pep_mask, hla_mask)
            outputs = model(pHLA.squeeze().float(), src_key_padding_mask = padding_mask.squeeze(1).squeeze(1)).squeeze()

            batch_loss = criterion(outputs, labels.float())
            loss += batch_loss.item() * labels.size(0)
    loss = loss / len(dataloader.sampler)
    print(f"Avg. validation epoch {ep} loss: {loss}")
    logging.info(f"Avg. validation epoch {ep} loss: {loss}")
    return loss


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
            pHLA = data[0].to(device)
            padding_mask = data[1].to(device)
            labels = data[2].to(device)
            indices = data[3].to(device)
            # pep = data[0][0].to(device)
            # hla = data[0][1].to(device)
            # pep_mask = data[0][2].to(device)
            # hla_mask = data[0][3].to(device)
            # # pep_enumerated = data[0][0].to(device)
            # # pephla_enumerated = data[0][1].to(device)
            # # bos_tensor = data[0][2].to(device)
            # labels = data[1].to(device)
            # indices = data[2].to(device)



            # input_lst.append(inputs)
            target_lst.append(labels)
            index_lst.append(indices)

            # Iterate over replicates
            predictions = torch.zeros(labels.shape[0], replicates)
            for j in range(0,replicates):
                # outputs = model(pep, hla, pep_mask, hla_mask)
                outputs = model(pHLA.squeeze().float(), src_key_padding_mask = padding_mask.squeeze(1).squeeze(1)).squeeze()
                logits = outputs.data
                predictions[:,j] = logits.squeeze() #torch.argmax(outputs.data, dim=1)

            prediction_lst.append(predictions)

    # Combine data from epochs and return
    return [], target_lst, index_lst, prediction_lst # TODO: remove inputs from return