import torch
import torch.nn.functional as F
from torch import nn, optim
from os.path import exists
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import copy
from torch.utils.data import DataLoader
import logging
import json
import datetime
import os

from hlathena.peptide_transformer import (
    Embeddings,
    Sigmoid_Classifier,
    PositionalEncoding,
    MultiHeadedAttention,
    PositionwiseFeedForward,
    Encoder,
    EncoderLayer,
    Encoder_NoClassifier,
    clones,
    SublayerConnection,
    LayerNorm,
    NoamOpt,
    get_std_opt,
    EarlyStopper,
    initialize_param
)


class OverallModel_4(nn.Module):
    def __init__(self, src_vocab, d_model, h, N=6, d_ff=2048, dropout=0.1, model_type="encode"):
        super(OverallModel_4, self).__init__()
        self.d_model = d_model
        self.model_type = model_type
        self.peptide_transformer_block = TransformerBlock_SelfAttention(src_vocab, N=N, d_model=d_model, d_ff=d_ff, h=h,
                                                                        dropout=dropout)
        self.hla_transformer_block = TransformerBlock_SelfAttention(src_vocab, N=N, d_model=d_model, d_ff=d_ff, h=h,
                                                                    dropout=dropout)
        self.cross_attention_transformer_block = TransformerBlock_CrossAttention(src_vocab, N=N, d_model=d_model,
                                                                                 d_ff=d_ff, h=h, dropout=dropout)

    def forward(self, peptide, hla, mask):  # mask should be the peptide mask
        peptide_attention = self.peptide_transformer_block(data=peptide, model_type=self.model_type, padding_mask=mask)
        hla_attention = self.hla_transformer_block(data=hla, model_type=self.model_type)

        # value and key come from peptide and query comes from hla; mask is from peptide
        output = self.cross_attention_transformer_block(hla_attention, peptide_attention, mask)

        return output


class TransformerBlock_CrossAttention(nn.Module):
    def __init__(self, src_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(TransformerBlock_CrossAttention, self).__init__()
        c = copy.deepcopy
        embedding = Embeddings(d_model, src_vocab)
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        # self attention blocks for embed and encode
        self.cross_attention_transformer = EncoderClassifier_CrossAttention(
            Encoder_CrossAttention(EncoderLayer_CrossAttention(d_model, c(attn), c(ff), dropout), N),
            Sigmoid_Classifier(d_model))

    def forward(self, hla, peptide, mask=None):
        return self.cross_attention_transformer(hla, peptide, mask)


class TransformerBlock_SelfAttention(nn.Module):
    def __init__(self, src_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(TransformerBlock_SelfAttention, self).__init__()
        c = copy.deepcopy
        embedding = Embeddings(d_model, src_vocab)
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        # self attention blocks for embed and encode
        self.embed_transformer_model = Encoder_NoClassifier(nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                                                            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N))

        # note that the d_model param for PositionalEncoding is equal to peptide_dim here bc with one-hot encoding, the dim of each encoding = # encodings
        self.encode_transformer_model = Encoder_NoClassifier(PositionalEncoding(src_vocab, dropout),
                                                             Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N))

    def forward(self, data, model_type,
                padding_mask=None):  # bos = beg of seq; bos_data should just be a series of 0's (one 0 for every input seq)
        if model_type == "embed":
            return self.embed_transformer_model(data, padding_mask)
        elif model_type == "encode":
            return self.encode_transformer_model(data, padding_mask)
        else:
            print("model_type must be 'embed' or 'encode'")


class Encoder_CrossAttention(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder_CrossAttention, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, y, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, y, mask)
        return self.norm(x)


class EncoderLayer_CrossAttention(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, cross_attn, feed_forward, dropout):
        super(EncoderLayer_CrossAttention, self).__init__()
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, y, mask):  # value and key come from peptide and query comes from hla; mask is from peptide
        x = self.sublayer[0](x, lambda x: self.cross_attn(x, y, y, mask))  # query, key, value --> hla = x, peptide = y
        return self.sublayer[1](x, self.feed_forward)


class EncoderClassifier_CrossAttention(nn.Module):
    def __init__(self, encoder, classifier):
        super(EncoderClassifier_CrossAttention, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, hla, peptide, mask=None):  # hla = src, peptide = tgt
        return self.classifier(self.encoder(hla, peptide, mask))


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
        pep = data[0][0].to(device)
        hla = data[0][1].to(device)
        pep_mask = data[0][2].to(device)
        hla_mask = data[0][3].to(device)
        # pep_enumerated = data[0][0].to(device)
        # pephla_enumerated = data[0][1].to(device)
        # bos_tensor = data[0][2].to(device)
        labels = data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(pep, hla, pep_mask)

        batch_loss = criterion(outputs, labels.unsqueeze(-1).float())
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
            pep = data[0][0].to(device)
            hla = data[0][1].to(device)
            pep_mask = data[0][2].to(device)
            hla_mask = data[0][3].to(device)
            # pep_enumerated = data[0][0].to(device)
            # pephla_enumerated = data[0][1].to(device)
            # bos_tensor = data[0][2].to(device)
            labels = data[1].to(device)

            # forward
            outputs = model(pep, hla, pep_mask)

            batch_loss = criterion(outputs, labels.unsqueeze(-1).float())
            loss += batch_loss.item() * labels.size(0)
    loss = loss / len(dataloader.sampler)
    print(f"Avg. validation epoch {ep} loss: {loss}")
    logging.info(f"Avg. validation epoch {ep} loss: {loss}")
    return loss


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
            pep = data[0][0].to(device)
            hla = data[0][1].to(device)
            pep_mask = data[0][2].to(device)
            hla_mask = data[0][3].to(device)
            # pep_enumerated = data[0][0].to(device)
            # pephla_enumerated = data[0][1].to(device)
            # bos_tensor = data[0][2].to(device)
            labels = data[1].to(device)
            indices = data[2].to(device)



            # input_lst.append(inputs)
            target_lst.append(labels)
            index_lst.append(indices)

            # Iterate over replicates
            predictions = torch.zeros(labels.shape[0], replicates)
            for j in range(0,replicates):
                outputs = model(pep, hla, pep_mask)
                logits = outputs.data
                predictions[:,j] = logits.squeeze() #torch.argmax(outputs.data, dim=1)

            prediction_lst.append(predictions)

    # Combine data from epochs and return
    return [], target_lst, index_lst, prediction_lst # TODO: remove inputs from return