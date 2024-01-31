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
import math
import copy
from torch.utils.data import DataLoader
import logging

from hlathena.definitions import AA_MAPPING


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# old one
'''class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        "Pass the input through each layer in turn."
        if mask == None:
            for layer in self.layers:
                x = layer(x)
        else: 
            for layer in self.layers:
                x = layer(x, mask)
        return self.norm(x)'''


# new one
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# old one
'''class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        "Follow the connections of the encoder layer."
        if mask == None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        else: 
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)'''


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    '''print("###############################")
    print(p_attn.shape)
    print(p_attn[0][0])
    print("###############################")
    print(p_attn[0][1])
    print("###############################")
    print(p_attn[0][2])
    print("###############################")'''
    return torch.matmul(p_attn, value), p_attn


# old one
'''class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)'''


# new one
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for lin, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = (x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k))

        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


'''class EncoderClassifier(nn.Module):
    """
    An Encoder-Only architecture for binary classification.
    """
    def __init__(self, src_embed, encoder, classifier):
        super(EncoderClassifier, self).__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, src):
        return self.classifier(self.encoder(self.src_embed(src)))'''


class EncoderClassifier(nn.Module):
    """
    An Encoder-Only architecture for binary classification.
    """

    def __init__(self, src_embed, encoder, classifier):
        super(EncoderClassifier, self).__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, src, mask=None):
        return self.classifier(self.encoder(self.src_embed(src), mask))


class Encoder_NoClassifier(nn.Module):
    """
    An Encoder-Only architecture for binary classification.
    """

    def __init__(self, src_embed, encoder):
        super(Encoder_NoClassifier, self).__init__()
        self.src_embed = src_embed
        self.encoder = encoder

    def forward(self, src, mask=None):
        return self.encoder(self.src_embed(src), mask)


class EncoderClassifier_NoEmbedPos(nn.Module):
    """
    An Encoder-Only architecture for binary classification.
    """

    def __init__(self, encoder, classifier):
        super(EncoderClassifier_NoEmbedPos, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, src, mask=None):
        return self.classifier(self.encoder(src, mask))


class Sigmoid_Classifier(nn.Module):
    "A simple classification layer for binary classification."

    def __init__(self, d_model):
        super(Sigmoid_Classifier, self).__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x[:, 0]))


class Softmax_Classifier(nn.Module):
    "A simple classification layer for multi-class classification."

    def __init__(self, d_model, num_classes):
        super(Softmax_Classifier, self).__init__()
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Apply softmax to the output of the linear layer
        # Note: Softmax is applied on the second dimension, assuming x is of shape (batch_size, num_classes)
        return F.softmax(self.linear(x[:, 0]), dim=1)


def make_model(src_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, num_classes=2):
    "Construct an encoder-only model."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = Encoder_NoClassifier(
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N))
    # Parameter initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def make_model_softmax(src_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, num_classes=2):
    "Construct an encoder-only model."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderClassifier(
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Softmax_Classifier(d_model, num_classes))
    # Parameter initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def make_model_softmax_noembedpos(N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, num_classes=2):
    "Construct an encoder-only model."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderClassifier_NoEmbedPos(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Softmax_Classifier(d_model, num_classes))
    # Parameter initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def make_model_sigmoid(src_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Construct an encoder-only model."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderClassifier(
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Sigmoid_Classifier(d_model))
    # Parameter initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def make_model_sigmoid_noembedpos(N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Construct an encoder-only model."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderClassifier_NoEmbedPos(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Sigmoid_Classifier(d_model))
    # Parameter initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#######################################
# run this after you have instantiated the model but before you run the model on data
def initialize_param(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


class OverallModel(nn.Module):
    def __init__(self, src_vocab1, src_vocab2, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, num_classes=2):
        super(OverallModel, self).__init__()

        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        self.peptide_transformer = Encoder_NoClassifier(nn.Sequential(Embeddings(d_model, src_vocab1), c(position)),
                                                        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N))
        # anchor block
        self.pephla_transformer = Encoder_NoClassifier(nn.Sequential(Embeddings(d_model, src_vocab2), c(position)),
                                                       Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N))
        self.final_transformer = EncoderClassifier_NoEmbedPos(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Softmax_Classifier(d_model, num_classes))

        self.bos_token_layer = nn.Sequential(Embeddings(d_model, src_vocab1), c(position))
        # self.peptide_transformer = make_model(src_vocab1, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)
        # self.pephla_transformer = make_model(src_vocab2, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)
        # self.final_transformer = make_model_softmax_noembedpos(N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)

        # question: create the padding mask here or outside of this?

    def forward(self, pep_data, padding_token, pephla_data,
                bos_data):  # bos = beg of seq; bos_data should just be a series of 0's (one 0 for every input seq)

        padding_mask = (pep_data != padding_token).unsqueeze(1)

        # going to comment this out since I don't want to re-initialize every time the forward loop is run, but also unsure if I need to initialize the param for each transformer separately or if they will all be initialized correctly if i initialize once for the OverallModel
        # initialize_param(self.peptide_transformer)
        # question: unsure where to put this:
        # self.peptide_transformer.train()
        pep_outputs = self.peptide_transformer(pep_data, padding_mask)

        # commented this out for same reason as above
        # initialize_param(self.pephla_transformer)
        # question: unsure where to put this:
        # self.pephla_transformer.train()
        pephla_outputs = self.pephla_transformer(pephla_data)

        bos_outputs = self.bos_token_layer(bos_data)

        # concat outputs with bos_token
        concat_outputs = torch.cat((bos_outputs, pep_outputs, pephla_outputs), dim=1)

        # question: does this go here haha (the initialize param)
        initialize_param(self.final_transformer)
        # self.final_transformer.train()
        return self.final_transformer(concat_outputs)


##################################


######## OPTIMIZER
class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
        pep_enumerated = data[0][0].to(device)
        pephla_enumerated = data[0][1].to(device)
        bos_tensor = data[0][2].to(device)
        labels = data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(pep_enumerated, AA_MAPPING['-'], pephla_enumerated, bos_tensor)

        batch_loss = criterion(outputs, F.one_hot(labels).float())
        batch_loss.backward(retain_graph=True)
        optimizer.step()

        loss += batch_loss.item() * labels.size(0)
    loss = loss / len(trainloader.sampler)
    logging.info(f"Avg. train epoch {ep} loss: {loss}")
    return loss


def eval_one_epoch(model, ep, dataloader, criterion, device): # TODO: optional replicates, no dropout/model.train() if no rep
    # criterion = nn.BCELoss()
    model.train() # Set Training mode to enable dropouts
    with torch.no_grad(): # Not doing backward pass, just return predictions w/ dropout
        loss = 0

        # Iterate over the test data and generate predictions
        for _, data in enumerate(dataloader, 0):
            pep_enumerated = data[0][0].to(device)
            pephla_enumerated = data[0][1].to(device)
            bos_tensor = data[0][2].to(device)
            labels = data[1].to(device)

            # forward
            outputs = model(pep_enumerated, AA_MAPPING['-'], pephla_enumerated, bos_tensor)

            batch_loss = criterion(outputs, F.one_hot(labels).float())
            loss += batch_loss.item() * labels.size(0)
    loss = loss / len(dataloader.sampler)
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
            pep_enumerated = data[0][0].to(device)
            pephla_enumerated = data[0][1].to(device)
            bos_tensor = data[0][2].to(device)
            labels = data[1].to(device)
            # inputs = torch.cat((data[0]), 1).to(torch.float).to(device)
            # inputs = data[0].to(torch.float).to(device)
            # labels = data[1].to(device)
            indices = data[2].to(device)

            # input_lst.append(inputs)
            target_lst.append(labels)
            index_lst.append(indices)

            # Iterate over replicates
            predictions = torch.zeros(labels.shape[0], replicates)
            for j in range(0,replicates):
                outputs = model(pep_enumerated, AA_MAPPING['-'], pephla_enumerated, bos_tensor)
                # logits = outputs.data
                predictions[:,j] = torch.argmax(outputs.data, dim=1)

            prediction_lst.append(predictions)

    # Combine data from epochs and return
    return [], target_lst, index_lst, prediction_lst # TODO: remove inputs from return
