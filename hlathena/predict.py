"""Peptide prediction module"""
import os
import numpy as np
from typing import List
import torch
import pandas as pd
from hlathena.peptide_nn import PeptideNN
from hlathena.peptide_dataset import PeptideDataset


def predict(model_path: os.PathLike, peptides: List[str], dropout_rate: float = 0.1, replicates: int = 1):
    """
    Predict the scores of the given peptides using a saved PyTorch model.

    Args:
        model_path (os.PathLike): path to the saved PyTorch model
        peptides (List[str]): list of peptides to predict scores for
        dropout_rate (float): dropout rate to use during model prediction (default 0.1)
        replicates (int): number of replicates to perform for each peptide (default 1)

    Returns:
        pandas.DataFrame: dataframe containing the predicted scores for each peptide
        
    Raises:
        IndexError: No peptides submitted for prediction
    """
    
    # Check if any peptides were provided
    if not len(peptides):
        raise IndexError("No peptides submitted for prediction")
    
    # Determine device to use (GPU or CPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Reformat peptides as dataframe to be used as input to PeptideDataset
    pep_df = pd.DataFrame(peptides, columns=['seq'])
    peptide_data = PeptideDataset(pep_df)
    
    # Load the saved model
    model = PeptideNN(peptide_data.feature_dimensions, dropout_rate)

    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()

    input_lst, predictions_lst = [], []
    with torch.no_grad():
        for _, data in enumerate(peptide_data, 0):
            inputs = data.to(torch.float).to(device)
            inputs = torch.reshape(inputs, (1,-1))
            input_lst.append(inputs)
            model(inputs)
            predictions = torch.zeros(inputs.shape[0], replicates)
            for j in range(0,replicates):
                outputs = model(inputs)
                logits = outputs.data
                predictions[:,j] = logits.squeeze()

            predictions_lst.append(predictions)

    # Stack the input and prediction tensors into one tensor for easier processing
    input_lst = torch.vstack(input_lst).cpu()
    predictions_lst = torch.vstack(predictions_lst).cpu()

    # Convert the input tensor back into peptides and convert the prediction tensor to a list of scores
    input_lst = [peptide_data.decode_peptide(p) for p in input_lst]
    predictions_lst = [pred[0] for pred in predictions_lst.tolist()]
    
    # Create a dataframe with the input peptides and their corresponding predicted scores
    pred_df = pd.DataFrame(np.column_stack([input_lst, predictions_lst]), columns=['seq','score'])

    return pred_df

