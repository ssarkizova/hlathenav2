import os
import numpy as np
from typing import List
import torch
import pandas as pd
from hlathena.peptide_nn import PeptideNN
from hlathena.peptide_dataset import PeptideDataset

"""
Assumes input is peptide_dataset_2 (i.e. returns encoded pep but no target). Already w/ features set.
    (?? Alternatively, could just ignore the return binding and have the class return tensor, -1)
Do we want to predict across reps? And find avg?
"""
def predict(model_path: os.PathLike, peptides: List[str], dropout_rate: float = 0.1, replicates: int = 1):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # reformat peptides as dataframe to appease peptide_dataset init
    pep_df = pd.DataFrame(peptides, columns=['seq'])
    peptide_data = PeptideDataset(pep_df)
    
    # Loading the saved model
    model = PeptideNN(peptide_data.feature_dimensions, dropout_rate)

    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()

    input_lst, predictions_lst = [], []
    with torch.no_grad():
        for _, data in enumerate(peptide_data, 0):
            # print(_,data)
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

    input_lst = torch.vstack(input_lst).cpu()
    predictions_lst = torch.vstack(predictions_lst).cpu()

    input_lst = [peptide_data.decode_peptide(p) for p in input_lst]
    predictions_lst = [pred[0] for pred in predictions_lst.tolist()]
    pred_df = pd.DataFrame(np.column_stack([input_lst, predictions_lst]), columns=['seq','score'])

    return pred_df