
import logging
import os
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
import importlib_resources
import sklearn.preprocessing

from hlathena import data
from hlathena.definitions import AMINO_ACIDS
from hlathena.peptide_dataset import PeptideDataset


def PCA_numpy_SVD(X, rowvar=False):
    """Computes the PCA of a matrix using SVD.

    Args:
        X: Input matrix.
        rowvar: True if each row represents an observation, False if each column represents an observation (default False).

    Returns:
        Eigenvalues, eigenvectors and explained variances.
    """
    u, s, vh = np.linalg.svd(X)
    n = X.shape[0]
    sdev = s / np.sqrt(max(1, n - 1))

    evals, evecs = s ** 2, vh.T

    explained_variances = []
    for i in range(len(evals)):
        explained_variances.append(evals[i] / np.sum(evals))

    return evals, evecs, explained_variances


def pep_pos_weight(encoded_pep_df: pd.DataFrame, pos_weights: List[float], aafeatmat: pd.DataFrame):
    """Weight amino acid features by position.

    Args:
        encoded_pep_df (pd.DataFrame): Encoded peptide sequences.
        pos_weights (List[float]): Average of the allele-specific and pan-allele entropies.
        aafeatmat (AminoAcidFeatureMap): AA x Feature matrix.

    Returns:
        Encoded feature matrix weighted by position.
    """
    num_feats_per_pos = aafeatmat.get_feature_count()
    peplen = int(encoded_pep_df.shape[1]/num_feats_per_pos)
    for i in range(peplen):
        pos_cols = 'p{0}_'.format(i+1) + aafeatmat.feature_map.columns.values
        encoded_pep_df[pos_cols] = np.multiply(encoded_pep_df[pos_cols], pos_weights[i])
    return encoded_pep_df


def PCA_encode(peptides: List[str], \
               allele: str, \
               peplen: int, \
               aa_featurefiles: List[os.PathLike]=None, \
               precomp_PCA_path: os.PathLike=None, \
               save_PCA_path: os.PathLike=None) -> pd.DataFrame:
    """Encodes peptides and performs PCA.

    Args:
        peptides (List[str]): List of peptides to encode.
        allele (str): HLA allele.
        peplen (int): Length of peptides.
        aa_featurefiles (List[os.PathLike], optional): List of paths to files containing amino acid features (default None).
        precomp_PCA_path (str, optional): Path to precomputed PCA object (default None).
        save_PCA_path (str, optional): Path to save PCA object (default None).

    Returns:
        Encoded peptides with PCA applied.
    """
    data = importlib_resources.files('hlathena').joinpath('data')
    molecularEntropies_MS_file = data.joinpath(f'molecularEntropies_{str(peplen)}_MS.txt')
    molecularEntropies_IEDB_file = data.joinpath(f'molecularEntropies_{str(peplen)}_IEDB.txt')
    
    pep_df = pd.DataFrame(peptides, columns=['seq'])
    
    peptide_dataset = PeptideDataset(pep_df, peplen=9, aa_featurefiles=aa_featurefiles)
    
    encoded_peptides = peptide_dataset.get_aa_encoded_peptide_map()
    aa_featuremap = peptide_dataset.aa_feature_map
    
    
    ###  Weight positions by entropy
    molecularEntropies_MS = pd.read_csv(molecularEntropies_MS_file, sep=' ', header=0)
    molecularEntropies_IEDB = pd.read_csv(molecularEntropies_IEDB_file, sep=' ', header=0)
    molecularEntropies_MS_IEDB = (molecularEntropies_MS + molecularEntropies_IEDB)/2
    
    # Average of the allele-specific and pan-allele entropies so we don't miss plausible anchors/subanchors
    pos_weights = (((1-molecularEntropies_MS_IEDB.loc[allele,:]) + (1-molecularEntropies_MS_IEDB.loc['Avg',:]))/2).tolist()
    
    encoded_peps_wE = pep_pos_weight(encoded_peptides, pos_weights, aa_featuremap)
    if precomp_PCA_path != None:
        npz_tmp = np.load(precomp_PCA_path)
        evecs = npz_tmp['evecs']
        explained_variances = npz_tmp['explained_variances']
    else:
        evals, evecs, explained_variances = PCA_numpy_SVD(encoded_peps_wE)
        
    if save_PCA_path != None:
        np.savez(save_PCA_path, evals=evals, evecs=evecs, explained_variances=explained_variances)
            
    peps_wE_pca_nocenter_df = pd.DataFrame(
        np.dot(encoded_peps_wE, evecs), 
        index=encoded_peps_wE.index, 
        columns=['PC{0}'.format(i) for i in range(len(evecs))])    
    
    return peps_wE_pca_nocenter_df
