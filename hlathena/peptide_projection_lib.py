
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

def pep_encode_onehot(peps, pep_len):
    """One-hot encodes peptide sequences.

    :param peps: List of peptide sequences.
    :type peps: list of str
    :param pep_len: Length of peptide sequences.
    :type pep_len: int
    :raises ValueError: If peptide sequences are not of the same length.
    :return: One-hot encoded peptide sequences.
    :rtype: numpy.ndarray
    """
    encoder = sklearn.preprocessing.OneHotEncoder(categories=[AMINO_ACIDS] * pep_len)
    encoder.fit(peps)
    encoded = encoder.transform(peps).toarray()
    return encoded


def pep_encode_aafeatmat(peps, aafeatmat):
    """Encodes peptide sequences according to AA x Feats matrix.

    :param peps: List of peptide sequences.
    :type peps: list of str
    :param aafeatmat: AA x Feats matrix.
    :type aafeatmat: pandas.DataFrame
    :raises AssertionError: If peptide sequences are not of the same length.
    :return: Encoded peptide sequences.
    :rtype: pandas.DataFrame
    """
    # Input peptide sequences need to be of the same length
    pep_len = len(peps[0])
    assert(all(len(pep) == pep_len for pep in peps))

    # Split up each peptide string into individual amino acids
    if isinstance(peps[0], str):
        peps_split = [list(s) for s in peps]

    # One-hot (binary) encoding
    encoded = pep_encode_onehot(peps_split, pep_len)

    # Transform one-hot encoding according to AA x Feats matrix
    # Ensure the rows have the same order as the onehot encoding
    # This enables efficient transformation to other encodings
    # by multiplication (below).
    aafeatmat = aafeatmat.loc[AMINO_ACIDS, :]
    # Block diagonal aafeatmat
    aafeatmat_bd = np.kron(np.eye(pep_len, dtype=int), aafeatmat)
    # Feature encoding (@ matrix multiplication)
    num_feats_per_pos = aafeatmat.shape[1]  # assumes AA x Feats
    feat_names = list(np.concatenate(
        [('p{0}_'.format(i + 1) + aafeatmat.columns.values).tolist() for i in range(pep_len)]).flat)
    peps_aafeatmat = pd.DataFrame(encoded @ aafeatmat_bd, columns=feat_names, index=peps)
    return peps_aafeatmat


def PCA_numpy_SVD(X, rowvar=False):
    """Computes the PCA of a matrix using SVD.

    :param X: Input matrix.
    :type X: numpy.ndarray
    :param rowvar: True if each row represents an observation, False if each column represents an observation, defaults to False
    :type rowvar: bool, optional
    :return: Eigenvalues, eigenvectors and explained variances.
    :rtype: tuple of numpy.ndarray
    """
    u, s, vh = np.linalg.svd(X)
    n = X.shape[0]
    sdev = s / np.sqrt(max(1, n - 1))

    evals, evecs = s ** 2, vh.T

    explained_variances = []
    for i in range(len(evals)):
        explained_variances.append(evals[i] / np.sum(evals))

    return evals, evecs, explained_variances


def pep_pos_weight(dat, pos_weights, aafeatmat):
    """
    Weight amino acid features by position.

    :param dat: Encoded peptide sequences
    :type dat: pd.DataFrame
    :param pos_weights: Average of the allele-specific and pan-allele entropies
    :type pos_weights: float64
    :param aafeatmat: AA x Feature matrix
    :type aafeatmat: pd.DataFrame
    :return: Encoded feature matrix weighted by position
    :rtype: pd.DataFrame
    """
    num_feats_per_pos = aafeatmat.shape[1]
    peplen = int(dat.shape[1]/num_feats_per_pos)
    for i in range(peplen):
        pos_cols = 'p{0}_'.format(i+1) + aafeatmat.columns.values
        dat[pos_cols] = np.multiply(dat[pos_cols], pos_weights[i])
    return dat


def encode_KF_wE_PCA(
    tsv_file, allele, peplen, pep_col='seq',
    use_precomp_PCA=True):
    """
    PCA transform peptides using Kidera Factor encoding.

    :param tsv_file: The path to the input TSV file.
    :type tsv_file: str
    :param allele: The HLA allele to plot.
    :type allele: str
    :param length: The length of peptides to plot.
    :type length: int
    :param pep_col: The name of the input file's peptide column, defaults to 'seq'
    :type pep_col: str, optional
    :param use_precomp_PCA: Indicate whether to use a precomputed PCA model, defaults to True
    :type use_precomp_PCA: bool, optional
    :return: The peptide PCA feature matrix
    :rtype: pd.DataFrame
    """
    data = importlib_resources.files('hlathena').joinpath('data')
    KFs_file = data.joinpath('kideraFactors.txt')
    molecularEntropies_MS_file = data.joinpath(f'molecularEntropies_{str(peplen)}_MS.txt')
    molecularEntropies_IEDB_file = data.joinpath(f'molecularEntropies_{str(peplen)}_IEDB.txt')
    npz_file = data.joinpath(f'./projection_models/PCA_KFwE_{allele}_{str(peplen)}.npz')
    
    peplen = int(peplen)
    
    pep_df = pd.read_csv(tsv_file, sep='\t')
    
    pep_df = pep_df[ (pep_df['allele']==allele) & (pep_df['length']==peplen)].copy()
    
    aa_code = list('GPAVLIMCFYWHKRQNEDST')
    
    sequences = pep_df['seq'].to_list()
    
    
    # Load AA feature matrix and encodfe peptides
    KFs = pd.read_csv(KFs_file, sep=' ', header=0)
    peps_KF = pep_encode_aafeatmat(pep_df[pep_col].values, KFs)
    
    ###  Weight positions by entropy
    molecularEntropies_MS = pd.read_csv(molecularEntropies_MS_file, sep=' ', header=0)
    molecularEntropies_IEDB = pd.read_csv(molecularEntropies_IEDB_file, sep=' ', header=0)
    molecularEntropies_MS_IEDB = (molecularEntropies_MS + molecularEntropies_IEDB)/2
    # Average of the allele-specific and pan-allele entropies so we don't miss plausible anchors/subanchors
    pos_weights = ((1-molecularEntropies_MS_IEDB.loc[allele,:]) + (1-molecularEntropies_MS_IEDB.loc['Avg',:]))/2    
    peps_KFwE = pep_pos_weight(peps_KF, pos_weights, KFs)
    peps_KFwE.shape
    
    ### PCA-transform
    if not use_precomp_PCA:
        evals, evecs, explained_variances = PCA_numpy_SVD(peps_KFwE)        
        np.savez(npz_file, 
            evals=evals, evecs=evecs, explained_variances=explained_variances)
    else:
        npz_tmp = np.load(npz_file)
        evecs = npz_tmp['evecs']
        explained_variances = npz_tmp['explained_variances']
    
    logging.info('Explained variance per PC:\n{' '.join([str(np.round(ev.real,6)) for ev in explained_variances])}')
    
    peps_KFwE_pca_nocenter_df = pd.DataFrame(
        np.dot(peps_KFwE, evecs), 
        index=peps_KFwE.index, 
        columns=['PC{0}'.format(i) for i in range(len(evecs))])    
    
    return peps_KFwE_pca_nocenter_df
