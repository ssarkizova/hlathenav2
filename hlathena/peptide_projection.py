
import logging
import os
import time
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import sklearn.preprocessing
import importlib_resources
from sklearn.cluster import DBSCAN
from collections import Counter

from hlathena.definitions import AMINO_ACIDS
from hlathena.peptide_dataset import PeptideDataset
from hlathena.amino_acid_feature_map import AminoAcidFeatureMap
from hlathena.pep_encoder import PepEncoder

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


def pep_pos_weight(encoded_pep_df: pd.DataFrame, 
                   pos_weights: List[float], 
                   aafeatmat: pd.DataFrame):
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


def PCA_encode(peptides: Union[List[str], PeptideDataset],
               allele: str,
               peplen: int,
               aa_featuremap,
               precomp_PCA_path: os.PathLike=None,
               save_PCA_path: os.PathLike=None) -> pd.DataFrame:
    """Encodes peptides and performs PCA.

    Args:
        peptides (List[str]): List of peptides to encode.
        allele (str): HLA allele.
        peplen (int): Length of peptides.
        #aa_featurefiles (List[os.PathLike], optional): List of paths to files containing amino acid features (default None).
        aa_featuremap # TO DO
        precomp_PCA_path (str, optional): Path to precomputed PCA object (default None).
        save_PCA_path (str, optional): Path to save PCA object (default None).

    Returns:
        Encoded peptides with PCA applied.
    """


    data = importlib_resources.files('hlathena').joinpath('data')
    molecularEntropies_MS_file = data.joinpath(f'molecularEntropies_{str(peplen)}_MS.txt')
    molecularEntropies_IEDB_file = data.joinpath(f'molecularEntropies_{str(peplen)}_IEDB.txt')
   
    encoded_peptides = PepEncoder.get_encoded_peps(peptides, aa_featuremap)
    
    ###  Weight positions by entropy
    molecularEntropies_MS = pd.read_csv(molecularEntropies_MS_file, sep=' ', header=0)
    molecularEntropies_IEDB = pd.read_csv(molecularEntropies_IEDB_file, sep=' ', header=0)
    molecularEntropies_MS_IEDB = (molecularEntropies_MS + molecularEntropies_IEDB)/2 # TO DO, we should pre-calc and save this so it's ready to use vs computing each time
    
    # Average of the allele-specific and pan-allele entropies so we don't miss plausible anchors/subanchors - TO DO, we should pre-calc and save this so it's ready to use vs computing each time
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


def get_umap_embedding(feature_matrix: pd.DataFrame, \
                        n_neighbors: int = 5, \
                        min_dist: float = 0.5, \
                        random_state: int = 42):
    """Create UMAP embedding dataframe for peptides.

    Args:
        feature_matrix (pd.DataFrame): A dataframe with the peptide PCA encoding.
        n_neighbors (int, optional):   This parameter controls the balance between local versus global structure in the data.
        min_dist (float, optional):    This parameter controls how tightly points are packed together.
        random_state (int, optional):  This is the random state seed value which can be fixed to ensure reproducibility.


    Returns:
        A pandas dataframe with the UMAP embedding of the peptide sequences.

    """
    # UMAP embedding
    umap_transform = umap.UMAP(n_neighbors=n_neighbors, 
                               in_dist=min_dist, 
                               random_state=random_state).fit(feature_matrix)
    # the hits, i.e. identical to above but here as an example how to embed new data - TO DO CHECK
    umap_embedding = umap_transform.transform(feature_matrix)     
    umap_embedding_df = pd.DataFrame(
        np.column_stack((feature_matrix.index.values, umap_embedding)), 
        columns=['pep','umap_1','umap_2'])
    
    return umap_embedding_df


def get_peptide_clustering(umap_embedding: pd.DataFrame,
                           eps: int = 3,
                           min_samples: int = 7):
    """Label peptide clusters.

    Args:
        umap_embedding (pd.DataFrame): A dataframe with the peptide PCA encoding.
        eps (int, optional):           The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int, optional):   The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        

    Returns:
        UMAP DataFrame with 'cluster' column. 

    """
    
    subclust = DBSCAN(eps=eps, min_samples=min_samples).fit(umap_embedding.loc[:,['umap_1','umap_2']])
    subclust_freqs = Counter(subclust.labels_)
    nclust = len(np.unique(subclust.labels_))
    ci_order_by_size = subclust_freqs.most_common()
    
    # Re-order cluster numbers by size of cluster (high to low)
    size_map_dict = dict(zip([ci[0] for ci in ci_order_by_size], range(nclust)))

    umap_embedding['cluster'] = pd.Series(
        pd.Series(subclust.labels_).map(size_map_dict), 
        index=umap_embedding.index)

    return umap_embedding
                           