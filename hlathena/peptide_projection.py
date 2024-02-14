
import logging
import os
from typing import List, Tuple, Union, Optional
import importlib_resources

import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.cluster import DBSCAN
from collections import Counter

from hlathena.definitions import AMINO_ACIDS, aa_feature_file_Kidera
from hlathena.peptide_dataset import PeptideDataset
from hlathena.amino_acid_feature_map import AminoAcidFeatureMap
from hlathena.pep_hla_encoder import PepHLAEncoder


def PCA_numpy_SVD(X, rowvar=False):
    """
    Computes the PCA of a matrix using SVD.

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
                   aafeatmat: AminoAcidFeatureMap):
    """
    Weight amino acid features by position.

    Args:
        encoded_pep_df (pd.DataFrame): Encoded peptide sequences.
        pos_weights (List[float]): Average of the allele-specific and pan-allele entropies.
        aafeatmat (AminoAcidFeatureMap): Amino acid feature matrix.

    Returns:
        Encoded feature matrix weighted by position.
    """
    num_feats_per_pos = aafeatmat.get_aa_feature_count()
    peplen = int(encoded_pep_df.shape[1]/num_feats_per_pos)
    for i in range(peplen):
        pos_cols = 'p{0}_'.format(i+1) + aafeatmat.get_aa_feature_map().columns.values
        encoded_pep_df[pos_cols] = np.multiply(encoded_pep_df[pos_cols], pos_weights[i])
    return encoded_pep_df


def PCA_encode(peptides: Union[List[str], PeptideDataset],               
               peplen: int,  # TO DO: enable projection for mixed-length peptide input
               allele: Optional[str] = None,
               aa_feature_map: AminoAcidFeatureMap = None,
               precomp_PCA_path: os.PathLike = None,
               save_PCA_path: os.PathLike = None) -> pd.DataFrame:
    """
    Encodes peptides and performs PCA.

    Args:
        peptides (Union[List[str], PeptideDataset]): List of peptides to encode.
        peplen (int, optional): Length used to subset dataset before encoding (default None).
        allele (str, optional): HLA allele used to subset dataset before encoding (default None).
        aa_feature_map (AminoAcidFeatureMap, optional): Feature map with desired amino acid encoding features
                                            (default Kidera Factors feature map)
        precomp_PCA_path (os.PathLike, optional): Path to precomputed PCA object (default None).
        save_PCA_path (os.PathLike, optional): Path to save PCA object (default None).

    Returns:
        Encoded peptides with PCA applied.
    """

    # Ensure valid sequences, identical lengths etc. by using a PeptideDataset object
    if not isinstance(peptides,PeptideDataset):
        peptides = PeptideDataset(peptides, allele)
    peptides.subset_data(peplens=[peplen], alleles=[allele])
    peptides = peptides.get_peptides()

    # Use Kidera Factors features if none specified
    if aa_feature_map is None:
        aa_feature_map = AminoAcidFeatureMap(aa_feature_files=[aa_feature_file_Kidera])

    encoded_peptides = PepHLAEncoder.get_encoded_peps(peptides, aa_feature_map)

    #  Weight positions by entropy
    data = importlib_resources.files('hlathena').joinpath('data').joinpath('motif_entropies')
    motif_entropies_file = data.joinpath(f'motifEntropies_{str(peplen)}_MS_IEDB.txt')
    motif_entropies = pd.read_csv(motif_entropies_file, sep=' ', header=0)
    
    if not allele is None:
        # Average of the allele-specific and pan-allele entropies so we don't miss plausible anchors/subanchors
        pos_weights = (((1-motif_entropies.loc[allele,:]) + (1-motif_entropies.loc['Avg',:]))/2).tolist()
    else:
        # Average of pan-allele entropies
        pos_weights = (1-motif_entropies.loc['Avg',:]).tolist()
    
    encoded_peps_wE = pep_pos_weight(encoded_peptides, pos_weights, aa_feature_map)
    if precomp_PCA_path is not None:
        npz_tmp = np.load(precomp_PCA_path)
        evecs = npz_tmp['evecs']
        explained_variances = npz_tmp['explained_variances']
    else:
        evals, evecs, explained_variances = PCA_numpy_SVD(encoded_peps_wE)
        
    if save_PCA_path is not None:
        np.savez(save_PCA_path, evals=evals, evecs=evecs, explained_variances=explained_variances)
            
    peps_wE_pca_nocenter_df = pd.DataFrame(
        np.dot(encoded_peps_wE, evecs), 
        index=encoded_peps_wE.index, 
        columns=['PC{0}'.format(i) for i in range(len(evecs))])    
    
    peps_wE_pca_nocenter_df.index = peptides

    return peps_wE_pca_nocenter_df


def get_umap_embedding(feature_matrix: pd.DataFrame,
                       n_neighbors: int = 5,
                       min_dist: float = 0.5,
                       random_state: int = 42):
    """
    Create UMAP embedding dataframe for peptides.

    Args:
        feature_matrix (pd.DataFrame): A dataframe with the peptide PCA encoding.
        n_neighbors (int, optional):   This parameter controls the balance between local versus global structure in the data (default 5).
        min_dist (float, optional):    This parameter controls how tightly points are packed together (default 0.5).
        random_state (int, optional):  This is the random state seed value which can be fixed to ensure reproducibility (default 42).


    Returns:
        A pandas dataframe with the UMAP embedding of the peptide sequences.

    """
    # UMAP embedding
    umap_transform = umap.UMAP(n_neighbors=n_neighbors, 
                               min_dist=min_dist, 
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
    """
    Label peptide clusters.

    Args:
        umap_embedding (pd.DataFrame): A dataframe with the peptide PCA encoding.
        eps (int, optional):           The maximum distance between two samples for one to be considered as in the neighborhood of the other (default 3).
        min_samples (int, optional):   The number of samples (or total weight) in a neighborhood for a point to be considered as a core point (default 7).
        

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
