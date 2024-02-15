"""Module for peptide projection and clustering"""
import os
from typing import List, Tuple, Union, Optional
from collections import Counter
import importlib_resources

import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.cluster import DBSCAN

from hlathena.definitions import aa_feature_file_Kidera
from hlathena.peptide_dataset import PeptideDataset
from hlathena.amino_acid_feature_map import AminoAcidFeatureMap
from hlathena.pep_hla_encoder import PepHLAEncoder


def PCA_numpy_SVD(
    posn_weights_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:  # pylint: disable=invalid-name
    """Computes the PCA of a matrix using SVD.

    Args:
        posn_weights_df: Input matrix.

    Returns:
        Eigenvalues, eigenvectors and explained variances as a tuple.
    """
    _, s, vh = np.linalg.svd(posn_weights_df)  # pylint: disable=invalid-name
    evals, evecs = s ** 2, vh.T
    explained_variances = [evals[i] / np.sum(evals) for i in range(len(evals))]
    return evals, evecs, explained_variances


def pep_pos_weight(
    encoded_pep_df: pd.DataFrame,
    pos_weights: List[float],
    aa_feature_map: AminoAcidFeatureMap,
) -> pd.DataFrame:
    """Weight amino acid features by position.

    Args:
        encoded_pep_df: Encoded peptide sequences.
        pos_weights: Average of the allele-specific and pan-allele entropies.
        aa_feature_map: Amino acid feature matrix.

    Returns:
        Encoded feature matrix weighted by position.
    """
    num_feats_per_pos = aa_feature_map.get_aa_feature_count()
    peplen = int(encoded_pep_df.shape[1] / num_feats_per_pos)
    for i in range(peplen):
        pos_cols = f'p{i+1}_' + aa_feature_map.get_aa_feature_map().columns.values
        encoded_pep_df[pos_cols] = np.multiply(encoded_pep_df[pos_cols], pos_weights[i])
    return encoded_pep_df


def PCA_encode(  # pylint: disable=invalid-name
    peptides: Union[List[str], PeptideDataset],
    peplen: int,  # TO DO: enable projection for mixed-length peptide input
    allele: Optional[str] = None,
    aa_feature_map: Optional[AminoAcidFeatureMap] = None,
    precomp_PCA_path: Optional[os.PathLike] = None,
    save_PCA_path: Optional[os.PathLike] = None,
) -> pd.DataFrame:  # pylint: disable=too-many-locals disable=too-many-locals
    """Encodes peptides and performs PCA. If no aa_feature_map is provided,
    defaults to Kidera Factors.

    Args:
        peptides: List of peptides to encode.
        peplen: Length used to subset dataset before encoding.
        allele: HLA allele used to subset dataset before encoding.
        aa_feature_map: Amino acid feature matrix.
        precomp_PCA_path: Path to precomputed PCA object.
        save_PCA_path: Path to save PCA object.

    Returns:
        Dataframe indexed by peptide with one column per PC.
    """

    # Ensure valid sequences, identical lengths etc. by using a PeptideDataset object
    if not isinstance(peptides, PeptideDataset):
        peptides = PeptideDataset(peptides, allele)
    peptides.subset_data(peplens=[peplen], alleles=[allele])
    peptides = peptides.get_peptides()

    if aa_feature_map is None:
        aa_feature_map = AminoAcidFeatureMap(aa_feature_files=[aa_feature_file_Kidera])

    encoded_peptides = PepHLAEncoder.get_encoded_peps(peptides, aa_feature_map)

    #  Weight positions by entropy
    data = importlib_resources.files('hlathena').joinpath('data').joinpath('motif_entropies')
    motif_entropies_file = data.joinpath(f'motifEntropies_{str(peplen)}_MS_IEDB.txt')
    motif_entropies = pd.read_csv(motif_entropies_file, sep=' ', header=0)

    if allele is not None:
        # Average allele-specific and pan-allele entropies
        # so we keep all plausible anchors and subanchors
        pos_weights = ((
                        (1 - motif_entropies.loc[allele, :]) +
                        (1 - motif_entropies.loc['Avg', :])
                       ) / 2).tolist()
    else:
        # Average of pan-allele entropies
        pos_weights = (1 - motif_entropies.loc['Avg', :]).tolist()

    encoded_peps_wE = pep_pos_weight(encoded_peptides, pos_weights, aa_feature_map)
    if precomp_PCA_path is not None:
        npz_tmp = np.load(precomp_PCA_path)
        evecs = npz_tmp['evecs']
    else:
        evals, evecs, explained_variances = PCA_numpy_SVD(encoded_peps_wE)

        if save_PCA_path is not None:
            np.savez(
                save_PCA_path,
                evals=evals,
                evecs=evecs,
                explained_variances=explained_variances
            )

    peps_wE_pca_nocenter_df = pd.DataFrame(
                                            np.dot(encoded_peps_wE, evecs),
                                            index=encoded_peps_wE.index,
                                            columns=[f'PC{i}' for i in range(len(evecs))]
                                          )

    peps_wE_pca_nocenter_df.index = peptides

    return peps_wE_pca_nocenter_df


def get_umap_embedding(
    feature_matrix: pd.DataFrame,
    n_neighbors: int = 5,
    min_dist: float = 0.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create UMAP embedding dataframe for peptides.

    Args:
        feature_matrix: A dataframe with the peptide PCA encoding.
        n_neighbors: Controls balance between local vs. global structure in the data.
        min_dist: Controls how tightly points are packed together.
        random_state: Random state seed value which can be set to ensure reproducibility.

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
        columns=['pep', 'umap_1', 'umap_2'])

    return umap_embedding_df


def get_peptide_clustering(
    umap_embedding: pd.DataFrame,
    eps: int = 3,
    min_samples: int = 7,
) -> pd.DataFrame:
    """Cluster peptides using DBSCAN algorithm on the UMAP embedding.

    Args:
        umap_embedding: A dataframe with the peptide PCA encoding.
        eps: Maximum distance between two samples for one to be considered as in the neighborhood
          of the other.
        min_samples: The number of samples (or total weight) in a neighborhood for a point
          to be considered as a core point.

    Returns:
        UMAP embedding dataframe with appended 'cluster' column.
    """

    subclust = DBSCAN(eps=eps, min_samples=min_samples).fit(
        umap_embedding.loc[:, ['umap_1', 'umap_2']])
    subclust_freqs = Counter(subclust.labels_)
    nclust = len(np.unique(subclust.labels_))
    ci_order_by_size = subclust_freqs.most_common()

    # Re-order cluster numbers by size of cluster (high to low)
    size_map_dict = dict(zip([ci[0] for ci in ci_order_by_size], range(nclust)))

    umap_embedding['cluster'] = pd.Series(
        pd.Series(subclust.labels_).map(size_map_dict),
        index=umap_embedding.index)

    return umap_embedding
