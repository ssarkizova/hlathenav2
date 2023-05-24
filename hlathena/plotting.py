import math
from collections import Counter
from typing import List
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import umap
import logomaker
from sklearn.cluster import DBSCAN

from hlathena.definitions import AMINO_ACIDS


def plot_length(peptides: List[str]) -> None:
    """Plot the distribution of peptide lengths.

    Args:
        peptides (List[str]): A list of peptide sequences.

    Returns:
        None
        
    Raises:
        IndexError: No peptides provided for plotting
    """
    if not len(peptides):
        raise IndexError("No peptides provided for plotting")
        
    ncol_plot = 1
    fig, axs = plt.subplots(1, ncol_plot, sharex=False, sharey=False, figsize=(4.5*ncol_plot, 4));
    pep_df = pd.DataFrame(peptides, columns = ['seq'])
    pep_df['length'] = pep_df['seq'].str.len()
    ax = axs
    pep_df['length'].value_counts().sort_index().plot.bar(ax=ax);
    ax.set_title(f'Length distribution (n={pep_df.shape[0]})');
    

    
def plot_logo(peptides: List[str], length: int = None) -> None:
    """Plot the sequence logo for a given allele and peptide length.

    Args:
        peptides (List[str]):   A list of peptide sequences.
        length (int, optional): The length of peptides to plot. If not provided, the function 
                                will plot all lengths found in `peptides`

    Returns:
        None
        
    Raises:
        IndexError: If `peptides` is empty.

    """
    
    if not len(peptides):
        raise IndexError("No peptides for plotting")

    pep_lengths = list(set([len(pep) for pep in peptides]))
    
    if not length is None or len(pep_lengths) < 2:
        length = pep_lengths[0] if length is None else length
        logo_df = get_logo_df(peptides, length)
        logo = logomaker.Logo(logo_df)
        logo.ax.set_title(f'Length {length} (n={logo_df.shape[0]})');
    
    
    else:
        num_cols = 3
        num_rows = math.ceil(len(pep_lengths) / num_cols)
        height_per_row = 2
        width_per_col = 4
        fig = plt.figure(figsize=[width_per_col * num_cols, height_per_row * num_rows])

        for i in range(len(pep_lengths)):
            l = pep_lengths[i]
            peps = [pep for pep in peptides if len(pep)==l]

            num_row, num_col = divmod(i, num_cols)    
            ax = plt.subplot2grid((num_rows, num_cols), (num_row, num_col))

            logomaker.Logo(get_logo_df(peps, l),
                           ax=ax,                    
                           show_spines=False);

            ax.set_xticks([]);
            ax.set_yticks([]);
            ax.set_title('Length {0} n={1}'.format(l, len(peps)));

        # style and save figure
        fig.tight_layout()



def plot_umap(feature_matrix: pd.DataFrame, title: str=None, save_path: str=None) -> None:
    """Plot the UMAP embedding for a given allele and peptide length.

    Args:
        tsv_file (str): The path to the input TSV file.
        allele (str): The HLA allele to plot.
        length (int): The length of peptides to plot.

    Returns:
        None

    """

    # UMAP embedding
    umap_transform = umap.UMAP(n_neighbors=5, min_dist=0.5, random_state=42).fit(feature_matrix)
    # the hits, i.e. identical to above but here as an example how to embed new data
    umap_embedding_hits = umap_transform.transform(feature_matrix) 

    
    ### UMAP plot
    plt.figure(figsize = (6,6))
    plt.scatter(umap_embedding_hits[:, 0], umap_embedding_hits[:, 1], 
                s=10, facecolors='none', edgecolors='black', linewidths=0.1, alpha=0.75)
    plt.xlabel('umap1', fontsize=15)
    plt.ylabel('umap2', fontsize=15)
    plt.axis('equal')
    
    if title != None:
        plt.title(title, fontsize=15)
    
    if save_path != None:
        plt.savefig(save_path);
    
    
def get_logo_df(peptides, length):
    """Return a pandas dataframe for the input peptide sequences.

    Args:
        peptides: A list of peptide sequences.
        length: The length of peptides to include in the returned dataframe.

    Returns:
        A pandas dataframe containing the amino acid frequencies for the input peptides of the specified length.

    """
    peps = [pp for pp in peptides if len(pp) == length]
    
    aa_counts = pd.Series([pd.DataFrame(peps)[0].str.slice(i,i+1).str.cat() for i in range(0,length)]).apply(Counter)
    aa_freq = pd.concat([pd.DataFrame(aa_counts[i],columns=AMINO_ACIDS,index=[i]) for i in range(0,length)]).fillna(0)
    aa_freq_norm = aa_freq.div(aa_freq.sum(axis=1),axis=0)
    R = np.log2(20) - stats.entropy(aa_freq_norm, base=2, axis=1)

    df = aa_freq_norm.mul(R, axis=0)
    return df


def plot_clustered_umap(feature_matrix: pd.DataFrame, \
                        label_df: pd.DataFrame = None, \
                        label_col: str = 'label', \
                        eps: int = 3, \
                        min_samples: int = 7, \
                        title: str=None, \
                        save_path: str=None):
    # UMAP embedding
    umap_transform = umap.UMAP(n_neighbors=5, min_dist=0.5, random_state=42).fit(feature_matrix)
    
    # the hits, i.e. identical to above but here as an example how to embed new data
    umap_embedding_hits = umap_transform.transform(feature_matrix) 
    hits_peps = feature_matrix.index.values

    umap_embedding_df = pd.DataFrame(np.column_stack((hits_peps, umap_embedding_hits)), columns=['seq','d1','d2'])
    
    
    subclust = DBSCAN(eps=eps, min_samples=min_samples).fit(umap_embedding_df.loc[:,['d1','d2']])
    subclust_freqs = Counter(subclust.labels_)
    nclust = len(np.unique(subclust.labels_))
    ci_order_by_size = subclust_freqs.most_common()
    
    # Re-order cluster numbers by size of cluster (high to low)
    size_map_dict = dict(zip([ci[0] for ci in ci_order_by_size], range(nclust)))

    umap_embedding_df['cluster'] = pd.Series(
                                             pd.Series(subclust.labels_).map(size_map_dict), 
                                             index=umap_embedding_df.index)
    umap_embedding_df = umap_embedding_df.set_index('seq')
    
    cmap = plt.cm.get_cmap('tab20', nclust)
    markers = list(plt.Line2D.filled_markers)

    plt.figure(figsize = (6,6))
    
    if not label_df is None:
        umap_embedding_df = umap_embedding_df.join(label_df)
        for i, (label, d) in enumerate(umap_embedding_df.groupby(label_col)):
            plt.scatter(d.loc[:, 'd1'], 
                    d.loc[:, 'd2'], 
                    s=40, facecolors='none', linewidths=0.5, 
                    edgecolors=cmap(d.loc[:, 'cluster']),
                    marker=markers[i], label = label)
    else:
        plt.scatter(umap_embedding_df.loc[:, 'd1'], 
            umap_embedding_df.loc[:, 'd2'], 
            s=10, facecolors='none', linewidths=0.5, 
            edgecolors=cmap(umap_embedding_df.loc[:, 'cluster'])) 

    plt.xlabel('umap1', fontsize=15)
    plt.ylabel('umap2', fontsize=15)
    plt.legend(fontsize = 15)
    plt.axis('equal')
    
    if title != None:
        plt.title(title, fontsize=15)
    
    if save_path != None:
        plt.savefig(save_path)
    
    plt.show()