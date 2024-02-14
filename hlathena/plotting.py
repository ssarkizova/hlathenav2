import math
from typing import List, Optional
from collections import Counter

import matplotlib.axes
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
# import umap
import logomaker
# from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from hlathena.definitions import AMINO_ACIDS
from hlathena.peptide_dataset import PeptideDataset

markers = list(plt.Line2D.filled_markers)



# SISI - we should have a version for this plotting function that takes in the peptide dataset class as well
def plot_length(pep_df: pd.DataFrame,
                pep_col: str = 'seq',
                label_col: Optional[str] = None) -> matplotlib.axes.Axes:
    """Plot the distribution of peptide lengths.

    Args:
        pep_df (pd.DataFrame):     A dataframe with a column of peptide sequences.
        pep_col (str, optional):   The name of the peptide sequence column. Default value is 'seq'.
        label_col (str, optional): The name of a column denoting the peptide category for length comparison. Default value is None, in which case no length category comparison will be performed.

    Returns:
        The matplotlib Axes object with the length plot.
        
    Raises:
        IndexError: No peptides provided for plotting
    """
    if not pep_df.shape[0]:
        raise IndexError("No peptides provided for plotting")
    
    pep_df['length'] = pep_df[pep_col].str.len()
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(4.5, 4));

    if not label_col is None:
        sns.countplot(data=pep_df, x='length', hue=label_col, ax=ax)
    else:
        pep_df['length'].value_counts().sort_index().plot.bar(ax=ax);
        ax.set_title(f'Length distribution (n={pep_df.shape[0]})')
    return ax
        

    
def plot_logo(pep_df: pd.DataFrame,
              length: Optional[int] = None,
              pep_col: str = 'seq',
              label_col: Optional[str] = None) -> List[matplotlib.axes.Axes]:
    """Plot the sequence logo for a given allele and peptide length.

    Args:
        pep_df (pd.DataFrame):     A dataframe with a column of peptide sequences.
        length (int, optional):    The length of peptides to plot. If not provided, the function 
                                   will plot all lengths found in `peptides`.
        pep_col (str, optional):   The name of the peptide sequence column. Default value is 'seq'.
        label_col (str, optional): The name of a column denoting the peptide category for length comparison. Default value is None, in which case no logo category comparison will be performed.

    Returns:
        None
        
    Raises:
        IndexError: If `peptides` is empty.

    """
    peptides = pep_df[pep_col]
    
    if not len(peptides):
        raise IndexError("No peptides for plotting")

    
    pep_lengths = list(set([len(pep) for pep in peptides]))
    if len(pep_lengths) == 1:
        length = pep_lengths[0]
    
    if length is not None:
        if label_col is not None:
            
            num_cols = 3
            num_rows = math.ceil(len(pep_df[label_col].unique()) / num_cols)
            height_per_row = 2
            width_per_col = 4
            fig = plt.figure(figsize=(width_per_col * num_cols, height_per_row * num_rows))

            for i, (label, d) in enumerate(pep_df.groupby(label_col)):
                num_row, num_col = divmod(i, num_cols)
                ax = plt.subplot2grid((num_rows, num_cols), (num_row, num_col))
                # peps = d[pep_col]
                peps = [pp for pp in d[pep_col] if len(pp) == length]
                if not len(peps): continue
                logo_df = get_logo_df(peps, length)
                logomaker.Logo(logo_df, ax=ax, show_spines=False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'{label}, Length {length} (n={len(peps)})')
            return fig.get_axes()
        else: 
            peps = [pp for pp in peptides if len(pp) == length]
            logo_df = get_logo_df(peps, length)
            logo = logomaker.Logo(logo_df)
            logo.ax.set_title(f'Length {length} (n={len(peps)})')
            return [logo.ax]
    
    
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
            logo_df = get_logo_df(peps, l)
            logomaker.Logo(logo_df,
                           ax=ax,                    
                           show_spines=False);

            ax.set_xticks([]);
            ax.set_yticks([]);
            ax.set_title('Length {0} (n={1})'.format(l, len(peps)));

        # style and save figure
        fig.tight_layout()
        return fig.get_axes()



def plot_umap(umap_embedding_df: pd.DataFrame, 
              clustered: bool = False,
              label_col: Optional[str] = None,
              title: Optional[str] = None, 
              save_path: Optional[str] = None) -> None:
    """Plot the UMAP for a given UMAP embedding dataframe. 

    Args:
        umap_embedding_df (pd.DataFrame): A dataframe with the UMAP features. Columns 'seq' (peptide sequences),'umap_1', and 'umap_2' are expected.
        clustered (bool, optional):       Indicates whether UMAP embedding has a 'cluster' column. Default is false.
        label_col (str, optional):        The name of a column denoting the peptide category for comparison. Default value is None, in which case no category comparison will be performed.
        title (str, optional):            Optional plot title, defaults to None.
        save_path (str, optional):        Path where that figure will be saved to. Optional, default is None.

    Returns:
        None

    """
    
    if clustered:
        plot_clustered_umap(umap_embedding_df, label_col=label_col, title=title, save_path=save_path)
    else:
        fig, ax = plt.subplots(1,1, figsize=(8,8))
        
        if label_col is not None:
            for i, (label, d) in enumerate(umap_embedding_df.groupby(label_col)):
                ax.scatter(d.loc[:, 'umap_1'], 
                        d.loc[:, 'umap_2'], 
                        s=40, facecolors='none', edgecolors='black', linewidths=0.5,
                        marker=markers[i], label = label)
                ax.legend()

        else:
            ax.scatter(umap_embedding_df.loc[:, 'umap_1'], 
                        umap_embedding_df.loc[:, 'umap_2'], 
                        s=40, facecolors='none', edgecolors='black', linewidths=0.5, alpha=0.75)
    
        plt.xlabel('umap_1', fontsize=15)
        plt.ylabel('umap_2', fontsize=15)
        plt.axis('equal')
    
        if title is not None:
            plt.title(title, fontsize=15)

        if save_path is not None:
            plt.savefig(save_path);
    
    
def get_logo_df(peptides: List[str], length: int):
    """Return a pandas dataframe for the input peptide sequences.

    Args:
        peptides (List[str]): A list of peptide sequences.
        length (int): The length of peptides to include in the returned dataframe.

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



def plot_clustered_umap(umap_embedding_df: pd.DataFrame,
                        # label_df: pd.DataFrame = None,
                        label_col: Optional[str] = None,
                        title: Optional[str] = None,
                        save_path: Optional[str] = None):
    """Plot the  clustered UMAP for a given UMAP embedding dataframe. 

    Args:
        umap_embedding_df (pd.DataFrame): A dataframe with the UMAP features. Columns 'seq', 'umap_1', 'umap_2', and 'cluster' are expected.
        label_col (str, optional):        The name of a column denoting the peptide category for comparison. Default value is None, in which case no category comparison will be performed.
        title (str, optional):            Optional plot title, defaults to None.
        save_path (str, optional):        Path where that figure will be saved to. Optional, default is None.


    Returns:
        None

    """
    nclust = len(umap_embedding_df['cluster'].unique())
    sns_colors = sns.color_palette('tab20', nclust)
    cmap = ListedColormap(sns_colors)
    
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(10,8), gridspec_kw={'width_ratios': [4, 1]})
    if not label_col is None:
        for i, (label, d) in enumerate(umap_embedding_df.groupby(label_col)):
            ax0.scatter(d.loc[:, 'umap_1'], 
                    d.loc[:, 'umap_2'], 
                    s=40, facecolors='none', linewidths=0.5, 
                    edgecolors=cmap(d.loc[:, 'cluster']),
                    marker=markers[i], label=label)
        
        sns.countplot(data=umap_embedding_df, y='cluster', hue=label_col, ax=ax1)
        
    else:
        ax0.scatter(umap_embedding_df.loc[:, 'umap_1'], 
            umap_embedding_df.loc[:, 'umap_2'], 
            s=40, facecolors='none', linewidths=0.5, 
            edgecolors=cmap(umap_embedding_df.loc[:, 'cluster'])) 
        handles, labels = ax0.get_legend_handles_labels()
        
        sns.countplot(data=umap_embedding_df, y='cluster', ax=ax1, palette=sns_colors)

    ax0.set_xlabel('umap_1', fontsize=15)
    ax0.set_ylabel('umap_2', fontsize=15)
    
    clust_labels = [c for c in np.sort(umap_embedding_df['cluster'].unique())]
    clust_handles = [mpatches.Patch(color=cmap(c), label=c) for c in clust_labels]
    handles, labels = ax0.get_legend_handles_labels()
    handles.extend(clust_handles)
    labels.extend(clust_labels)
        
    ax0.legend(handles=handles, labels=labels)
    ax0.axis('equal')
    
    if title != None:
        plt.title(title, fontsize=15)
    
    if save_path != None:
        plt.savefig(save_path)
    
    plt.show()
    

# def get_umap_embedding(feature_matrix: pd.DataFrame, 
#                        n_neighbors: int = 5, 
#                        min_dist: float = 0.5, 
#                        random_state: int = 42):
#      """Create UMAP embedding dataframe for peptides.

#     Args:
#         feature_matrix (pd.DataFrame): A dataframe with the peptide PCA encoding.
#         n_neighbors (int, optional):   This parameter controls the balance between local versus global structure in the data.
#         min_dist (float, optional):          This parameter controls how tightly points are packed together.
#         random_state (int, optional):        This is the random state seed value which can be fixed to ensure reproducibility.


#     Returns:
#         None

#     """
    
#     # UMAP embedding
#     umap_transform = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state).fit(feature_matrix)
    
#     # the hits, i.e. identical to above but here as an example how to embed new data
#     umap_embedding_hits = umap_transform.transform(feature_matrix) 
#     hits_peps = feature_matrix.index.values

#     umap_embedding_df = pd.DataFrame(np.column_stack(
#                                         (hits_peps, umap_embedding_hits)), 
#                                         columns=['seq','d1','d2'])
    
#     return umap_embedding_df


# def get_peptide_clustering(umap_embedding: pd.DataFrame,
#                            eps: int = 3,
#                            min_samples: int = 7):
#     """Label peptide clusters.

#     Args:
#         umap_embedding (pd.DataFrame): A dataframe with the peptide PCA encoding.
#         eps (int, optional):           The maximum distance between two samples for one to be considered as in the neighborhood of the other.
#         min_samples (int, optional):   The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        

#     Returns:
#         UMAP DataFrame with 'cluster' column. 

#     """
    
#     subclust = DBSCAN(eps=eps, min_samples=min_samples).fit(umap_embedding.loc[:,['d1','d2']])
#     subclust_freqs = Counter(subclust.labels_)
#     nclust = len(np.unique(subclust.labels_))
#     ci_order_by_size = subclust_freqs.most_common()
    
#     # Re-order cluster numbers by size of cluster (high to low)
#     size_map_dict = dict(zip([ci[0] for ci in ci_order_by_size], range(nclust)))

#     umap_embedding['cluster'] = pd.Series(
#                                              pd.Series(subclust.labels_).map(size_map_dict), 
#                                              index=umap_embedding.index)

#     return umap_embedding
                           
