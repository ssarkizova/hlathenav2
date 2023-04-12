import math
from collections import Counter
from typing import List
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import umap
from hlathena import peptide_projection_lib
from hlathena.definitions import AMINO_ACIDS
import logomaker


def plot_length(peptides: List[str]) -> None:
    """
    Plots the distribution of peptide lengths.

    :param tsv_file: The path to the input TSV file.
    :type tsv_file: str
    :return: None
    """
    ncol_plot = 1
    fig, axs = plt.subplots(1, ncol_plot, sharex=False, sharey=False, figsize=(4.5*ncol_plot, 4));
    pep_df = pd.DataFrame(peptides, columns = ['seq'])
    pep_df['length'] = pep_df['seq'].str.len()
    ax = axs
    pep_df['length'].value_counts().sort_index().plot.bar(ax=ax);
    ax.set_title(f'Length distribution (n={pep_df.shape[0]})');
    

    
def plot_logo(peptides: List[str], length: int = None) -> None:
    """
    Plots the sequence logo for a given allele and peptide length.

    :param peptides: List of peptide sequences
    :type peptides: List[str]
    :param length: The length of peptides to plot.
    :type length: int, optional
    :return: None
    """
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



def plot_umap(tsv_file: str, allele: str, length: int) -> None:
    """
    Plots the UMAP embedding for a given allele and peptide length.

    :param tsv_file: The path to the input TSV file.
    :type tsv_file: str
    :param allele: The HLA allele to plot.
    :type allele: str
    :param length: The length of peptides to plot.
    :type length: int
    :return: None
    """
    out_dir = 'out/'
    hits_KFwEPCA =  peptide_projection_lib.encode_KF_wE_PCA(tsv_file, allele, length, # TODO: THIS SHOULD BE SEPARATE AND PROVIDED TO THIS FUNCTION
                                    pep_col='seq', use_precomp_PCA=False)
    
    # UMAP embedding
    umap_transform = umap.UMAP(n_neighbors=5, min_dist=0.5, random_state=42).fit(hits_KFwEPCA)
    # the hits, i.e. identical to above but here as an example how to embed new data
    umap_embedding_hits = umap_transform.transform(hits_KFwEPCA) 

    
    ### UMAP plot
    plt.figure(figsize = (6,6))
    plt.scatter(umap_embedding_hits[:, 0], umap_embedding_hits[:, 1], 
                s=10, facecolors='none', edgecolors='black', linewidths=0.1, alpha=0.75)
    plt.title(f'{allele}\n(black:hits)', fontsize=15)
    plt.xlabel('umap1', fontsize=15)
    plt.ylabel('umap2', fontsize=15)
    plt.axis('equal')
    plt.savefig(f'{out_dir}KFwE_PCA_UMAP_{allele}_{str(length)}_hits.pdf');
    
    
def get_logo_df(peptides, length):
    peps = [pp for pp in peptides if len(pp) == length]
    
    aa_counts = pd.Series([pd.DataFrame(peps)[0].str.slice(i,i+1).str.cat() for i in range(0,length)]).apply(Counter)
    aa_freq = pd.concat([pd.DataFrame(aa_counts[i],columns=AMINO_ACIDS,index=[i]) for i in range(0,length)]).fillna(0)
    aa_freq_norm = aa_freq.div(aa_freq.sum(axis=1),axis=0)
    R = np.log2(20) - stats.entropy(aa_freq_norm, base=2, axis=1)

    df = aa_freq_norm.mul(R, axis=0)
    return df