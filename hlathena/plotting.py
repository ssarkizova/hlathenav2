"""Peptide dataset plotting module"""

import math
from typing import List
from collections import Counter
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import logomaker
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from hlathena.definitions import AMINO_ACIDS

markers = list(plt.Line2D.filled_markers)


# TODO: SISI - we should have a version for this plotting function that takes
#   in the peptide dataset class as well
def plot_length(
        pep_df: pd.DataFrame,
        pep_col: str = 'seq',
        label_col: str = None,
) -> None:
    """Plot the distribution of peptide lengths.

    Args:
        pep_df: A dataframe with a column of peptide sequences.
        pep_col: The name of the peptide sequence column. Default value is 'seq'.
        label_col: The name of a column denoting the peptide category for length
            comparison. Default value is None, in which case no length category
            comparison will be performed.

    Returns:
        The matplotlib Axes object with the length plot.
        
    Raises:
        IndexError: No peptides provided for plotting
    """
    if not pep_df.shape[0]:
        raise IndexError("No peptides provided for plotting")

    pep_df['length'] = pep_df[pep_col].str.len()
    _, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(4.5, 4))

    if label_col is not None:
        sns.countplot(data=pep_df, x='length', hue=label_col, ax=ax)
    else:
        pep_df['length'].value_counts().sort_index().plot.bar(ax=ax)
        ax.set_title(f'Length distribution (n={pep_df.shape[0]})')
    return ax


def plot_logo(
        pep_df: pd.DataFrame,
        length: int = None,
        pep_col: str = 'seq',
        label_col: str = None,
) -> None:
    """Plot the sequence logo for a given allele and peptide length.

    Args:
        pep_df: A dataframe with a column of peptide sequences.
        length: The length of peptides to plot. If not provided, the function
            will plot all lengths found in `peptides`.
        pep_col: The name of the peptide sequence column. Default value is 'seq'.
        label_col: The name of a column denoting the peptide category for length
            comparison. Default value is None, in which case no logo category
            comparison will be performed.
        
    Raises:
        IndexError: Raised if no peptides are provided.
    """
    peptides = pep_df[pep_col]

    if len(peptides) == 0:
        raise IndexError("No peptides for plotting")

    pep_lengths = list({len(pep) for pep in peptides})

    if length is not None or len(pep_lengths) < 2:
        if label_col is not None:
            num_cols = 3
            num_rows = math.ceil(len(pep_df[label_col].unique()) / num_cols)
            height_per_row = 2
            width_per_col = 4
            fig = plt.figure(figsize=[width_per_col * num_cols, height_per_row * num_rows])

            for i, (label, d) in enumerate(pep_df.groupby(label_col)):
                num_row, num_col = divmod(i, num_cols)
                ax = plt.subplot2grid((num_rows, num_cols), (num_row, num_col))
                # peps = d[pep_col]
                peps = [pp for pp in d[pep_col] if len(pp) == length]
                if len(peps) == 0:
                    continue
                logo_df = get_logo_df(peps, length)
                logomaker.Logo(logo_df, ax=ax, show_spines=False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'{label}, Length {length} (n={len(peps)})')
        else:
            peps = [pp for pp in peptides if len(pp) == length]
            logo_df = get_logo_df(peps, length)
            logo = logomaker.Logo(logo_df)
            logo.ax.set_title(f'Length {length} (n={len(peps)})')

    else:
        num_cols = 3
        num_rows = math.ceil(len(pep_lengths) / num_cols)
        height_per_row = 2
        width_per_col = 4
        fig = plt.figure(figsize=[width_per_col * num_cols, height_per_row * num_rows])

        for i, l in enumerate(pep_lengths):  # in range(len(pep_lengths)):
            peps = [pep for pep in peptides if len(pep) == l]

            num_row, num_col = divmod(i, num_cols)
            ax = plt.subplot2grid((num_rows, num_cols), (num_row, num_col))
            logo_df = get_logo_df(peps, l)
            logomaker.Logo(
                logo_df,
                ax=ax,
                show_spines=False
            )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Length {l} (n={len(peps)})')

        # style and save figure
        fig.tight_layout()


def plot_umap(
        umap_embedding_df: pd.DataFrame,
        clustered: bool = False,
        label_col: str = None,
        title: str = None,
        save_path: str = None,
) -> None:
    """Plot the UMAP for a given UMAP embedding dataframe. 

    Args:
        umap_embedding_df: A dataframe with the UMAP features. Columns 'seq'
            (peptide sequences),'umap_1', and 'umap_2' are expected.
        clustered: Indicates whether UMAP embedding has a 'cluster' column.
        label_col: The name of a column denoting the peptide category for
            comparison. Default value is None, in which case no category
            comparison will be performed.
        title: Optional plot title, defaults to None.
        save_path: Path where that figure will be saved to. Default is None.
    """

    if clustered:
        plot_clustered_umap(
            umap_embedding_df,
            label_col=label_col,
            title=title,
            save_path=save_path
        )
    else:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

        if label_col is not None:
            for i, (label, df) in enumerate(umap_embedding_df.groupby(label_col)):
                ax.scatter(
                    df.loc[:, 'umap_1'],
                    df.loc[:, 'umap_2'],
                    s=40,
                    facecolors='none',
                    edgecolors='black',
                    linewidths=0.5,
                    marker=markers[i],
                    label=label
                )
                ax.legend()

        else:
            ax.scatter(
                umap_embedding_df.loc[:, 'umap_1'],
                umap_embedding_df.loc[:, 'umap_2'],
                s=40,
                facecolors='none',
                edgecolors='black',
                linewidths=0.5,
                alpha=0.75
            )

        plt.xlabel('umap_1', fontsize=15)
        plt.ylabel('umap_2', fontsize=15)
        plt.axis('equal')

        if title is not None:
            plt.title(title, fontsize=15)

        if save_path is not None:
            plt.savefig(save_path)


def get_logo_df(
        peptides: List[str],
        length: int
) -> pd.DataFrame:
    """Creates amino acid frequency dataframe.

    Args:
        peptides: A list of peptide sequences.
        length: The length of peptides to include in the returned dataframe.

    Returns:
        Dataframe containing the amino acid frequencies for the input peptides
        of the specified length.
    """
    peps = [pp for pp in peptides if len(pp) == length]

    aa_counts = pd.Series(
        [pd.DataFrame(peps)[0].str.slice(i, i + 1).str.cat() for i in range(0, length)]
    ).apply(Counter)
    aa_freq = pd.concat(
        [pd.DataFrame(
            aa_counts[i],
            columns=AMINO_ACIDS,
            index=[i]
        ) for i in range(0, length)]
    ).fillna(0)
    aa_freq_norm = aa_freq.div(aa_freq.sum(axis=1), axis=0)
    R = np.log2(20) - stats.entropy(aa_freq_norm, base=2, axis=1)

    df = aa_freq_norm.mul(R, axis=0)
    return df


def plot_clustered_umap(
        umap_embedding_df: pd.DataFrame,
        label_col: str = None,
        title: str = None,
        save_path: str = None,
) -> None:
    """Plot the  clustered UMAP for a given UMAP embedding dataframe. 

    Args:
        umap_embedding_df: A dataframe with the UMAP features. Columns 'seq',
            'umap_1', 'umap_2', and 'cluster' are expected.
        label_col: The name of a column denoting the peptide category for
            comparison. Default value is None, in which case no category
            comparison will be performed.
        title: Optional plot title, defaults to None.
        save_path: Path where that figure will be saved to. Optional, default
            is None.
    """
    nclust = len(umap_embedding_df['cluster'].unique())
    sns_colors = sns.color_palette('tab20', nclust)
    cmap = ListedColormap(sns_colors)

    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 8), gridspec_kw={'width_ratios': [4, 1]})
    if label_col is not None:
        for i, (label, df) in enumerate(umap_embedding_df.groupby(label_col)):
            ax0.scatter(
                df.loc[:, 'umap_1'],
                df.loc[:, 'umap_2'],
                s=40,
                facecolors='none',
                linewidths=0.5,
                edgecolors=cmap(df.loc[:, 'cluster']),
                marker=markers[i],
                label=label
            )

        sns.countplot(data=umap_embedding_df, y='cluster', hue=label_col, ax=ax1)

    else:
        ax0.scatter(
            umap_embedding_df.loc[:, 'umap_1'],
            umap_embedding_df.loc[:, 'umap_2'],
            s=40,
            facecolors='none',
            linewidths=0.5,
            edgecolors=cmap(umap_embedding_df.loc[:, 'cluster'])
        )
        handles, labels = ax0.get_legend_handles_labels()

        sns.countplot(data=umap_embedding_df, y='cluster', ax=ax1, palette=sns_colors)

    ax0.set_xlabel('umap_1', fontsize=15)
    ax0.set_ylabel('umap_2', fontsize=15)

    clust_labels = list(np.sort(umap_embedding_df['cluster'].unique()))
    clust_handles = [mpatches.Patch(color=cmap(c), label=c) for c in clust_labels]
    handles, labels = ax0.get_legend_handles_labels()
    handles.extend(clust_handles)
    labels.extend(clust_labels)

    ax0.legend(handles=handles, labels=labels)
    ax0.axis('equal')

    if title is not None:
        plt.title(title, fontsize=15)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
