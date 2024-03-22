"""Perform statistical tests on peptide datasets"""

import pandas as pd
import scipy.stats

from hlathena.peptide_dataset import PeptideDataset


def peptide_length_chi2(
    peps1: PeptideDataset, peps2: PeptideDataset
) -> scipy.stats.contingency.Chi2ContingencyResult:
    """Perform a chi-squared test on the lengths of peptides in two peptide datasets.

    Args:
        peps1: The first peptide dataset to be compared.
        peps2: The scond peptide dataset to be compared.

    Returns:
        A Chi2ContingencyResult containing the results of the test.
    """

    lengths = pd.concat(
        [peps1.pep_df['ha__pep'].str.len().value_counts(),
        peps2.pep_df['ha__pep'].str.len().value_counts()],
        axis=1).fillna(0).astype(int)

    return scipy.stats.chi2_contingency(lengths)
