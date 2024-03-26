"""Perform statistical tests on peptide datasets"""

import pandas as pd
import scipy.stats


def peptide_length_chi2(
    pep_df: pd.DataFrame,
    label_col: str,
    pep_col: str = 'ha__pep'
) -> scipy.stats.contingency.Chi2ContingencyResult:
    """Perform a chi-squared test comparing the lengths of peptides with different labels.

    Args:
        pep_df: A dataframe with a column of peptide sequences.
        label_col: The name of a column denoting the peptide category for length
            comparison.
        pep_col: The name of the peptide sequence column. Default value is 'ha__pep'.

    Returns:
        A Chi2ContingencyResult containing the results of the test.
    """

    # Create a dataframe whose rows are labels and whose columns are peptide length.
    # The entry for a particular (label, length) pair is the number of peptides with that length
    # and label.
    lengths = pep_df[pep_col].str.len().groupby(pep_df[label_col]).value_counts().unstack().fillna(0).astype(int)

    return scipy.stats.chi2_contingency(lengths)
