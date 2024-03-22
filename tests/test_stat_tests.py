import unittest
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection

import pandas as pd

from hlathena import PeptideDataset
from hlathena.stat_tests import peptide_length_chi2

class TestStatTests(unittest.TestCase):
    def test_peptide_length_chi2_different(self):
        peptide_dataset_1 = PeptideDataset(995 * [8 * 'A'] + 5 * [9 * 'A'])
        peptide_dataset_2 = PeptideDataset(5 * [8 * 'A'] + 995 * [9 * 'A'])

        chi2_values = peptide_length_chi2(peptide_dataset_1, peptide_dataset_2)
        # The length distributions in the two dataset are very different; this should be reflected
        # in the test statistic.
        self.assertLess(chi2_values.pvalue, 0.01)

    def test_peptide_length_chi2_similar(self):
        peptide_dataset_1 = PeptideDataset(995 * [8 * 'A'] + 5 * [9 * 'A'])
        peptide_dataset_2 = PeptideDataset(995 * [8 * 'A'] + 5 * [9 * 'A'])

        chi2_values = peptide_length_chi2(peptide_dataset_1, peptide_dataset_2)
        # The length distributions in the two dataset are exactly the same; this should be reflected
        # in the test statistic.
        self.assertAlmostEqual(chi2_values.pvalue, 1)
        self.assertAlmostEqual(chi2_values.statistic, 0)


if __name__ == '__main__':
    unittest.main()

