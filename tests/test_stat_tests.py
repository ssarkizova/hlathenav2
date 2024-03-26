import unittest

import pandas as pd

from hlathena.stat_tests import peptide_length_chi2


class TestStatTests(unittest.TestCase):
    def test_peptide_length_chi2_different(self):
        # Label 1 has 995 peptides of length 8 and 5 of length 9.
        # Label 2 has 5 peptides of length 8 and 995 of length 9.
        pep_df = pd.DataFrame(
            995 * [[8 * 'A' , 1]] + 5 * [[9 * 'A', 1]] + 5 * [[8 * 'A', 2]] + 995 * [[9 * 'A', 2]],
            columns=['ha__pep', 'label']
        )

        chi2_values = peptide_length_chi2(pep_df, 'label')
        # The length distributions in the two data frames are very different; this should be reflected
        # in the p-value.
        self.assertLess(chi2_values.pvalue, 0.01)

    def test_peptide_length_chi2_similar(self):
        # Both labels have 995 peptides of length 8 and 5 of length 9.
        pep_df = pd.DataFrame(
            995 * [[8 * 'A' , 1]] + 5 * [[9 * 'A', 1]] + 995 * [[8 * 'A', 2]] + 5 * [[9 * 'A', 2]],
            columns=['ha__pep', 'label']
        )

        chi2_values = peptide_length_chi2(pep_df, 'label')
        # The length distributions in the two dataset are exactly the same; this should be reflected
        # in the test statistic.
        self.assertAlmostEqual(chi2_values.pvalue, 1)
        self.assertAlmostEqual(chi2_values.statistic, 0)

    def test_peptide_length_chi2_three_labels(self):
        # Labels 1 and 2 have 995 peptides of length 8 and 5 of length 9. Label 3 has
        # 5 peptides of length 8 and 995 of length 9.
        pep_df = pd.DataFrame(
            995 * [[8 * 'A' , 1]] + 5 * [[9 * 'A', 1]] + 995 * [[8 * 'A', 2]] + 5 * [[9 * 'A', 2]] +
            5 * [[8 * 'A', 3]] + 995 * [[9 * 'A', 3]],
            columns=['ha__pep', 'label']
        )
        chi2_values = peptide_length_chi2(pep_df, 'label')

        # Since the third label has a very different distribution, the p-value should be low.
        self.assertLess(chi2_values.pvalue, 0.01)


if __name__ == '__main__':
    unittest.main()

