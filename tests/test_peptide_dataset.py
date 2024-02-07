import unittest

import pandas as pd

import hlathena


class TestPeptideDataset(unittest.TestCase):
    def setUp(self):
        peptide_df = pd.DataFrame([['KSSFLSSPE', 0, 'A0101'],
                                   ['RTEAAFSYY', 1, 'A0101'],
                                   ['ASPQTLVLY', 1, 'A0101'],
                                   ['GVMLDDYIR', 0, 'A0101'],
                                   ['ADMGHLKY', 0, 'A0101'],
                                   ['TVLCAAGQA', 0, 'B4002'],
                                   ], columns=['pep', 'tgt', 'mhc'])
        self.peptide_dataset = hlathena.PeptideDataset(peptide_df, target_col_name='tgt', allele_col_name='mhc')

    def test_init(self):
        self.assertEqual(len(self.peptide_dataset), 6)
        self.assertListEqual(self.peptide_dataset.get_peptide_lengths(), [8, 9])

    def test_peptide_counts(self):
        peptide_counts = self.peptide_dataset.get_allele_peptide_counts()
        self.assertIsNotNone(peptide_counts)
        self.assertEqual(len(peptide_counts.loc['A0101']), 1)
        self.assertEqual(
            peptide_counts.loc['A0101'].iloc[0],
            5
        )
        self.assertEqual(len(peptide_counts.loc['B4002']), 1)
        self.assertEqual(
            peptide_counts.loc['B4002'].iloc[0],
            1
        )

if __name__ == '__main__':
    unittest.main()
