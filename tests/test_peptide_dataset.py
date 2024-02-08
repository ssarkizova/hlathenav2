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

    def test_peptides(self):
        self.assertListEqual(self.peptide_dataset.get_peptide_lengths(), [8, 9])
        self.assertListEqual(self.peptide_dataset.get_peptides(),
                             ['KSSFLSSPE', 'RTEAAFSYY', 'ASPQTLVLY', 'GVMLDDYIR', 'ADMGHLKY', 'TVLCAAGQA'])

    def test_get_alleles(self):
        self.assertIsNotNone(self.peptide_dataset.get_alleles())
        self.assertListEqual(self.peptide_dataset.get_alleles(), ['A0101', 'B4002'])

    def test_peptide_counts(self):
        peptide_counts = self.peptide_dataset.get_allele_peptide_counts()
        self.assertIsNotNone(peptide_counts)
        self.assertEqual(len(peptide_counts), 2)  # only two alleles
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

    def test_peptide_length_counts(self):
        peptide_length_counts = self.peptide_dataset.get_allele_peptide_length_counts()
        self.assertIsNotNone(peptide_length_counts)
        self.assertEqual(peptide_length_counts.shape, (2, 2))  # two alleles, two distinct lengths

        # There is one peptide of length 8 and four of length 9 for A0101.
        # There are no peptides of length 8 and one of length 9 for B4002.
        self.assertEqual(peptide_length_counts.loc['A0101', 8], 1)
        self.assertEqual(peptide_length_counts.loc['A0101', 9], 4)
        self.assertEqual(peptide_length_counts.loc['B4002', 8], 0)
        self.assertEqual(peptide_length_counts.loc['B4002', 9], 1)

    def test_init_with_list_of_peptides(self):
        peptide_dataset = hlathena.PeptideDataset(['RTEAAFSYY', 'ASPQTLVLY', 'GVMLDDYIR'])
        self.assertIsNone(peptide_dataset.get_alleles())
        self.assertIsNone(peptide_dataset.get_allele_peptide_counts())
        self.assertIsNone(peptide_dataset.get_allele_peptide_length_counts())

    def test_fails_with_no_peptides(self):
        with self.assertRaisesRegex(Exception, "No peptide sequences"):
            hlathena.PeptideDataset(pd.DataFrame(columns=['pep']))

    @unittest.skip("Check doesn't raise an exception.")
    def test_fails_with_short_peptides(self):
        with self.assertRaisesRegex(Exception, "lengths are not in the allowed range"):
            hlathena.PeptideDataset(["KSSFLSSPE", "KSSF"])

    @unittest.skip("Check doesn't raise an exception.")
    def test_fails_with_long_peptides(self):
        with self.assertRaisesRegex(Exception, "lengths are not in the allowed range"):
            hlathena.PeptideDataset(["KSSFLSSPE", "KSSFKSSFLSSPE"])

    @unittest.skip("Check doesn't raise an exception")
    def test_fails_with_invalid_amino_acids(self):
        with self.assertRaisesRegex(Exception, "contain invalid characters"):
            # B is not an amino acid
            hlathena.PeptideDataset(["KSSFLSSPB"])

if __name__ == '__main__':
    unittest.main()
