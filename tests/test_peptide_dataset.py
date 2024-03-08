import copy
import os
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

    def test_warns_with_invalid_amino_acids(self):
        with self.assertWarnsRegex(Warning, "contain invalid characters"):
            # B is not an amino acid
            hlathena.PeptideDataset(["KSSFLSSPB"])

    def test_subsets(self):
        peptide_dataset_len_8 = copy.deepcopy(self.peptide_dataset)
        peptide_dataset_len_8.subset_data(peplens=8)
        self.assertEqual(len(peptide_dataset_len_8), 1)

        # Same as above three lines but using a list [8] instead of the int 8
        peptide_dataset_len_8 = copy.deepcopy(self.peptide_dataset)
        peptide_dataset_len_8.subset_data(peplens=[8])
        self.assertEqual(len(peptide_dataset_len_8), 1)

        peptide_dataset_allele_a0101 = copy.deepcopy(self.peptide_dataset)
        peptide_dataset_allele_a0101.subset_data(alleles='A0101')
        self.assertEqual(len(peptide_dataset_allele_a0101), 5)

        # Same as above three lines but using a list ['A0101'] instead of the string 'A0101'
        peptide_dataset_allele_a0101 = copy.deepcopy(self.peptide_dataset)
        peptide_dataset_allele_a0101.subset_data(alleles=['A0101'])
        self.assertEqual(len(peptide_dataset_allele_a0101), 5)

        peptide_dataset_len_9_allele_a0101 = copy.deepcopy(self.peptide_dataset)
        peptide_dataset_len_9_allele_a0101.subset_data(peplens=9, alleles='A0101')
        self.assertEqual(len(peptide_dataset_len_9_allele_a0101), 4)

    def test_allele_standardization(self):
        peptide_df = pd.DataFrame([['KSSFLSSPE', 'A*01:01'],
                                   ['RTEAAFSYY', 'A*0101'],
                                   ['ASPQTLVLY', 'A01:01'],
                                   ['GVMLDDYIR', 'A0101'],
                                   ['TVLCAAGQA', 'HLA-B*40:02'],
                                   ], columns=['pep', 'allele'])
        peptide_dataset = hlathena.PeptideDataset(peptide_df, allele_col_name='allele')
        self.assertListEqual(peptide_dataset.get_alleles(), ['A0101', 'B4002'])

    def test_tile_peptides(self):
        fasta_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'fake_fasta.fa')
        tiled_peptide_dataset = hlathena.tile_peptides(fasta_path, (11, 12))
        peptides = tiled_peptide_dataset.get_peptides()

        # The first 11 and 12 characters of the second gene in the fasta file.
        self.assertIn('DTEFPNFKYDT', peptides)
        self.assertIn('DTEFPNFKYDTE', peptides)
        # The last 11 and 12 characters of the second gene in the fasta file.
        self.assertIn('LVLYLLLIDLL', peptides)
        self.assertIn('LVLYLLLIDLLK', peptides)

        self.assertSetEqual({len(pep) for pep in peptides}, {11, 12})

        pep_df = tiled_peptide_dataset.pep_df

        # The two peptides at the start of the second gene only occur once in the dataset.
        self.assertEqual(len(pep_df[pep_df['pep'] == 'DTEFPNFKYDT']), 1)
        self.assertEqual(len(pep_df[pep_df['pep'] == 'DTEFPNFKYDTE']), 1)

        first_eleven = pep_df[pep_df['pep'] == 'DTEFPNFKYDT'].iloc[0]
        self.assertEqual(first_eleven['length'], 11)
        self.assertEqual(first_eleven['start_pos'], 0)
        self.assertEqual(first_eleven['record_id'], 'TEST00000000002|TST2')

        first_twelve = pep_df[pep_df['pep'] == 'DTEFPNFKYDTE'].iloc[0]
        self.assertEqual(first_twelve['length'], 12)
        self.assertEqual(first_twelve['start_pos'], 0)
        self.assertEqual(first_twelve['record_id'], 'TEST00000000002|TST2')


if __name__ == '__main__':
    unittest.main()
