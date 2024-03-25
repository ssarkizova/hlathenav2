import os
import unittest

import hlathena


class TestUtil(unittest.TestCase):
    def test_tile_peptides(self):
        fasta_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'data/fake_fasta.fa')
        tiled_peptide_dataset = hlathena.tile_peptides(fasta_path, (10, 11))
        peptides = tiled_peptide_dataset.get_peptides()

        # The first 10 and 11 characters of the second gene in the fasta file.
        self.assertIn('DTEFPNFKYD', peptides)
        self.assertIn('DTEFPNFKYDT', peptides)
        # The last 10 and 11 characters of the second gene in the fasta file.
        self.assertIn('VLYLLLIDLL', peptides)
        self.assertIn('VLYLLLIDLLK', peptides)

        self.assertSetEqual({len(pep) for pep in peptides}, {10, 11})

        pep_df = tiled_peptide_dataset.pep_df

        # The two peptides at the start of the second gene only occur once in the dataset.
        self.assertEqual(len(pep_df[pep_df['pep'] == 'DTEFPNFKYD']), 1)
        self.assertEqual(len(pep_df[pep_df['pep'] == 'DTEFPNFKYDT']), 1)

        first_eleven = pep_df[pep_df['pep'] == 'DTEFPNFKYD'].iloc[0]
        self.assertEqual(first_eleven['length'], 10)
        self.assertEqual(first_eleven['start_pos'], 0)
        self.assertEqual(first_eleven['record_id'], 'TEST00000000002|TST2')

        first_twelve = pep_df[pep_df['pep'] == 'DTEFPNFKYDT'].iloc[0]
        self.assertEqual(first_twelve['length'], 11)
        self.assertEqual(first_twelve['start_pos'], 0)
        self.assertEqual(first_twelve['record_id'], 'TEST00000000002|TST2')

    def test_tile_peptides_warning(self):
        fasta_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'data/fake_fasta.fa')
        with self.assertWarnsRegex(Warning, "length outside the range"):
            hlathena.tile_peptides(fasta_path, (7,))

        with self.assertWarnsRegex(Warning, "length outside the range"):
            hlathena.tile_peptides(fasta_path, (12,))


if __name__ == '__main__':
    unittest.main()
