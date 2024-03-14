import os
import unittest

import pandas as pd

from hlathena import annotate, PeptideDataset

class TestAnnotate(unittest.TestCase):
    def setUp(self) -> None:
        self.peptides = ['AADIFYSRY', 
                         'AADLNLVLY', 
                         'DTEFPNFKY',
                         'AADLNLVLYLLL', 
                         'IDLLKEIYH']
        self.labels = [0, 0, 0, 1, 1]
        self.peptide_df = pd.DataFrame({'pep': self.peptides,
                                        'label': self.labels,
                                        'Hugo': ['RPS4Y2', 'RPS4Y2', 'RPS4Y2', 'DAZ1', 'DAZ1']})
        self.peptide_dataset = PeptideDataset(self.peptide_df)

    def test_list_tcga_expression_references(self):
        tcga_expression_references = annotate.list_tcga_expression_references()
        self.assertEqual(tcga_expression_references.shape[1], 1)  # Only one column

        self.assertEqual(tcga_expression_references.index.name, 'Study Abbreviation')
        self.assertEqual(tcga_expression_references.columns[0], 'Study Name')

        self.assertEqual(tcga_expression_references.loc['THYM'].iloc[0], 'Thymoma')

    def test_get_reference_gene_ids(self):
        ref_fasta = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'data/fake_fasta.fa')
        reference_df = annotate.get_reference_gene_ids(self.peptide_dataset, ref_fasta=ref_fasta).pep_df
        # The peptide 'AADIFYSRY' occurs in genes TST1, TST2 and TST3 in the fake fasta file.
        self.assertEqual(len(reference_df[reference_df['pep'] == 'AADIFYSRY']), 3)
        self.assertSetEqual(set(reference_df[reference_df['pep'] == 'AADIFYSRY']['Hugo_Symbol']),
                            set(['TST1', 'TST2', 'TST3']))
        # The context for 'AADIFYSRY' in TST1 is 30 Qs before and 30 Ys after.
        self.assertEqual(
            reference_df[(reference_df['pep'] == 'AADIFYSRY') &
                         (reference_df['Hugo_Symbol'] == 'TST1')]['ctex_up'].iloc[0],
            30 * 'Q'
        )
        self.assertEqual(
            reference_df[(reference_df['pep'] == 'AADIFYSRY') &
                         (reference_df['Hugo_Symbol'] == 'TST1')]['ctex_dn'].iloc[0],
            30 * 'Y'
        )
        # 'AADIFYSRY' has the same context in TST3.
        self.assertEqual(
            reference_df[(reference_df['pep'] == 'AADIFYSRY') &
                         (reference_df['Hugo_Symbol'] == 'TST3')]['ctex_up'].iloc[0],
            30 * 'Q'
        )
        self.assertEqual(
            reference_df[(reference_df['pep'] == 'AADIFYSRY') &
                         (reference_df['Hugo_Symbol'] == 'TST3')]['ctex_dn'].iloc[0],
            30 * 'Y'
        )


        # Test that context works near the beginning.
        # DTEFPNFKY occurs only in TST2. Once with empty up context and once with up context 'DTEFPNFKY'.
        self.assertEqual(len(reference_df[reference_df['pep'] == 'DTEFPNFKY']), 2)
        self.assertSetEqual(set(reference_df[reference_df['pep'] == 'DTEFPNFKY']['Hugo_Symbol']),
                            set(['TST2']))
        self.assertSetEqual(
            set(reference_df[(reference_df['pep'] == 'DTEFPNFKY') &
                             (reference_df['Hugo_Symbol'] == 'TST2')
                             ]['ctex_up']),
            set([30 * '-', (30 - len('DTEFPNFKY')) * '-' + 'DTEFPNFKY'])
        )

        # Test that context works near the end.
        # 'AADLNLVLYLLL' occurs only in TST2. The down context is 'IDLLKEIYH'.
        self.assertEqual(len(reference_df[reference_df['pep'] == 'AADLNLVLYLLL']), 1)
        self.assertSetEqual(set(reference_df[reference_df['pep'] == 'AADLNLVLYLLL']['Hugo_Symbol']),
                            set(['TST2']))
        self.assertEqual(
            reference_df[(reference_df['pep'] == 'AADLNLVLYLLL') &
                         (reference_df['Hugo_Symbol'] == 'TST2')
                         ]['ctex_dn'].iloc[0],
            'IDLLKEIYH' + (30 - len('IDLLKEIYH')) * '-'
        )

        # 'IDLLKEIYH' occurs only in TST2. The down context is empty.
        self.assertEqual(len(reference_df[reference_df['pep'] == 'IDLLKEIYH']), 1)
        self.assertSetEqual(set(reference_df[reference_df['pep'] == 'IDLLKEIYH']['Hugo_Symbol']),
                            set(['TST2']))
        self.assertEqual(
            reference_df[(reference_df['pep'] == 'IDLLKEIYH') &
                         (reference_df['Hugo_Symbol'] == 'TST2')
                         ]['ctex_dn'].iloc[0],
            30 * '-'
        )

    def test_add_tcga_expression(self):
        annotated_df = annotate.add_tcga_expression(self.peptide_dataset, 'THYM', 'Hugo').pep_df
        # RPS4Y2 occurs in the THYM table, but DAZ1 does not, so only the first three peptides should be present.
        self.assertListEqual(list(annotated_df['pep']), ['AADIFYSRY', 'AADLNLVLY', 'DTEFPNFKY'])
        # The value for THYM_TPM should be the same for these three peptides.
        self.assertEqual(len(set(annotated_df['THYM_TPM'])), 1)


if __name__ == '__main__':
    unittest.main()

