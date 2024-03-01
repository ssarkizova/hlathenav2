import os
import unittest

from hlathena import predict


class TestPlotting(unittest.TestCase):
    def test_predict_no_peptides(self):
        self.assertRaises(IndexError, predict, "path", [])

    def test_predict(self):
        peptides = ['AADIFYSRY',
                    'AADLNLVLY',
                    'DTEFPNFKY',
                    'IDLLKEIYH']
        model_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '../models/NN-time2022-12-16_10-fold0.pt'
        ))
        pred_df = predict(model_path, peptides)
        # There should be one row for each peptide.
        # There should be two columns for each peptide, seq and score.
        self.assertEqual(pred_df.shape, (len(peptides), 2))
        self.assertListEqual(list(pred_df.columns), ['seq', 'score'])
        self.assertListEqual(list(pred_df['seq']), peptides)
