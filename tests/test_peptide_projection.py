import unittest

import numpy as np
import pandas as pd

from hlathena import peptide_projection


class TestPeptideProjection(unittest.TestCase):
    def setUp(self) -> None:
        self.peptides = ['AADIFYSRY', 
                         'AADLNLVLY', 
                         'DTEFPNFKY',
                         'AADLNLVLYLLL', 
                         'IDLLKEIYH']

    def test_PCA_encode(self):
        pca_encoded = peptide_projection.PCA_encode(self.peptides, 9)
        self.assertListEqual(
            list(pca_encoded.index),
            [peptide for peptide in self.peptides if len(peptide) == 9]
        )

        # Kidera factors encode length-9 peptides in a 90-dimensional space.
        # Is this always true?
        self.assertEqual(pca_encoded.shape[1], 90)

        # The four length-9 peptides should live in a four-dimensional subspace, which should
        # be spanned by the first four principal components. So all coefficients after the
        # fourth one should be zero (up to floating point error).
        self.assertAlmostEqual(np.linalg.norm(pca_encoded.iloc[:, 4:]), 0)

    def test_get_umap_embedding(self):
        pca_encoded = peptide_projection.PCA_encode(self.peptides, 9)
        umap_embedding = peptide_projection.get_umap_embedding(pca_encoded, n_neighbors=3)

        # The UMAP embedding should be two-dimensional.
        self.assertListEqual(
            list(umap_embedding.columns),
            ['pep', 'umap_1', 'umap_2']
        )

        # There should be one row for each 9-dimensional peptide.
        self.assertListEqual(
            list(umap_embedding['pep']),
            [peptide for peptide in self.peptides if len(peptide) == 9]
        )

    def test_get_peptide_clustering(self):
        # Create a fake umap embedding where the first two peptides are near (0, 0) and
        # The second two are near (10, 10)
        fake_umap_embedding = pd.DataFrame(
            {
                'pep': ['AADIFYSRY', 'AADLNLVLY', 'DTEFPNFKY', 'IDLLKEIYH'],
                'umap_1': [1, -1, 9, 11],
                'umap_2': [-1, 1, 9, 11]
            }
        )
        embedding = peptide_projection.get_peptide_clustering(
            fake_umap_embedding,
            eps=3,
            min_samples=2
        )
        # There are four peptides. There should be four columns: the peptide, the two umap
        # coordinates, and the cluster.
        self.assertEqual(embedding.shape, (4, 4))

        # The first two peptides should be in one cluster and the last two should be in another.
        self.assertEqual(embedding.iloc[0].loc['cluster'], embedding.iloc[1].loc['cluster'])
        self.assertEqual(embedding.iloc[2].loc['cluster'], embedding.iloc[3].loc['cluster'])
        self.assertNotEqual(embedding.iloc[0].loc['cluster'], embedding.iloc[2].loc['cluster'])


if __name__ == '__main__':
    unittest.main()
