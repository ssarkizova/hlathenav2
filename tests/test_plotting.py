import unittest
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection

import pandas as pd

from hlathena import peptide_projection, plotting

class TestPlotting(unittest.TestCase):
    def setUp(self) -> None:
        self.peptides = ['AADIFYSRY', 
                         'AADLNLVLY', 
                         'AADIFYSRYE', 
                         'AADLNLVLYLLL', 
                         'IDLLKEIYH']
        self.labels = [0, 0, 0, 1, 1]
        self.peptide_df = pd.DataFrame({'seq': self.peptides,
                                        'label': self.labels})

        self.umap_peptide_df = pd.DataFrame({
            'pep': ['AADIFYSRY', 'AADLNLVLY', 'DTEFPNFKY', 'IDLLKEIYH'],
            'umap_1': [1, -1, 9, 11],
            'umap_2': [-1, 1, 9, 11],
            'cluster': [0, 0, 1, 1],
            'label': ['A', 'A', 'A', 'B']
        })

        self.length = 9

    def test_plot_logo_no_peptides(self):
        self.assertRaises(IndexError, plotting.plot_logo, self.peptide_df[0:0])

    def test_plot_logo(self):
        axes_list = plotting.plot_logo(self.peptide_df)
        self.assertEqual(len(axes_list), 3)  # There should be three subplots for the three lengths

    def test_plot_logo_with_length(self):
        axes_list = plotting.plot_logo(self.peptide_df, length=9)
        self.assertEqual(len(axes_list), 1)  # Just one plot
        self.assertEqual(axes_list[0].get_title(), "Length 9 (n=3)")

    def test_plot_logo_with_label(self):
        axes_list = plotting.plot_logo(self.peptide_df, length=9, label_col='label')
        self.assertEqual(len(axes_list), 2)  # One plot for each label.

    def test_plot_logo_with_only_one_length(self):
        # Should work whether or not the length parameter is specified.
        axes_list = plotting.plot_logo(self.peptide_df[self.peptide_df['seq'].str.len() == 9])
        self.assertEqual(len(axes_list), 1)

        axes_list = plotting.plot_logo(self.peptide_df[self.peptide_df['seq'].str.len() == 9], length=9)
        self.assertEqual(len(axes_list), 1)

    def test_plot_length_no_peptides(self):
        self.assertRaises(IndexError, plotting.plot_length, self.peptide_df[0:0])

    def test_plot_length(self):
        ax = plotting.plot_length(self.peptide_df)

        # There are three lengths of peptides: 9, 10, and 12.
        self.assertEqual(len(ax.patches), 3)

        # There are three peptides of length 9.
        self.assertEqual(ax.get_xticklabels()[0].get_text(), '9')
        self.assertEqual(ax.patches[0].get_height(), 3)

        # There is one peptide of length 10.
        self.assertEqual(ax.get_xticklabels()[1].get_text(), '10')
        self.assertEqual(ax.patches[1].get_height(), 1)

        # There is one peptide of length 12.
        self.assertEqual(ax.get_xticklabels()[2].get_text(), '12')
        self.assertEqual(ax.patches[2].get_height(), 1)

    def test_plot_length_with_label(self):
        ax = plotting.plot_length(self.peptide_df, label_col='label')

        # Six combinations of length, label
        self.assertEqual(len(ax.patches), 6)

        # The x-axis is labeled by length.
        self.assertEqual(ax.get_xticklabels()[0].get_text(), '9')
        self.assertEqual(ax.get_xticklabels()[1].get_text(), '10')
        self.assertEqual(ax.get_xticklabels()[2].get_text(), '12')

        # There are two peptides of length 9 with label 0. There is one of length 9 with label 1.
        # There is one of length 10 with label 0 and one of length 12 with label 1.
        # There are none of length 10 with label 1 or length 12 with label 0.
        # So the bar heights should be 0, 0, 1, 1, 1, 2 (in some order).
        bar_heights = sorted([patch.get_height() for patch in ax.patches])
        self.assertListEqual(bar_heights, [0, 0, 1, 1, 1, 2])

    def test_plot_umap(self):
        ax = plotting.plot_umap(self.umap_peptide_df)

        # Should be one set of axes for non-clustered umap.
        self.assertIsInstance(ax, Axes)

        # Scatter plots are represented internally as PathCollections, so the axes should contain one.
        self.assertTrue(any([isinstance(child, PathCollection) for child in ax.get_children()]))
        
    def test_plot_clustered_umap(self):
        axes_tuple = plotting.plot_umap(self.umap_peptide_df, clustered=True)

        # There should be two plots.
        self.assertIsInstance(axes_tuple, tuple)
        self.assertEqual(len(axes_tuple), 2)

        ax0, ax1 = axes_tuple
        # The first set of axes should contain a scatterplot.
        # Scatter plots are represented internally as PathCollections, so the axes should contain one.
        self.assertTrue(any([isinstance(child, PathCollection) for child in ax0.get_children()]))
        # assert False

        # The second set of axes should contain a bar plot with one bar for each cluster (of which there are two).
        self.assertEqual(len(ax1.patches), 2)

    def test_plot_clustered_umap_with_labels(self):
        axes_tuple = plotting.plot_umap(self.umap_peptide_df, clustered=True, label_col='label')

        # There should be two plots.
        self.assertIsInstance(axes_tuple, tuple)
        self.assertEqual(len(axes_tuple), 2)

        ax0, ax1 = axes_tuple
        # The first set of axes should contain a scatterplot.
        # Scatter plots are represented internally as PathCollections, so the axes should contain one.
        self.assertTrue(any([isinstance(child, PathCollection) for child in ax0.get_children()]))
        # assert False

        # The second set of axes should contain a bar plot.
        # There is one bar for each nonempty (cluster, label) pair, and two for the legend, giving five in total.
        self.assertEqual(len(ax1.patches), 5)


if __name__ == '__main__':
    unittest.main()

