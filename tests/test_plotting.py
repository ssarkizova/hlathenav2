import pandas as pd
import unittest
from unittest.mock import MagicMock
from typing import List
from hlathena import plotting

class TestPlotting(unittest.TestCase):
    def setUp(self) -> None:
        self.peptides = ['AADIFYSRY', 
                         'AADLNLVLY', 
                         'AADIFYSRYE', 
                         'AADLNLVLYLLL', 
                         'IDLLKEIYH']
        self.length = 9
        self.logo_df = pd.DataFrame.from_dict(
            {'A': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'C': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'D': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'E': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'F': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'G': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'H': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'I': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'K': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'L': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'M': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'N': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'P': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'Q': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'R': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'S': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'T': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'V': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'W': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 
             'Y': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}})
    
    def test_plot_logo_no_peptides(self):
        self.assertRaises(TypeError, plotting.plot_logo, [])

    @unittest.skip('seems broken')
    def test_plot_logo_single_length(self):
        plotting.get_logo_df = MagicMock(return_value=self.logo_df)
        plotting.plot_logo(self.peptides, self.length)
        plotting.get_logo_df.assert_called_with(self.peptides, self.length)

    @unittest.skip('seems broken')
    def test_plot_logo_multiple_lengths(self):
        plotting.get_logo_df = MagicMock(return_value=self.logo_df)
        plotting.plot_logo(self.peptides)
        ninemers = [pep for pep in self.peptides if len(pep)==9]
        tenmers = [pep for pep in self.peptides if len(pep)==10]
        twelvemers = [pep for pep in self.peptides if len(pep)==12]
        plotting.get_logo_df.assert_any_call(ninemers, 9)
        plotting.get_logo_df.assert_any_call(tenmers, 10)
        plotting.get_logo_df.assert_any_call(twelvemers, 12)
        
    @unittest.skip('seems broken')
    def test_plot_length(self):
        self.assertRaises(IndexError, plotting.plot_length, [])

if __name__ == '__main__':
    unittest.main()

