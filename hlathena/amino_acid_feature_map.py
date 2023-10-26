"""Amino acid feature map module"""
import os
from typing import List
import pandas as pd

from hlathena.definitions import AMINO_ACIDS


class AminoAcidFeatureMap:
    """
    Creates an amino acid feature mapping from a list of amino acid property tables.

    Attributes:
        feature_files (List[os.PathLike]): list of paths to amino acid feature tables
        feature_map (AminoAcidFeatureMap): pandas dataframe of amino acid feature properties
        feature_count (int): integer count of amino acid properties in the aa feature map

    """

    def __init__(self, 
                 aa_feature_files: List[os.PathLike]=None):
        """Inits AminoAcidFeatureMap with amino acid level features included in files.
        File format: One row per amino acid, include index column; 
                     one amino acid properties per colum, include header
                     space-delimited

        Args:
            aa_feature_files (list[os.PathLike]): list of amino acid feature matrix files
        """
        self.aa_feature_files: List[os.PathLike] = \
                [] if aa_feature_files is None else aa_feature_files
        self.aa_feature_map: pd.DataFrame = self._set_aa_feature_map()
        self.aa_feature_count: int = self._set_aa_feature_count()
        

    def _set_aa_feature_map(self) -> pd.DataFrame:
        """Creates amino acid feature map from class' feature files

        Returns:
            pd.DataFrame:  amino acid features mapping
        """
        if not self.aa_feature_files:
            return pd.DataFrame()
        
        # Read each amino acid properties file
        feature_dfs = [pd.read_csv(f, sep=' ', header=0) for f in self.aa_feature_files]
        # Join amino acid properties files (ensure consistent amino acid ordering)
        # TODO: need to handle properties with overlapping names, e.g. similarities s.a. PMBEC and Blosum? input files as dict with name as prefix for each col?
        aa_feature_df = pd.concat(feature_dfs, axis=1, join='inner')
        # Ensure the rows have the same order as the onehot encoding        
        aa_feature_df = aa_feature_df.loc[AMINO_ACIDS,:]

        return aa_feature_df
    

    def get_aa_feature_map(self) -> pd.DataFrame:
        """Return the amino acid feature map

        Returns:
            pd.DataFrame:  amino acid features
        """
        return self.aa_feature_map
    
    
    def _set_aa_feature_count(self) -> int:
        """Set amino acid feature count
        """
        return self.aa_feature_map.shape[1]


    def get_aa_feature_count(self) -> int:
        """Return amino acid feature count
        """
        return self.aa_feature_count

    
