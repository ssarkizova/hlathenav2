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

    # Number of feature columns
    feature_count: int = 0

    def __init__(self, featurefiles: List[os.PathLike]=None):
        """Inits AminoAcidFeatureMap with features included in files

        Args:
            featurefiles (list[os.PathLike]): list of amino acid feature matrix files
        """
        self.feature_files: List[os.PathLike] = \
                [] if featurefiles is None else featurefiles # List of feature file paths
        self.feature_map: pd.DataFrame = self.encode_aa_featmap() # Feature map dataframe
        self._set_feature_count()


    def encode_aa_featmap(self) -> pd.DataFrame:
        """Creates amino acid feature map from class' feature files

        Returns:
            pd.DataFrame:  amino acid features mapping
        """
        if not self.feature_files:
            return pd.DataFrame()

        # joining files with concat and read_csv
        feature_dfs = [pd.read_csv(f, sep=' ', header=0) for f in self.feature_files]
        aa_feature_df = pd.concat(feature_dfs,axis=1,join='inner')
        # Ensure the rows have the same order as the onehot encoding
        # This enables efficient transfprmation to other encodings
        # by multiplication (below).
        return aa_feature_df.loc[AMINO_ACIDS,:]

    def get_feature_count(self) -> int:
        """Return amino acid feature count
        """
        return self.feature_count

    def _set_feature_count(self) -> None:
        """Set class' feature count property
        """
        if not self.feature_files:
            self.feature_count = len(AMINO_ACIDS)
        else:
            self.feature_count = self.feature_map.shape[1]
