""" Featurized peptide hLA dataset """

import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder  # Peptide encoding
import torch
from torch import Tensor
import importlib_resources

from hlathena.amino_acid_feature_map import AminoAcidFeatureMap
from hlathena.definitions import AMINO_ACIDS


class PepHLAEncoder:  # TODO: add support for no hla encoding, if none, just return pep + pep feats tensor

    def __init__(self,
                 pep_length: int,
                 aa_feature_files: List[os.PathLike] = None,
                 hla_encoding_file: os.PathLike = None,
                 allele_col_name: str = 'mhc'):

        if hla_encoding_file is None:
            hla_encoding_file = importlib_resources.files('hlathena').joinpath('data').joinpath('hla_seqs_onehot.csv')

        self.hla_encoding = pd.read_csv(hla_encoding_file, index_col=allele_col_name)
        self.pep_len = pep_length

        self.aa_feature_map = AminoAcidFeatureMap(aa_feature_files)

        self.pep_feature_dim = (self.aa_feature_map.aa_feature_count * self.pep_len) + (self.pep_len * len(AMINO_ACIDS))
        self.hla_feature_dim = len(self.hla_encoding.columns)
        self.feature_dimensions = self.pep_feature_dim + self.hla_feature_dim

    def encode_peptide(self, pep):
        peptide = [list(pep)]
        encoder = OneHotEncoder(
            categories=[AMINO_ACIDS] * self.pep_len)
        encoder.fit(peptide)
        encoded = encoder.transform(peptide).toarray()[0]

        onehot_only = not self.aa_feature_map.aa_feature_files

        # Add additional encoding features if present
        if not onehot_only:
            aafeatmat_bd: np.ndarray = np.kron(np.eye(self.pep_len, dtype=int), self.aa_feature_map.aa_feature_map)
            encoded = np.concatenate((encoded, (encoded @ aafeatmat_bd)))

        return encoded

    def encode_hla(self, hla):
        return np.array(self.hla_encoding.loc[hla])

    def encode(self, pep, hla, pep_feats):
        encoding = np.hstack((self.encode_peptide(pep), self.encode_hla(hla)))
        if pep_feats.size > 0:
            encoding = np.concatenate((encoding, pep_feats))
        return torch.as_tensor(encoding).float()
