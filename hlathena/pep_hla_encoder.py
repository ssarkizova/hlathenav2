""" Featurized peptide hLA dataset """

import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder  # Peptide encoding
import torch
import torch.nn.functional as F
from torch import Tensor
import importlib_resources

from hlathena.amino_acid_feature_map import AminoAcidFeatureMap
from hlathena.definitions import AMINO_ACIDS, AMINO_ACIDS_EXT, LOCI


class PepHLAEncoder:  # TODO: add support for no hla encoding, if none, just return pep + pep feats tensor

    def __init__(self,
                 pep_lens: List[int],
                 # maxlen: int = 11,
                 # pep_length: int,
                 aa_feature_files: List[os.PathLike] = None,
                 hla_encoding_file: os.PathLike = None,
                 allele_col_name: str = 'mhc',
                 is_pan_allele: bool = True):

        if hla_encoding_file is None:
            hla_encoding_file = importlib_resources.files('hlathena').joinpath('data').joinpath('hla_seqs_onehot.csv')

        self.hla_encoding = pd.read_csv(hla_encoding_file, index_col=allele_col_name)
        # self.pep_len = pep_length
        # self.maxlen = maxlen
        self.pep_lens = pep_lens
        self.is_pan_allele = is_pan_allele

        self.aa_feature_map = AminoAcidFeatureMap(aa_feature_files)

        pep_dim = ((self.aa_feature_map.aa_feature_count * max(self.pep_lens)) +
                                   (max(self.pep_lens) * len(AMINO_ACIDS_EXT)))
        hla_dim = len(self.hla_encoding.columns) + len(LOCI) if self.is_pan_allele else 0 # also appending number of loci categories
        pep_len_dim = len(self.pep_lens) if len(self.pep_lens) > 1 else 0
        self.feature_dimensions = pep_dim + hla_dim + pep_len_dim

    def encode_pep_len(self, pep: str):
        return F.one_hot(torch.tensor(len(pep)) % min(self.pep_lens), num_classes=len(self.pep_lens))

    def encode_peptide(self, pep):

        pep += (max(self.pep_lens) - len(pep)) * '-'

        peptide = [list(pep)]
        encoder = OneHotEncoder(
            categories=[AMINO_ACIDS_EXT] * len(pep))
        encoder.fit(peptide)
        encoded = encoder.transform(peptide).toarray()[0]

        onehot_only = not self.aa_feature_map.aa_feature_files
        # Add additional encoding features if present
        if not onehot_only:
            aafeatmat_bd: np.ndarray = np.kron(np.eye(len(pep), dtype=int), self.aa_feature_map.aa_feature_map)
            encoded = np.concatenate((encoded, (encoded @ aafeatmat_bd)))
        return torch.as_tensor(encoded).float()

    def get_loci(self, allele):
        if 'HLA-' in allele:
            return allele[4]
        else:
            return allele[0]

    def encode_loci(self, loci):
        encoder = OneHotEncoder(categories=[LOCI])
        encoder.fit([[loci]])
        return torch.as_tensor(encoder.transform([[loci]]).toarray()[0]).float()

    def encode_hla(self, hla):
        loc_enc = self.encode_loci(self.get_loci(hla)) # encoding allele loci
        return torch.cat((loc_enc, torch.as_tensor(np.array(self.hla_encoding.loc[hla])).float()))

    def encode_pepfeats(self, pep_feats):
        return torch.as_tensor(pep_feats)

    def encode(self, pep, hla, pep_feats):
        encoding = torch.cat((self.encode_peptide(pep), self.encode_pepfeats(pep_feats)))
        if self.is_pan_allele:
            encoding = torch.cat((encoding, self.encode_hla(hla)))
        if len(self.pep_lens) > 1:
            encoding = torch.cat((encoding, self.encode_pep_len(pep)))
        return encoding
